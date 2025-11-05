import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Optional
from timm.layers import DropPath
from blocks.convolution import ConvLayer1D, SimpleConvBlock

class LayerNorm1D(nn.Module):
    """LayerNorm cho các kênh của tensor 1D (B C L)"""
    def __init__(self, num_channels, eps=1e-5, affine=True):
        super(LayerNorm1D, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            x_normalized = x_normalized * self.weight + self.bias
        return x_normalized
class HSMSSD(nn.Module):
    def __init__(self, d_model, ssd_expand=1, A_init_range=(1, 16), state_dim=64):
        super(HSMSSD, self).__init__()
        self.ssd_expand = ssd_expand
        self.d_inner = int(self.ssd_expand * d_model)
        self.state_dim = state_dim

        conv_dim = 3 * state_dim
        self.BCdt_proj = ConvLayer1D(d_model, conv_dim, kernel_size=1, norm=None, act_layer=None)
        self.dw_conv = SimpleConvBlock(conv_dim, conv_dim, kernel_size=3,
                                           stride=1, padding=1, groups=conv_dim, bias=True, use_norm=False, use_act=False)
        self.hz_proj = ConvLayer1D(d_model, 2 * self.d_inner, kernel_size=1, norm=None, act_layer=None)
        self.out_proj = ConvLayer1D(self.d_inner, d_model, kernel_size=1, norm=None, act_layer=None)

        A = torch.empty(self.state_dim, dtype=torch.float32).uniform_(*A_init_range)
        self.A = torch.nn.Parameter(A)
        self.act = nn.SiLU()
        self.D = nn.Parameter(torch.ones(1))
        self.D._no_weight_decay = True
    def forward(self, x, H, W):
        # x có shape (B, C, L)
        batch_size, channels, length = x.shape
        
        # (B, C, L) -> (B, 3*state_dim, L)
        BCdt_proj = self.BCdt_proj(x)
        
        # (B, 3*state_dim, L) -> (B, 3*state_dim, H, W)
        BCdt_view = BCdt_proj.view(batch_size, -1, H, W)
        
        # (B, 3*state_dim, H, W) -> (B, 3*state_dim, L)
        BCdt = self.dw_conv(BCdt_view).flatten(2)
        
        B, C, dt = torch.split(BCdt, [self.state_dim, self.state_dim, self.state_dim], dim=1)
        A = (dt + self.A.view(1, -1, 1)).softmax(dim=-1)

        AB = (A * B) # (B, state_dim, L)
        h = x @ AB.transpose(-2, -1)

        # h (B, C, state_dim) -> (B, 2*d_inner, state_dim)
        h, z = torch.split(self.hz_proj(h), [self.d_inner, self.d_inner], dim=1)
        
        # (B, d_inner, state_dim) -> (B, d_model, state_dim)
        h = self.out_proj(h * self.act(z) + h * self.D)
        
        # h (B, C, state_dim) @ C (B, state_dim, L) -> y (B, C, L)
        y = h @ C
        y = y.view(batch_size, -1, H, W).contiguous()
        return y, h

class PureTorch_SSM_Wrapper(nn.Module):
    def __init__(self, d_model, ssd_expand=1, A_init_range=(1, 16), state_dim=64):
        super().__init__()
        self.ssm_core = HSMSSD(d_model, ssd_expand, A_init_range, state_dim)

    def forward(self, x, H, W):
        #Chuyển sang channels-first: (B, L, C) -> (B, C, L)
        x_t = x.transpose(1, 2).contiguous() 
        # HSMSSD trả về y (B, C, H, W) và h
        y_hsm, _ = self.ssm_core(x_t, H, W)
        # (B, C, H, W) -> (B, C, L)
        y_flat = y_hsm.flatten(2)
        # (B, C, L) -> (B, L, C)
        y_out = y_flat.transpose(1, 2).contiguous()
        return y_out
class MambaIR_Block(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        drop_path: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm, 
        d_conv: int = 3,
        conv_bias: bool = True,
        ssd_expand=1, 
        state_dim=64,
        A_init_range=(1, 16),
        bias: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.in_proj = nn.Linear(hidden_dim, hidden_dim * 2, bias=bias)
        self.conv_branch = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            groups=hidden_dim,
            bias=conv_bias,
        )
        self.ssm_branch = PureTorch_SSM_Wrapper(
            d_model=hidden_dim, 
            ssd_expand=ssd_expand,
            state_dim=state_dim,
            A_init_range=A_init_range
        )
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.norm = norm_layer(hidden_dim) 
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        L = H * W
        #(B, C, H, W) -> (B, H, W, C)
        x_permuted = x.permute(0, 2, 3, 1).contiguous() 
        #(B, H, W, C) -> (B, H, W, 2*C)
        x_proj = self.in_proj(x_permuted)
        x_mamba, x_conv = x_proj.chunk(2, dim=-1)
        # (B, H, W, C) -> (B, C, H, W)
        x_conv = x_conv.permute(0, 3, 1, 2).contiguous()
        x_conv = self.conv_branch(x_conv)
        # (B, C, H, W) -> (B, H, W, C)
        x_conv = x_conv.permute(0, 2, 3, 1)
        # Gating
        x_conv = x_conv * F.silu(x_conv) 

        # (B, H, W, C) -> (B, L, C)
        x_mamba_flat = x_mamba.reshape(B, L, C)
        # Áp dụng LayerNorm (B, L, C)
        x_mamba_norm = self.norm(x_mamba_flat)
        # Gọi wrapper (pure torch): (B, L, C) -> (B, L, C)
        x_mamba_out = self.ssm_branch(x_mamba_norm, H, W) 
        # (B, L, C) -> (B, H, W, C)
        x_mamba_out = x_mamba_out.reshape(B, H, W, C)

        x_out = x_mamba_out + x_conv

        # (B, H, W, C)
        x_out = self.out_proj(x_out)
        
        # (B, H, W, C) -> (B, C, H, W)
        x_out = x_out.permute(0, 3, 1, 2).contiguous()

        # (B, C, H, W)
        x = x + self.drop_path(x_out)
        
        return x

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Sử dụng thiết bị: {device}")

    layer = MambaIR_Block(
        hidden_dim=64,     
        drop_path=0.1,
        state_dim=64,      
        ssd_expand=1
    ).to(device)
    sample_input = torch.randn(4, 64, 32, 32).to(device)
    print(f"Input: {sample_input.shape}")
    output = layer(sample_input)
    print(f"Output: {output.shape}")
    output.sum().backward()
    print("Check backward")