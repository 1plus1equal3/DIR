import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath
import math
from einops import rearrange, repeat, einsum
from .pscan import pscan 

class LayerNorm1D(nn.Module):
    """LayerNorm cho các kênh của tensor 1D (B C L) - (Giữ nguyên)"""
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


class MambaIR_Block(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        drop_path: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm, 
        d_conv: int = 3,
        conv_bias: bool = True, 
        state_dim=64,
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim 
        self.d_state = state_dim
        self.dt_rank = math.ceil(self.hidden_dim / 16)
        self.pscan = True 

        self.in_proj = nn.Linear(hidden_dim, hidden_dim * 2, bias=bias)
        self.conv_branch = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            groups=hidden_dim,
            bias=conv_bias,
        )
        self.dt_proj = nn.Linear(self.hidden_dim, self.dt_rank + self.d_state * 2, bias=True)
        self.dt_rank_proj = nn.Linear(self.dt_rank, self.hidden_dim, bias=True)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.hidden_dim, 1))
        )
        self.D = nn.Parameter(torch.ones(self.hidden_dim))
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

        x_conv = x_conv.permute(0, 3, 1, 2).contiguous()
        x_conv = self.conv_branch(x_conv)
        x_conv = x_conv.permute(0, 2, 3, 1)
        x_conv = x_conv * F.silu(x_conv) 

        # (B, H, W, C) -> (B, L, C)
        x_mamba_flat = x_mamba.reshape(B, L, C)
        # Áp dụng LayerNorm (B, L, C)
        x_mamba_norm = self.norm(x_mamba_flat) 
        
        A = -torch.exp(self.A_log.float()) # (C, d_state)
        D = self.D.float() # (C,)
        
        # (B, L, C) -> (B, L, dt_rank + 2*d_state)
        x_dt_B_C = self.dt_proj(x_mamba_norm)

        dt_raw, B_ssm, C_ssm = torch.split( # Đổi tên dt_ssm -> dt_raw
            x_dt_B_C,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1
        )
        dt_projected = self.dt_rank_proj(dt_raw)

        dt_ssm = F.softplus(dt_projected)  # (B, L, C)

        if self.pscan:
            x_mamba_out_flat = self.pscan_selective_scan(
                x_mamba_norm, dt_ssm, A, B_ssm, C_ssm, D
            )
        else:
            x_mamba_out_flat = self.selective_scan(
                x_mamba_norm, dt_ssm, A, B_ssm, C_ssm, D
            )

        # (B, L, C) -> (B, H, W, C)
        x_mamba_out = x_mamba_out_flat.reshape(B, H, W, C)

        x_out = x_mamba_out + x_conv

        # (B, H, W, C)
        x_out = self.out_proj(x_out)
        
        # (B, H, W, C) -> (B, C, H, W)
        x_out = x_out.permute(0, 3, 1, 2).contiguous()

        # (B, C, H, W)
        x = x + self.drop_path(x_out)
        
        return x
    
    def selective_scan(self, u, delta, A, B, C, D): 

        (b, l, d_in) = u.shape
        n = A.shape[1]

        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []    
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)
        y = y + u * D
    
        return y
    
    def pscan_selective_scan(self, u, delta, A, B, C, D):

        # (b, l, d_in) = u.shape
        # n = A.shape[1]
        
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

        h = pscan(deltaA, deltaB_u)
        y = (h @ C.unsqueeze(-1)).squeeze(3)
        y = y + u * D # 
        return y

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Sử dụng thiết bị: {device}")

    layer = MambaIR_Block(
        hidden_dim=64,    
        drop_path=0.1,
        state_dim=64
    ).to(device)
    sample_input = torch.randn(4, 64, 32, 32).to(device)
    print(f"Input: {sample_input.shape}")
    output = layer(sample_input)
    print(f"Output: {output.shape}")
    output.sum().backward()
    print("Check backward")