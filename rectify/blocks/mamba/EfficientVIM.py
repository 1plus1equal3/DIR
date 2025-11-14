import torch
import torch.nn as nn
from ..convolution import ConvLayer1D, SimpleConvBlock
from ..mlp import FullyConnectedFFN_v2, VMoEBlock

class LayerNorm2D(nn.Module):
    """LayerNorm for channels of 2D tensor(B C H W)"""
    def __init__(self, num_channels, eps=1e-5, affine=True):
        super(LayerNorm2D, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        var = x.var(dim=1, keepdim=True, unbiased=False)  # (B, 1, H, W)

        x_normalized = (x - mean) / torch.sqrt(var + self.eps)  # (B, C, H, W)

        if self.affine:
            x_normalized = x_normalized * self.weight + self.bias

        return x_normalized


class LayerNorm1D(nn.Module):
    """LayerNorm for channels of 1D tensor(B C L)"""
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
        mean = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        var = x.var(dim=1, keepdim=True, unbiased=False)  # (B, 1, H, W)

        x_normalized = (x - mean) / torch.sqrt(var + self.eps)  # (B, C, H, W)

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
        batch_size, _, _ = x.shape
        BCdt = self.dw_conv(self.BCdt_proj(x).view(batch_size, -1, H, W)).flatten(2)
        B, C, dt = torch.split(BCdt, [self.state_dim, self.state_dim, self.state_dim], dim=1)
        A = (dt + self.A.view(1, -1, 1)).softmax(dim=-1)

        AB = (A * B)
        h = x @ AB.transpose(-2, -1)

        h, z = torch.split(self.hz_proj(h), [self.d_inner, self.d_inner], dim=1)
        h = self.out_proj(h * self.act(z) + h * self.D)
        y = h @ C

        y = y.view(batch_size, -1, H, W).contiguous()
        return y, h

class EfficientVIMBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4, ssd_expand=1, state_dim=64, ffn_type='mlp', ffn_config=None):
        super(EfficientVIMBlock, self).__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio

        self.mixer = HSMSSD(d_model=dim, ssd_expand=ssd_expand, state_dim=state_dim)
        self.norm = LayerNorm1D(dim)

        self.dw_conv_1 = SimpleConvBlock(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False, use_norm=True, use_act=False)
        self.dw_conv_2 = SimpleConvBlock(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False, use_norm=True, use_act=False)

        if ffn_type == 'mlp':
            self.ffn = FullyConnectedFFN_v2(dim, int(dim * mlp_ratio), dim)
        elif ffn_type == 'vmoe':
            self.ffn = VMoEBlock(input_dim=dim, output_dim=dim, **ffn_config)
        elif ffn_type is None:
            self.ffn = nn.Identity()
        else:
            raise ValueError(f"Unsupported ffn_type: {ffn_type}")
        
        self.alpha = nn.Parameter(1e-4 * torch.ones(4, dim), requires_grad=True)

    def forward(self, x):
        alpha = torch.sigmoid(self.alpha).view(4, -1, 1, 1)

        x = (1 - alpha[0]) * x + alpha[0] * self.dw_conv_1(x)

        x_prev = x
        _, _, H, W = x.shape
        x, h = self.mixer(self.norm(x.flatten(2)), H, W)
        x = (1 - alpha[1]) * x_prev + alpha[1] * x

        x = (1 - alpha[2]) * x + alpha[2] * self.dw_conv_2(x)

        x_ffn = self.ffn(x)

        if isinstance(x_ffn, tuple):
            x_ffn, aux_loss = x_ffn
            x = (1 - alpha[3]) * x + alpha[3] * x_ffn
            return x, aux_loss
        else:
            x = (1 - alpha[3]) * x + alpha[3] * x_ffn
            return x

# Example usage
if __name__ == "__main__":
    layer = EfficientVIMBlock(dim=64, mlp_ratio=4.0, ssd_expand=1, state_dim=64, ffn_type='vmoe', ffn_config={'num_experts':4, 'top_k':2, 'capacity_factor':1.0, 'expert_dim':256, 'return_shape':'2d'})
    sample_input = torch.randn(1, 64, 32, 32)
    output, aux_loss = layer(sample_input)
    print(output.shape)
