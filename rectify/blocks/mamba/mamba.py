import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from dataclasses import dataclass
import math
from einops import rearrange, repeat, einsum
from .pscan import pscan

@dataclass
class MambaConfig:
    d_model: int # hidden dim
    d_state: int = 16 # latent state dim
    expand: int = 4 # expansion factor
    dt_rank: Union[int, str] = "auto" # rank of Δ
    d_conv: int = 4 # kernel size of convolution
    bias: bool = False # use bias in linear layers
    use_pscan: bool = True # use pscan for parallel computational

    def __post_init__(self):
        self.d_inner = self.d_model * self.expand # inner dim
        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)

class MambaBlock(nn.Module):
    def __init__(self, cfg: MambaConfig):
        super().__init__()
        self.cfg = cfg

        self.in_proj = nn.Linear(cfg.d_model, cfg.d_inner * 2, bias=cfg.bias)

        self.conv1d = nn.Conv1d(
            in_channels=cfg.d_inner,
            out_channels=cfg.d_inner,
            kernel_size=cfg.d_conv,
            groups=cfg.d_inner,
            padding=cfg.d_conv - 1,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_prj = nn.Linear(cfg.d_inner, cfg.dt_rank + cfg.d_state * 2, bias=cfg.bias)

        # dt_proj projects Δ from dt_rank to d_in
        self.dt_prj = nn.Linear(cfg.dt_rank, cfg.d_inner, bias=cfg.bias)

        A = repeat(torch.arange(1, cfg.d_state + 1), 'n -> d n', d=cfg.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(cfg.d_inner))
        self.out_proj = nn.Linear(cfg.d_inner, cfg.d_model, bias=cfg.bias)

    def forward(self, x):
        """ 
        x: (b, l, d_model)
        """
        (b, l, d) = x.shape
        x = self.in_proj(x) # (b, l, d_in * 2)
        x, res = x.chunk(2, dim=-1) # (b, l, d_in), (b, l, d_in)

        x = rearrange(x, 'b l d -> b d l') # (b, d_in, l)
        x = self.conv1d(x)[:, :, :l] # (b, d_in, l) # Causal conv with padding (remove extra padding by limiting to :l)
        x = rearrange(x, 'b d l -> b l d')

        x = F.silu(x)
        y = self.ssm(x) # (b, l, d_in)

        y = y * F.silu(res) # (b, l, d_in)

        return self.out_proj(y) # (b, l, d_model)

    def ssm(self, x):
        """ 
        x: (b, l, d_model)
        """
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        A = -torch.exp(self.A_log.float()) # (d_in, n)
        D = self.D.float() # (d_in,)

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
        if self.cfg.use_pscan:
            y = self.pscan_selective_scan(x, delta, A, B, C, D)  # (b, l, d_in)
        else:
            y = self.selective_scan(x, delta, A, B, C, D)  # (b, l, d_in)
        return y
    
    def selective_scan(self, u, delta, A, B, C, D): # Slow implementation
        """ 
        This is the classic discrete state space formula:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
        except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).
    
        Args:
            u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
            delta: shape (b, l, d_in)
            A: shape (d_in, n)
            B: shape (b, l, n)
            C: shape (b, l, n)
            D: shape (d_in,)
    
        Returns:
            output: shape (b, l, d_in)
        """
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
    
    def pscan_selective_scan(self, u, delta, A, B, C, D): # Fats implementation ?!
        """ 
        This is the classic discrete state space formula:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
        except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).
    
        Args:
            u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
            delta: shape (b, l, d_in)
            A: shape (d_in, n)
            B: shape (b, l, n)
            C: shape (b, l, n)
            D: shape (d_in,)
    
        Returns:
            output: shape (b, l, d_in)
        """

        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

        h = pscan(deltaA, deltaB_u)
        y = (h @ C.unsqueeze(-1)).squeeze(3)
        y = y + u * D # 
        return y