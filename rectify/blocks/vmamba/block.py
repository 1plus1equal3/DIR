import torch
import torch.nn as nn
import math
from .util import CrossScanF, CrossMergeF, selective_scan_torch

class SimpleSS2D(nn.Module):
    """
    Simplified SS2D implementation using the provided functions:
    - CrossScanF: custom autograd function for cross scanning
    - CrossMergeF: custom autograd function for merging scans
    - selective_scan_torch: SSM core operation
    """
    def __init__(
        self,
        d_model=96,      # input channel dimension
        d_state=16,      # SSM state dimension
        d_inner=192,     # inner dimension (typically 2*d_model)
        dt_rank=6,       # rank of delta projection (d_model/16)
        d_conv=3,        # depthwise conv kernel size
        scan_mode=0,     # 0: cross2d, 1: unidi, 2: bidi, 3: rot90
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_inner
        self.dt_rank = dt_rank
        self.k_group = 4  # 4 scanning directions
        self.scan_mode = scan_mode
        
        # Step 1: Input projection (d_model -> 2*d_inner for x and z)
        self.in_proj = nn.Linear(d_model, 2 * d_inner)
        
        # Step 2: Depthwise convolution for local features
        self.conv2d = nn.Conv2d(
            d_inner, d_inner, 
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            groups=d_inner,  # depthwise
        )
        
        # Step 3: Project to SSM parameters
        # For K directions, each needs: dt_rank + 2*d_state
        # Input will be K*d_inner (from all scanning directions)
        param_dim = dt_rank + 2 * d_state
        self.x_proj = nn.Linear(self.k_group * d_inner, self.k_group * param_dim)
        
        # Step 4: Project dt from rank to full dimension for all K directions
        self.dt_proj = nn.Linear(self.k_group * dt_rank, self.k_group * d_inner)
        
        # Step 5: SSM parameters (learnable)
        # A: state transition matrix (K*d_inner, d_state)
        self.A_log = nn.Parameter(torch.randn(self.k_group * d_inner, d_state))
        # D: skip connection weight (K*d_inner,)
        self.D = nn.Parameter(torch.ones(self.k_group * d_inner))
        # dt_bias: bias for delta (K*d_inner,)
        self.dt_bias = nn.Parameter(torch.randn(self.k_group * d_inner))
        
        # Step 6: Output projection
        self.out_proj = nn.Linear(d_inner, d_model)
        
        # Activations
        self.act = nn.SiLU()
        self.norm = nn.LayerNorm(d_inner)
        
    def forward(self, x):
        """
        Full forward pass using provided functions.
        Input: x (B, H, W, C) - channel last format
        Output: (B, H, W, C)
        """
        B, H, W, C_in = x.shape
        assert C_in == self.d_model
        L = H * W  # sequence length
        K = self.k_group
        D = self.d_inner
        N = self.d_state
        R = self.dt_rank
        
        # ========== STEP 1: Input Projection ==========
        # (B, H, W, d_model) -> (B, H, W, 2*d_inner)
        x = self.in_proj(x)

        # Split into x and z (gating mechanism - GLU style)
        x, z = x.chunk(2, dim=-1)  # Each: (B, H, W, d_inner)
        z = self.act(z)  # Apply activation to gate

        # ========== STEP 2: Depthwise Conv ==========
        # Need channel-first format for conv2d
        x = x.permute(0, 3, 1, 2)  # (B, d_inner, H, W)

        x = self.conv2d(x)  # Local feature extraction with depthwise conv
        x = self.act(x)
        
        # ========== STEP 3: Cross Scan using CrossScanF ==========
        # CrossScanF.apply(x, in_channel_first, out_channel_first, one_by_one, scans)
        # Input: (B, C, H, W) if in_channel_first=True
        # Output: (B, K, C, L) if out_channel_first=True
        xs = CrossScanF.apply(
            x,                    # (B, d_inner, H, W)
            True,                 # in_channel_first=True
            True,                 # out_channel_first=True
            False,                # one_by_one=False
            self.scan_mode        # scans=0 (cross2d)
        )

        # ========== STEP 4: Generate SSM Parameters ==========
        # Reshape xs for projection: (B, K, D, L) -> (B, K*D, L)
        xs_flat = xs.view(B, K * D, L)

        # Project to get parameters for each position in sequence
        # (B, K*D, L) -> (B, L, K*D) -> (B, L, K*(R+2*N)) -> (B, K*(R+2*N), L)
        x_dbl = self.x_proj(xs_flat.transpose(1, 2))  # (B, L, K*(R+2*N))
        x_dbl = x_dbl.transpose(1, 2)  # (B, K*(R+2*N), L)

        # Reshape to separate K directions
        x_dbl = x_dbl.view(B, K, R + 2*N, L)  # (B, K, R+2*N, L)

        # Split into dt, B, C parameters
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        # dts: (B, K, R, L) - time step parameters (rank-reduced)
        # Bs: (B, K, N, L) - input-to-state projection matrices
        # Cs: (B, K, N, L) - state-to-output projection matrices
        
        # ========== STEP 5: Project dt to full dimension ==========
        # Flatten K and R dimensions for projection
        dts = dts.contiguous().view(B, K * R, L)  # (B, K*R, L)

        # Project: (B, K*R, L) -> (B, L, K*R) -> (B, L, K*D) -> (B, K*D, L)
        dts = self.dt_proj(dts.transpose(1, 2))  # (B, L, K*D)
        dts = dts.transpose(1, 2).contiguous()    # (B, K*D, L)

        # ========== STEP 6: Expand B and C for all channels ==========
        # Currently Bs, Cs are (B, K, N, L)
        # Need to expand to (B, K, D, N, L) then reshape to (B, K*D, N, L)
        # This repeats the same B, C matrices for all D channels in each direction
        Bs = Bs.unsqueeze(2).repeat(1, 1, D, 1, 1)  # (B, K, D, N, L)
        Cs = Cs.unsqueeze(2).repeat(1, 1, D, 1, 1)  # (B, K, D, N, L)
        Bs = Bs.contiguous().view(B, K * D, N, L)
        Cs = Cs.contiguous().view(B, K * D, N, L)

        # ========== STEP 7: Prepare SSM Matrices ==========
        # A: (K*D, N) - state transition, negative exponential for stability
        As = -torch.exp(self.A_log)

        # D: (K*D,) - skip connection weights
        Ds = self.D

        # delta_bias: (K*D,) - bias for time steps
        delta_bias = self.dt_bias

        # ========== STEP 8: Selective Scan (Core SSM) ==========
        ys = selective_scan_torch(
            u=xs_flat,           # (B, K*D, L)
            delta=dts,           # (B, K*D, L)
            A=As,                # (K*D, N)
            B=Bs,                # (B, K, N, L) - will be expanded inside
            C=Cs,                # (B, K, N, L) - will be expanded inside
            D=Ds,                # (K*D,)
            delta_bias=delta_bias,  # (K*D,)
            delta_softplus=True,
            oflex=True,
        )

        # ========== STEP 9: Reshape for Cross Merge ==========
        # (B, K*D, L) -> (B, K, D, H, W)
        ys = ys.view(B, K, D, H, W)
        
        # ========== STEP 10: Cross Merge using CrossMergeF ==========
        # CrossMergeF.apply(ys, in_channel_first, out_channel_first, one_by_one, scans)
        # Input: (B, K, D, H, W) if out_channel_first=True
        # Output: (B, D, L) if in_channel_first=True (3D, needs reshape)
        y = CrossMergeF.apply(
            ys,                   # (B, K, D, H, W)
            True,                 # in_channel_first=True
            True,                 # out_channel_first=True
            False,                # one_by_one=False
            self.scan_mode        # scans=0 (cross2d)
        )
        # Reshape: (B, D, L) -> (B, D, H, W)
        y = y.view(B, D, H, W)

        # ========== STEP 11: Post-processing ==========
        # Convert back to channel-last for normalization
        y = y.permute(0, 2, 3, 1)  # (B, D, H, W) -> (B, H, W, D)

        # Apply layer normalization
        y = self.norm(y)

        # ========== STEP 12: Apply Gating (GLU) ==========
        # Element-wise multiplication with gate z
        y = y * z  # (B, H, W, d_inner)

        # ========== STEP 13: Output Projection ==========
        out = self.out_proj(y)  # (B, H, W, d_model)

        return out


class VMambaBlock(nn.Module):
    """
    Config-compatible wrapper for SimpleSS2D (VMamba block).

    Adapts SimpleSS2D to work with the config-driven model builder:
    - Maps in_channels/out_channels to d_model
    - Handles channel format conversion (channel-first <-> channel-last)
    - Provides consistent interface with other blocks (ConvBlock, etc.)

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (for optional projection)
        d_state: SSM state dimension (default: 16)
        d_inner_multiplier: Multiplier for inner dimension (d_inner = d_model * multiplier)
        dt_rank: Rank of delta projection. Can be int or 'auto' (d_model // 16)
        d_conv: Depthwise conv kernel size (default: 3)
        scan_mode: Scanning mode (0: cross2d, 1: unidi, 2: bidi, 3: rot90)

    Input: (B, C, H, W) channel-first format
    Output: (B, C', H, W) channel-first format (C' = out_channels)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        d_state: int = 16,
        d_inner_multiplier: float = 2.0,
        dt_rank = 'auto',
        d_conv: int = 3,
        scan_mode: int = 0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Calculate dt_rank if 'auto'
        if dt_rank == 'auto':
            dt_rank = max(1, in_channels // 16)

        # Calculate d_inner
        d_inner = int(in_channels * d_inner_multiplier)

        # Core VMamba block (SimpleSS2D)
        self.vmamba = SimpleSS2D(
            d_model=in_channels,
            d_state=d_state,
            d_inner=d_inner,
            dt_rank=dt_rank,
            d_conv=d_conv,
            scan_mode=scan_mode
        )

        # Optional projection if in_channels != out_channels
        if in_channels != out_channels:
            self.out_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.out_proj = nn.Identity()

    def forward(self, x):
        """
        Forward pass with channel format conversion.

        Args:
            x: Input tensor (B, C, H, W) channel-first format

        Returns:
            Output tensor (B, C', H, W) channel-first format
        """
        # Convert channel-first to channel-last for VMamba
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)

        # VMamba processing (operates on channel-last)
        x = self.vmamba(x)  # (B, H, W, C)

        # Convert back to channel-first for compatibility with other blocks
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)

        # Optional output projection
        x = self.out_proj(x)

        return x