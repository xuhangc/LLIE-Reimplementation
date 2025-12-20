"""
RetinexMamba（ICONIP 2024）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
import numbers


# ============================================================================
# Helper functions for tensor reshaping (replacing einops where possible)
# ============================================================================

def rearrange_b_c_h_w_to_b_hw_c(x):
    """Rearrange from (B, C, H, W) to (B, H*W, C)"""
    b, c, h, w = x.shape
    return x.permute(0, 2, 3, 1).reshape(b, h * w, c)


def rearrange_b_hw_c_to_b_c_h_w(x, h, w):
    """Rearrange from (B, H*W, C) to (B, C, H, W)"""
    b, _, c = x.shape
    return x.reshape(b, h, w, c).permute(0, 3, 1, 2)


def to_3d(x):
    """Convert (B, C, H, W) to (B, H*W, C)"""
    b, c, h, w = x.shape
    return x.permute(0, 2, 3, 1).reshape(b, h * w, c)


def to_4d(x, h, w):
    """Convert (B, H*W, C) to (B, C, H, W)"""
    b, _, c = x.shape
    return x.reshape(b, h, w, c).permute(0, 3, 1, 2)


# ============================================================================
# Layer Normalization modules
# ============================================================================

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# ============================================================================
# Feed Forward Network
# ============================================================================

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


# ============================================================================
# Attention module for IFA
# ============================================================================

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape
        kv = self.kv_dwconv(self.kv(x))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(y))
        
        # Reshape for multi-head attention
        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)
        
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        
        out = out.reshape(b, c, h, w)
        out = self.project_out(out)
        return out


# ============================================================================
# IFA (Illumination Feature Attention) module
# ============================================================================

class IFA(nn.Module):
    def __init__(self, dim_2, dim, num_heads=2, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super(IFA, self).__init__()
        self.conv1 = nn.Conv2d(dim_2, dim, (1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, input_R, input_S):
        input_S = F.interpolate(input_S, [input_R.shape[2], input_R.shape[3]])
        input_S = self.conv1(input_S)
        input_R = self.norm1(input_R)
        input_S = self.norm1(input_S)
        input_R = input_R + self.attn(input_R, input_S)
        input_R = input_R + self.ffn(self.norm2(input_R))
        return input_R


# ============================================================================
# Weight initialization utilities
# ============================================================================

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


# ============================================================================
# PreNorm and GELU modules
# ============================================================================

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


# ============================================================================
# Convolution utility
# ============================================================================

def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size // 2),
        bias=bias,
        stride=stride
    )


# ============================================================================
# SS2D (Selective State Space 2D) - Pure PyTorch Implementation
# Based on Mamba-2 architecture from nd-Mamba2-torch
# ============================================================================

def segsum(x, device=None):
    """Stable segment sum calculation for SSD."""
    T = x.size(-1)
    x = x.unsqueeze(-1).expand(*x.shape, T)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def ssd_minimal_discrete(X, A, B, C, block_len, initial_states=None):
    """
    Structured State Space Duality (SSD) - the core of Mamba-2
    
    Arguments:
        X: (batch, length, n_heads, d_head)
        A: (batch, length, n_heads)
        B: (batch, length, n_heads, d_state)
        C: (batch, length, n_heads, d_state)
    Return:
        Y: (batch, length, n_heads, d_head)
        final_state: final hidden state
    """
    batch, seqlen, nheads, headdim = X.shape
    
    # Ensure sequence length is divisible by block_len
    assert seqlen % block_len == 0, f"Sequence length {seqlen} must be divisible by block_len {block_len}"
    
    # Rearrange into blocks/chunks
    num_chunks = seqlen // block_len
    X = X.reshape(batch, num_chunks, block_len, nheads, headdim)
    B = B.reshape(batch, num_chunks, block_len, B.shape[2], B.shape[3])
    C = C.reshape(batch, num_chunks, block_len, C.shape[2], C.shape[3])
    A = A.reshape(batch, num_chunks, block_len, nheads)
    A = A.permute(0, 3, 1, 2)  # (batch, nheads, num_chunks, block_len)
    
    A_cumsum = torch.cumsum(A, dim=-1)
    
    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A))
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)
    
    # 2. Compute the state for each intra-chunk
    decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)
    
    # 3. Compute the inter-chunk SSM recurrence
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]
    
    # 4. Compute state -> output conversion per chunk
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', C, states, state_decay_out)
    
    # Add output of intra-chunk and inter-chunk terms
    Y = Y_diag + Y_off
    Y = Y.reshape(batch, seqlen, nheads, headdim)
    
    return Y, final_state


class SS2D(nn.Module):
    """
    Selective State Space 2D (SS2D) module - Pure PyTorch Implementation
    
    This implements the core Mamba-2 selective state space mechanism for 2D data.
    """
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        headdim=64,
        ngroups=1,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        chunk_size=64,
        dropout=0.0,
        bias=False,
        conv_bias=True,
        device=None,
        dtype=None,
        **kwargs
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.headdim = headdim
        self.ngroups = ngroups
        self.chunk_size = chunk_size
        
        # Ensure d_inner is divisible by headdim
        if self.d_inner % self.headdim != 0:
            self.headdim = self.d_inner // max(1, self.d_inner // 64)
        
        self.nheads = self.d_inner // self.headdim
        
        # Input projection: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias)
        
        # 2D Convolution
        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv2d = nn.Conv2d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            groups=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )
        
        self.act = nn.SiLU()
        
        # Initialize dt bias
        dt = torch.exp(
            torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        
        # A parameter (log scale)
        A = torch.empty(self.nheads, dtype=torch.float32).uniform_(*A_init_range)
        self.A_log = nn.Parameter(torch.log(A))
        
        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads))
        
        # Output normalization and projection
        self.norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        """
        Forward pass for SS2D
        
        Args:
            x: Input tensor of shape (B, H, W, C) or (B, L, C)
        Returns:
            Output tensor of same shape as input
        """
        # Handle input shape
        if x.dim() == 4:
            # Input is (B, H, W, C)
            B, H, W, C = x.shape
            x = x.reshape(B, H * W, C)
            reshape_back = True
        else:
            # Input is (B, L, C)
            B, L, C = x.shape
            H = W = int(math.sqrt(L))
            reshape_back = False
        
        seqlen = x.shape[1]
        
        # Pad sequence length to be divisible by chunk_size
        pad_len = (self.chunk_size - seqlen % self.chunk_size) % self.chunk_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
        
        padded_len = x.shape[1]
        
        # Input projection
        zxbcdt = self.in_proj(x)
        
        # Split projections
        z, xBC, dt = torch.split(
            zxbcdt,
            [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        )
        
        # Apply dt bias and softplus
        dt = F.softplus(dt + self.dt_bias)
        
        # 2D Convolution
        H_pad = W_pad = int(math.sqrt(padded_len))
        xBC = xBC.reshape(B, H_pad, W_pad, -1).permute(0, 3, 1, 2).contiguous()
        xBC = self.act(self.conv2d(xBC))
        xBC = xBC.permute(0, 2, 3, 1).reshape(B, padded_len, -1).contiguous()
        
        # Split into x, B, C
        x_inner, B_state, C_state = torch.split(
            xBC,
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1
        )
        
        # Get A from log scale
        A = -torch.exp(self.A_log)
        
        # Reshape for SSD computation
        x_inner = x_inner.reshape(B, padded_len, self.nheads, self.headdim)
        B_state = B_state.reshape(B, padded_len, self.ngroups, self.d_state)
        C_state = C_state.reshape(B, padded_len, self.ngroups, self.d_state)
        
        # Expand B and C if ngroups < nheads
        if self.ngroups < self.nheads:
            expand_factor = self.nheads // self.ngroups
            B_state = B_state.unsqueeze(3).expand(-1, -1, -1, expand_factor, -1).reshape(B, padded_len, self.nheads, self.d_state)
            C_state = C_state.unsqueeze(3).expand(-1, -1, -1, expand_factor, -1).reshape(B, padded_len, self.nheads, self.d_state)
        
        # Expand A for batch
        A_expanded = A.unsqueeze(0).unsqueeze(1).expand(B, padded_len, -1)
        
        # Apply SSD
        y, _ = ssd_minimal_discrete(
            x_inner * dt.unsqueeze(-1),
            A_expanded * dt,
            B_state,
            C_state,
            self.chunk_size
        )
        
        # Add skip connection with D
        y = y + x_inner * self.D.view(1, 1, -1, 1)
        
        # Reshape output
        y = y.reshape(B, padded_len, self.d_inner)
        
        # Apply normalization and gating
        y = self.norm(y)
        y = y * F.silu(z)
        
        # Output projection
        y = self.out_proj(y)
        y = self.dropout(y)
        
        # Remove padding
        if pad_len > 0:
            y = y[:, :seqlen, :]
        
        # Reshape back if needed
        if reshape_back:
            y = y.reshape(B, H, W, -1)
        
        return y


# ============================================================================
# FeedForward for IGAB
# ============================================================================

class FeedForwardIGAB(nn.Module):
    """Feed Forward Network for IGAB module"""
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1)


# ============================================================================
# IGAB (Interleaved Group Attention Block)
# ============================================================================

class IGAB(nn.Module):
    """
    Interleaved Group Attention Block combining IFA, SS2D, and FeedForward
    """
    def __init__(self, dim, dim_head=64, heads=8, num_blocks=2, d_state=16):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                IFA(dim_2=dim, dim=dim, num_heads=heads, ffn_expansion_factor=2.66, bias=True, LayerNorm_type='WithBias'),
                SS2D(d_model=dim, dropout=0, d_state=d_state),
                PreNorm(dim, FeedForwardIGAB(dim=dim))
            ]))

    def forward(self, x, illu_fea):
        """
        Args:
            x: Input tensor (B, C, H, W)
            illu_fea: Illumination feature tensor (B, C, H, W)
        Returns:
            Output tensor (B, C, H, W)
        """
        for (trans, ss2d, ff) in self.blocks:
            # Apply IFA and permute for SS2D
            y = trans(x, illu_fea).permute(0, 2, 3, 1)  # (B, H, W, C)
            # Apply SS2D and add residual
            x = ss2d(y) + x.permute(0, 2, 3, 1)  # (B, H, W, C)
            # Apply FeedForward and add residual
            x = ff(x) + x  # (B, H, W, C)
            x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        return x


# ============================================================================
# Illumination Estimator
# ============================================================================

class Illumination_Estimator(nn.Module):
    """
    Neural network module for estimating illumination conditions in images.
    """
    def __init__(self, n_fea_middle, n_fea_in=4, n_fea_out=3):
        super(Illumination_Estimator, self).__init__()
        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)
        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        """
        Args:
            img: Input image tensor (B, C=3, H, W)
        Returns:
            illu_fea: Illumination feature map
            illu_map: Illumination map (B, C=3, H, W)
        """
        mean_c = img.mean(dim=1).unsqueeze(1)
        input = torch.cat([img, mean_c], dim=1)
        x_1 = self.conv1(input)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map


# ============================================================================
# Denoiser
# ============================================================================

class Denoiser(nn.Module):
    """
    Denoiser network using encoder-decoder architecture with IGAB blocks.
    """
    def __init__(self, in_dim=3, out_dim=3, dim=31, level=2, num_blocks=[2, 4, 4], d_state=16):
        super(Denoiser, self).__init__()
        self.dim = dim
        self.level = level

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim

        for i in range(level):
            self.encoder_layers.append(nn.ModuleList([
                IGAB(dim=dim_level, num_blocks=num_blocks[i], dim_head=dim, heads=dim_level // dim, d_state=d_state),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False)
            ]))
            dim_level *= 2
            d_state *= 2

        # Bottleneck
        self.bottleneck = IGAB(dim=dim_level, dim_head=dim, heads=dim_level // dim, num_blocks=num_blocks[-1], d_state=d_state)

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                IGAB(dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i], dim_head=dim, heads=(dim_level // 2) // dim, d_state=d_state)
            ]))
            dim_level //= 2
            d_state //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, illu_fea):
        """
        Args:
            x: Input tensor (B, 3, H, W)
            illu_fea: Illumination feature tensor
        Returns:
            Output tensor (B, 3, H, W)
        """
        fea = self.embedding(x)

        # Encoder
        fea_encoder = []
        illu_fea_list = []
        for (IGAB_block, FeaDownSample, IlluFeaDownsample) in self.encoder_layers:
            fea = IGAB_block(fea, illu_fea)
            illu_fea_list.append(illu_fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            illu_fea = IlluFeaDownsample(illu_fea)

        # Bottleneck
        fea = self.bottleneck(fea, illu_fea)

        # Decoder
        for i, (FeaUpSample, Fusion, LeWinBlock) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fusion(torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            illu_fea = illu_fea_list[self.level - 1 - i]
            fea = LeWinBlock(fea, illu_fea)

        # Output
        out = self.mapping(fea) + x
        return out


# ============================================================================
# RetinexMamba Single Stage
# ============================================================================

class RetinexMamba_Single_Stage(nn.Module):
    """
    Single stage of RetinexMamba combining illumination estimation and denoising.
    """
    def __init__(self, in_channels=3, out_channels=3, n_feat=40, level=2, num_blocks=[1, 2, 2], d_state=16):
        super(RetinexMamba_Single_Stage, self).__init__()
        self.estimator = Illumination_Estimator(n_feat)
        self.denoiser = Denoiser(in_dim=in_channels, out_dim=out_channels, dim=n_feat, level=level, num_blocks=num_blocks, d_state=d_state)

    def forward(self, img):
        """
        Args:
            img: Input image tensor (B, C, H, W)
        Returns:
            Enhanced output image tensor
        """
        illu_fea, illu_map = self.estimator(img)
        input_img = img * illu_map + img
        output_img = self.denoiser(input_img, illu_fea)
        return output_img


# ============================================================================
# RetinexMamba (Multi-stage)
# ============================================================================

class RetinexMamba(nn.Module):
    """
    Multi-stage RetinexMamba network for image enhancement.
    """
    def __init__(self, in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 2], d_state=16):
        super(RetinexMamba, self).__init__()
        self.stage = stage
        modules_body = [
            RetinexMamba_Single_Stage(
                in_channels=in_channels,
                out_channels=out_channels,
                n_feat=n_feat,
                level=2,
                num_blocks=num_blocks,
                d_state=d_state
            ) for _ in range(stage)
        ]
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        """
        Args:
            x: Input image tensor (B, C, H, W)
        Returns:
            Enhanced output image tensor
        """
        out = self.body(x)
        return out


# ============================================================================
# Main function for testing
# ============================================================================

if __name__ == '__main__':
    # Test the model
    print("Testing RetinexMamba Pure PyTorch Implementation...")
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = RetinexMamba(stage=1, n_feat=40, num_blocks=[1, 2, 2]).to(device)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    with torch.no_grad():
        # Use 256x256 input (must be divisible by chunk_size for SS2D)
        inputs = torch.randn((1, 3, 256, 256)).to(device)
        print(f"Input shape: {inputs.shape}")
        
        res = model(inputs)
        print(f"Output shape: {res.shape}")
        print("Forward pass successful!")
    
    print("\nRetinexMamba Pure PyTorch implementation is working correctly!")