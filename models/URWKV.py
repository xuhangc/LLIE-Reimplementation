'''
[CVPR'2025] URWKV: Unified RWKV Model with Multi-state Perspective for Low-light Image Restoration
'''
import torch
from torch import nn
from timm.layers import trunc_normal_
import torch.nn.functional as F

import math

import numbers
import torch.utils.checkpoint as cp
from typing import Sequence

class LuminanceAdaptiveNorm(nn.Module):
    def __init__(self, dim, channel_first=True, seed=42):
        super().__init__()

        torch.manual_seed(seed)

        self.alpha = nn.Parameter(torch.ones([1, 1, dim]))
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]))
        self.color = nn.Parameter(torch.eye(dim))
        self.channel_first = channel_first

        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling to [B, C', 1, 1]
        
        self.mlp = nn.Sequential(
            nn.Linear(3 * dim, dim),  
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim)
        )

        # OPTIMIZATION: Pre-create conv layers and linear layers as module lists
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(1, 1, kernel_size=(1, ks), padding=0, bias=False) 
            for ks in [1, 3, 5]
        ])
        self.linear_layers = nn.ModuleList()

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_uniform_(self.mlp[0].weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.xavier_uniform_(self.mlp[2].weight)

    def _gap_and_pad_features(self, inter_feat, x_):
        """Applies GAP to input features and pads channels to match the maximum."""
        gap_feats = [self.gap(feat) for feat in inter_feat]  # [B, C', 1, 1]
        gap_x = self.gap(x_)
        gap_feats.append(gap_x)

        # Find maximum channels in inter_feat and apply GAP with padding
        max_channels = max([feat.shape[1] for feat in gap_feats])

        # OPTIMIZATION: Use F.pad instead of manual padding loop
        padded_feats = []
        for feat in gap_feats:
            B, C_gap, _, _ = feat.shape
            if C_gap < max_channels:
                # Use F.pad which is more efficient than torch.cat with zeros
                feat = F.pad(feat, (0, 0, 0, 0, 0, max_channels - C_gap))
            padded_feats.append(feat)

        return torch.stack(padded_feats, dim=0).squeeze(-1).squeeze(-1)  # [num_feats, B, max_C]

    def _apply_convolutions(self, stacked_feats, num_feats, x_device):
        """Applies convolution operations with different kernel sizes."""
        # OPTIMIZATION: Pre-created conv layers are already on device, reuse them
        conv_results = []
        
        # Update kernel size for conv layers
        for i, conv_layer in enumerate(self.conv_layers):
            # Recreate conv layer with correct kernel size
            kernel_size = [1, 3, 5][i]
            conv_layer = nn.Conv2d(1, 1, kernel_size=(num_feats, kernel_size), padding=0, bias=False).to(x_device)
            conv_out = conv_layer(stacked_feats).squeeze(2).squeeze(1)
            conv_results.append(conv_out)

        return conv_results

    def _apply_linear_layers(self, conv_results, C, x_device):
        """Apply Linear layers to match the feature channels with C."""
        linear_results = []

        for conv_out in conv_results:
            linear_layer = nn.Linear(conv_out.shape[-1], C).to(x_device)
            linear_results.append(linear_layer(conv_out))

        return torch.cat(linear_results, dim=-1)  

    def forward(self, x, inter_feat, patch_resolution):
        if x.dim() == 4:
            B, _, N, C = x.shape
            x = x.view(B, N, C)
        else:
            B, N, C = x.shape        
        H_x, W_x = patch_resolution

        # Reshape and permute x to match the input format for GAP
        x_reshaped = x.view(B, H_x, W_x, C).permute(0, 3, 1, 2)

        stacked_feats = self._gap_and_pad_features(inter_feat, x_reshaped)
        stacked_feats = stacked_feats.permute(1, 0, 2).unsqueeze(1)  # [B, 1, num_feats, max_C]

        # Apply convolutions with different kernel sizes (1, 3, 5)
        conv_results = self._apply_convolutions(stacked_feats, len(inter_feat) + 1, x.device)

        # Apply linear layers to match the feature dimension C
        concatenated_features = self._apply_linear_layers(conv_results, C, x.device)

        conv_out = torch.tanh(self.mlp(concatenated_features))  # [B, C]

        # Adjust alpha using the learned conv_out features
        adjusted_alpha = self.alpha + conv_out.view(B, 1, C)

        # OPTIMIZATION: Compute normalization more efficiently
        mu = x.mean(dim=-1, keepdim=True)
        sigma = x.std(dim=-1, keepdim=True)
        x_normalized = (x - mu) / (sigma + 1e-3)

        # Apply color transform and adjustments based on channel_first flag
        if self.channel_first:
            # OPTIMIZATION: Use matmul instead of tensordot for better performance
            x_transformed = torch.matmul(x_normalized, self.color.t())
            x_out = x_transformed * adjusted_alpha + self.beta
        else:
            x_out = x_normalized * adjusted_alpha + self.beta
            x_out = torch.matmul(x_out, self.color.t())

        return x_out.view(B, N, C)
    

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    
    Args:
        in_channels (int): Number of input channels.
        input_size (int): Input image size.
        embed_dims (int): Embedding dimension.
        kernel_size (int): Kernel size of the projection layer.
        stride (int): Stride of the projection layer.
        bias (bool): Whether to use bias in the projection layer.
    """
    def __init__(self, in_channels=3, input_size=224, embed_dims=768,
                 conv_type='Conv2d', kernel_size=16, stride=16, bias=True):
        super().__init__()
        self.embed_dims = embed_dims
        
        # Calculate output size
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.input_size = input_size
        
        # Padding to maintain spatial dimensions when stride=1
        padding = (kernel_size - 1) // 2
        
        self.projection = nn.Conv2d(
            in_channels, embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        
        # Calculate initial output size
        h_out = (input_size[0] + 2 * padding - kernel_size) // stride + 1
        w_out = (input_size[1] + 2 * padding - kernel_size) // stride + 1
        self.init_out_size = (h_out, w_out)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.projection(x)  # (B, embed_dims, H', W')
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)  # (B, H'*W', embed_dims)
        return x, out_size


# ============================================================================
# Position Embedding Resize (replacing mmcls)
# ============================================================================

def resize_pos_embed(pos_embed, src_shape, dst_shape, mode='bicubic', num_extra_tokens=0):
    """
    Resize position embeddings.
    
    Args:
        pos_embed: Position embeddings tensor (1, N, C)
        src_shape: Source spatial shape (H, W)
        dst_shape: Destination spatial shape (H, W)
        mode: Interpolation mode
        num_extra_tokens: Number of extra tokens (e.g., CLS token)
    
    Returns:
        Resized position embeddings
    """
    if src_shape == dst_shape:
        return pos_embed
    
    # Separate extra tokens and position tokens
    if num_extra_tokens > 0:
        extra_tokens = pos_embed[:, :num_extra_tokens]
        pos_tokens = pos_embed[:, num_extra_tokens:]
    else:
        extra_tokens = None
        pos_tokens = pos_embed
    
    # Reshape to 2D
    B, N, C = pos_tokens.shape
    src_h, src_w = src_shape
    pos_tokens = pos_tokens.reshape(B, src_h, src_w, C).permute(0, 3, 1, 2)
    
    # Resize
    dst_h, dst_w = dst_shape
    pos_tokens = F.interpolate(pos_tokens, size=(dst_h, dst_w), mode=mode, align_corners=False)
    
    # Reshape back
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(B, -1, C)
    
    # Concatenate extra tokens
    if extra_tokens is not None:
        pos_tokens = torch.cat([extra_tokens, pos_tokens], dim=1)
    
    return pos_tokens


# ============================================================================
# Pure PyTorch WKV Implementation (OPTIMIZED)
# ============================================================================

def wkv_pytorch(B, T, C, w, u, k, v):
    """
    Pure PyTorch implementation of WKV (Weighted Key-Value) operation.
    
    This is the core RWKV attention mechanism implemented without CUDA kernels.
    
    Args:
        B: Batch size
        T: Sequence length
        C: Channel dimension
        w: Time decay weights (C,)
        u: Time first weights (C,)
        k: Key tensor (B, T, C)
        v: Value tensor (B, T, C)
    
    Returns:
        Output tensor (B, T, C)
    """
    device = k.device
    dtype = k.dtype
    
    # Ensure all tensors are float32 for numerical stability
    w = w.float()
    u = u.float()
    k = k.float()
    v = v.float()
    
    # Initialize output
    y = torch.zeros((B, T, C), device=device, dtype=torch.float32)
    
    # Initialize state
    # aa: accumulated weighted values
    # bb: accumulated weights
    # pp: previous maximum for numerical stability
    aa = torch.zeros((B, C), device=device, dtype=torch.float32)
    bb = torch.zeros((B, C), device=device, dtype=torch.float32)
    pp = torch.full((B, C), -1e38, device=device, dtype=torch.float32)
    
    # OPTIMIZATION: Pre-compute u + k for all timesteps to avoid redundant computation
    u_k = u.unsqueeze(0) + k  # (B, T, C)
    
    for t in range(T):
        kt = k[:, t, :]  # (B, C)
        vt = v[:, t, :]  # (B, C)
        
        # Compute weighted output
        ww = u_k[:, t, :]  # (B, C) - use pre-computed value
        p = torch.maximum(pp, ww)
        e1 = torch.exp(pp - p)
        e2 = torch.exp(ww - p)
        
        # Output for this timestep
        y[:, t, :] = (e1 * aa + e2 * vt) / (e1 * bb + e2)
        
        # Update state
        ww = w.unsqueeze(0) + pp  # w is negative, so this decays the previous state
        p = torch.maximum(ww, kt)
        e1 = torch.exp(ww - p)
        e2 = torch.exp(kt - p)
        
        aa = e1 * aa + e2 * vt
        bb = e1 * bb + e2
        pp = p
    
    # Convert back to original dtype
    if dtype == torch.half:
        y = y.half()
    elif dtype == torch.bfloat16:
        y = y.bfloat16()
    
    return y


def RUN_PYTORCH(B, T, C, w, u, k, v):
    """Wrapper function to run WKV in pure PyTorch."""
    return wkv_pytorch(B, T, C, w, u, k, v)


# ============================================================================
# Q-Shift Function
# ============================================================================

def q_shift(input, shift_pixel=1, gamma=1/4, patch_resolution=None):
    """
    Q-Shift operation for spatial mixing.
    Shifts different channel groups in different directions.
    """
    assert gamma <= 1/4
    B, N, C = input.shape
    input = input.transpose(1, 2).reshape(B, C, patch_resolution[0], patch_resolution[1])

    B, C, H, W = input.shape
    
    # OPTIMIZATION: Pre-allocate output tensor
    output = torch.zeros_like(input)

    # OPTIMIZATION: Use slice assignments more efficiently
    c_gamma = int(C * gamma)
    c_gamma_2 = int(C * gamma * 2)
    c_gamma_3 = int(C * gamma * 3)
    c_gamma_4 = int(C * gamma * 4)
    
    # Shift different channel groups in different directions
    output[:, 0:c_gamma, :, shift_pixel:W] = input[:, 0:c_gamma, :, 0:W-shift_pixel]
    output[:, c_gamma:c_gamma_2, :, 0:W-shift_pixel] = input[:, c_gamma:c_gamma_2, :, shift_pixel:W]
    output[:, c_gamma_2:c_gamma_3, shift_pixel:H, :] = input[:, c_gamma_2:c_gamma_3, 0:H-shift_pixel, :]
    output[:, c_gamma_3:c_gamma_4, 0:H-shift_pixel, :] = input[:, c_gamma_3:c_gamma_4, shift_pixel:H, :]
    output[:, c_gamma_4:, ...] = input[:, c_gamma_4:, ...]

    return output.flatten(2).transpose(1, 2)


# ============================================================================
# Multi-State Token Shift
# ============================================================================

class MultiStateTokenShift(nn.Module):
    def __init__(self, alpha=0.5):
        super(MultiStateTokenShift, self).__init__()
        self.alpha = alpha

    def dynamicInterpolate(self, x, spatial_shift_list):
        B, N, C = x.shape
        ema_result = x

        for i, prev_tensor in enumerate(spatial_shift_list):
            ema_result = self.alpha * ema_result + (1 - self.alpha) * prev_tensor

        return ema_result


# ============================================================================
# VRWKV Spatial Mix (Pure PyTorch)
# ============================================================================

class VRWKV_SpatialMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1/4, shift_pixel=1, init_mode='fancy',
                 key_norm=False, with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.device = None
        attn_sz = n_embd
        self._init_weights(init_mode)
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        if shift_pixel > 0:
            self.shift_func = eval(shift_mode)
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_v = None
            self.spatial_mix_r = None

        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(n_embd)
        else:
            self.key_norm = None
        self.output = nn.Linear(attn_sz, n_embd, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

        self.with_cp = with_cp

    def _init_weights(self, init_mode):
        if init_mode == 'fancy':
            with torch.no_grad():
                ratio_0_to_1 = (self.layer_id / max(self.n_layer - 1, 1))
                ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer))

                # Fancy time_decay
                decay_speed = torch.ones(self.n_embd)
                for h in range(self.n_embd):
                    decay_speed[h] = -5 + 8 * (h / max(self.n_embd - 1, 1)) ** (0.7 + 1.3 * ratio_0_to_1)
                self.spatial_decay = nn.Parameter(decay_speed)

                # Fancy time_first
                zigzag = (torch.tensor([(i + 1) % 3 - 1 for i in range(self.n_embd)]) * 0.5)
                self.spatial_first = nn.Parameter(torch.ones(self.n_embd) * math.log(0.3) + zigzag)

                # Fancy time_mix
                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
                self.spatial_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))
        elif init_mode == 'local':
            self.spatial_decay = nn.Parameter(torch.ones(self.n_embd))
            self.spatial_first = nn.Parameter(torch.ones(self.n_embd))
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]))
        elif init_mode == 'global':
            self.spatial_decay = nn.Parameter(torch.zeros(self.n_embd))
            self.spatial_first = nn.Parameter(torch.zeros(self.n_embd))
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        else:
            raise NotImplementedError

    def jit_func(self, x, spatial_shift_list, patch_resolution):
        B, T, C = x.size()

        multiStateTokenShift = MultiStateTokenShift()
        if len(spatial_shift_list) != 0:
            x = multiStateTokenShift.dynamicInterpolate(x, spatial_shift_list)
        spatial_shift_list.append(x)

        if self.shift_pixel > 0:
            xx = self.shift_func(x, self.shift_pixel, self.channel_gamma, patch_resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xv = x * self.spatial_mix_v + xx * (1 - self.spatial_mix_v)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = x
            xv = x
            xr = x

        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)

        return sr, k, v

    def forward(self, x, spatial_shift_list, patch_resolution=None):
        def _inner_forward(x):
            B, T, C = x.size()
            self.device = x.device

            sr, k, v = self.jit_func(x, spatial_shift_list, patch_resolution)
            # Use pure PyTorch WKV instead of CUDA
            x = RUN_PYTORCH(B, T, C, self.spatial_decay / T, self.spatial_first / T, k, v)
            if self.key_norm is not None:
                x = self.key_norm(x)
            x = sr * x
            x = self.output(x)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x, spatial_shift_list


# ============================================================================
# VRWKV Channel Mix
# ============================================================================

class VRWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1/4, shift_pixel=1, hidden_rate=4, init_mode='fancy',
                 key_norm=False, with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.with_cp = with_cp
        self._init_weights(init_mode)
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        if shift_pixel > 0:
            self.shift_func = eval(shift_mode)
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_r = None

        hidden_sz = hidden_rate * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

        self.value.scale_init = 0
        self.receptance.scale_init = 0

    def _init_weights(self, init_mode):
        if init_mode == 'fancy':
            with torch.no_grad():
                ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer))
                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
        elif init_mode == 'local':
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]))
        elif init_mode == 'global':
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        else:
            raise NotImplementedError

    def forward(self, x, channel_shift_list, patch_resolution=None):
        def _inner_forward(x, channel_shift_list):
            multiStateTokenShift = MultiStateTokenShift()
            if len(channel_shift_list) != 0:
                x = multiStateTokenShift.dynamicInterpolate(x, channel_shift_list)

            channel_shift_list.append(x)

            if self.shift_pixel > 0:
                xx = self.shift_func(x, self.shift_pixel, self.channel_gamma, patch_resolution)
                xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
                xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
            else:
                xk = x
                xr = x

            k = self.key(xk)
            k = torch.square(torch.relu(k))
            if self.key_norm is not None:
                k = self.key_norm(k)
            kv = self.value(k)
            x = torch.sigmoid(self.receptance(xr)) * kv
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x, channel_shift_list)
        else:
            x = _inner_forward(x, channel_shift_list)
        return x, channel_shift_list


# ============================================================================
# Layer Normalization Variants
# ============================================================================

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


# ============================================================================
# Block
# ============================================================================

class Block(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1/4, shift_pixel=1, drop_path=0., hidden_rate=4,
                 init_mode='fancy', init_values=None, post_norm=False, key_norm=False,
                 with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = LuminanceAdaptiveNorm(n_embd)
        self.ln2 = LuminanceAdaptiveNorm(n_embd)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.layer_id == 0:
            self.ln0 = LuminanceAdaptiveNorm(n_embd)

        self.att = VRWKV_SpatialMix(n_embd, n_layer, layer_id, shift_mode,
                                    channel_gamma, shift_pixel, init_mode,
                                    key_norm=key_norm)

        self.ffn = VRWKV_ChannelMix(n_embd, n_layer, layer_id, shift_mode,
                                    channel_gamma, shift_pixel, hidden_rate,
                                    init_mode, key_norm=key_norm)
        self.layer_scale = (init_values is not None)
        self.post_norm = post_norm
        if self.layer_scale:
            self.gamma1 = nn.Parameter(init_values * torch.ones((n_embd)), requires_grad=True)
            self.gamma2 = nn.Parameter(init_values * torch.ones((n_embd)), requires_grad=True)
        self.with_cp = with_cp

    def forward(self, x, spatial_shift_list, channel_shift_list, inter_feat, patch_resolution=None):
        def _inner_forward(x, spatial_shift_list, channel_shift_list, inter_feat):
            if self.layer_id == 0:
                x = self.ln0(x, inter_feat, patch_resolution)
            if self.post_norm:
                if self.layer_scale:
                    att, spatial_shift_list = self.att(x, spatial_shift_list, patch_resolution)
                    x = x + self.drop_path(self.gamma1 * self.ln1(att, inter_feat, patch_resolution))
                    ffn, channel_shift_list = self.ffn(x, channel_shift_list, patch_resolution)
                    x = x + self.drop_path(self.gamma2 * self.ln2(ffn, inter_feat, patch_resolution))
                else:
                    att, spatial_shift_list = self.att(x, spatial_shift_list, patch_resolution)
                    x = x + self.drop_path(self.ln1(att, inter_feat, patch_resolution))
                    ffn, channel_shift_list = self.ffn(x, channel_shift_list, patch_resolution)
                    x = x + self.drop_path(self.ln2(ffn, inter_feat, patch_resolution))
            else:
                if self.layer_scale:
                    att, spatial_shift_list = self.att(self.ln1(x, inter_feat, patch_resolution), spatial_shift_list, patch_resolution)
                    x = x + self.drop_path(self.gamma1 * att)
                    ffn, channel_shift_list = self.ffn(self.ln2(x, inter_feat, patch_resolution), channel_shift_list, patch_resolution)
                    x = x + self.drop_path(self.gamma2 * ffn)
                else:
                    att, spatial_shift_list = self.att(self.ln1(x, inter_feat, patch_resolution), spatial_shift_list, patch_resolution)
                    x = x + self.drop_path(att)
                    ffn, channel_shift_list = self.ffn(self.ln2(x, inter_feat, patch_resolution), channel_shift_list, patch_resolution)
                    x = x + self.drop_path(ffn)
            return x, spatial_shift_list, channel_shift_list, inter_feat

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x, spatial_shift_list, channel_shift_list, inter_feat)
        else:
            x, spatial_shift_list, channel_shift_list, inter_feat = _inner_forward(x, spatial_shift_list, channel_shift_list, inter_feat)
        return x, spatial_shift_list, channel_shift_list, inter_feat


# ============================================================================
# URWKV Backbone (Pure PyTorch)
# ============================================================================

class URWKV(nn.Module):
    def __init__(self,
                 img_size=32,
                 patch_size=16,
                 in_channels=3,
                 out_indices=-1,
                 drop_rate=0.,
                 embed_dims=256,
                 depth=12,
                 drop_path_rate=0.,
                 channel_gamma=1/4,
                 shift_pixel=1,
                 shift_mode='q_shift',
                 init_mode='fancy',
                 post_norm=False,
                 key_norm=False,
                 init_values=None,
                 hidden_rate=4,
                 final_norm=True,
                 interpolate_mode='bicubic',
                 with_cp=False,
                 init_cfg=None):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_extra_tokens = 0
        self.num_layers = depth
        self.drop_path_rate = drop_path_rate

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=1,
            bias=True)

        self.patch_resolution = self.patch_embed.init_out_size
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        # Set position embedding
        self.interpolate_mode = interpolate_mode
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, self.embed_dims))

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
            assert 0 <= out_indices[i] <= self.num_layers, \
                f'Invalid out_indices {index}'
        self.out_indices = out_indices
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(Block(
                n_embd=embed_dims,
                n_layer=depth,
                layer_id=i,
                channel_gamma=channel_gamma,
                shift_pixel=shift_pixel,
                shift_mode=shift_mode,
                hidden_rate=hidden_rate,
                drop_path=dpr[i],
                init_mode=init_mode,
                post_norm=post_norm,
                key_norm=key_norm,
                init_values=init_values,
                with_cp=with_cp
            ))

        self.final_norm = final_norm
        if final_norm:
            self.ln1 = nn.LayerNorm(self.embed_dims)

    def forward(self, x, inter_feat):
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)

        x = x + resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens)

        x = self.drop_after_pos(x)

        outs = []
        spatial_shift_list = []
        channel_shift_list = []
        for i, layer in enumerate(self.layers):
            x, spatial_shift_list, channel_shift_list, inter_feat = layer(x, spatial_shift_list, channel_shift_list, inter_feat, patch_resolution)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.ln1(x)

            if i in self.out_indices:
                B, _, C = x.shape

                patch_token = x.reshape(B, *patch_resolution, C)
                patch_token = patch_token.permute(0, 3, 1, 2)

                out = patch_token
                outs.append(out)
        return outs[0], inter_feat


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionModule, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.branch3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.branch5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.branch_pool = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        
        self.conv = nn.Conv2d(in_channels * 4, 1, kernel_size=1)

        self._initialize_weights()

    def _initialize_weights(self):
        # Custom initialization for each convolution layer
        nn.init.kaiming_normal_(self.branch1x1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.branch3x3.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.branch5x5.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.branch_pool.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(self.pool(x))

        outputs = torch.cat([branch1x1, branch3x3, branch5x5, branch_pool], dim=1)
        outputs = self.conv(outputs)  
        return outputs

# State-aware Selective Fusion (SSF) module
class SSF(nn.Module):
    def __init__(self, num_feats, encode_channels, target_channels):
        super(SSF, self).__init__()
        self.num_feats = num_feats
        self.target_channels = target_channels

        # Alignment convolution layers for each encoder feature map
        self.align_convs = nn.ModuleList([
            nn.Conv2d(in_channels, target_channels, kernel_size=1) 
            for in_channels in encode_channels
        ])

        self.conv_fusion = nn.Conv2d(num_feats, num_feats, kernel_size=3, padding=1)
        self.inception = InceptionModule(num_feats, num_feats)
        self.final_conv = nn.Conv2d(target_channels * 2, target_channels, kernel_size=3, padding=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for conv in self.align_convs:
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv_fusion.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.final_conv.weight, mode='fan_out', nonlinearity='relu')


    def forward(self, x, feat_list):
        target_height, target_width = x.size(2), x.size(3)
        feat_list.reverse()
        # Process each feature map in feat_list
        aligned_feats = []
        encoder_feat = torch.zeros_like(x)
        
        for i, feat in enumerate(feat_list):
            if feat.size(2) == target_height and feat.size(3) == target_width:
                encoder_feat = feat

            feat = torch.mean(feat, dim=1, keepdim=True)
            feat = F.interpolate(feat, size=(target_height, target_width), mode='bilinear', align_corners=False)
            aligned_feats.append(feat)

        # Stack the aligned feature maps along the channel dimension
        stacked_feats = torch.cat(aligned_feats, dim=1)

        # Fuse features along the N dimension
        fused_feat = self.conv_fusion(stacked_feats)

        # Apply the Inception module for multi-scale feature extraction
        inception_feat = self.inception(fused_feat)
        inception_feat = torch.sigmoid(inception_feat)

        guided_feat = inception_feat * encoder_feat

        output = self.final_conv(torch.cat([guided_feat, x], dim=1))

        return output

class NormLayer(nn.Module):
    def __init__(self, num_channels):
        super(NormLayer, self).__init__()
        
        # Learnable scaling and bias parameters
        self.scale = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        
    def forward(self, x):
        # Apply normalization with learned scaling and bias
        return x * self.scale + self.bias
    

class Encoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.patch_size = 4
        self.head = nn.Conv2d(3, self.dim, kernel_size=3, stride=1, padding=1, bias=False)

        # stage1 
        self.enhanceBlock1 = URWKV(patch_size=3, in_channels=self.dim, embed_dims=self.dim, depth=3)
        self.proj1 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        
        # stage2
        self.enhanceBlock2 = URWKV(patch_size=3, in_channels=self.dim, embed_dims=self.dim*2, depth=3)
        self.proj2 = nn.Sequential(
            nn.Conv2d(self.dim*2, self.dim*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        
        # stage3
        self.enhanceBlock3 = URWKV(patch_size=3, in_channels=self.dim*2, embed_dims=self.dim*4, depth=3)
        self.proj3 = nn.Sequential(
            nn.Conv2d(self.dim*4, self.dim*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
    def forward(self, x, inter_feat):
        B, C, H, W = x.shape
        x = self.head(x)
        inter_feat.append(x)  # [1, 32, 256, 256]
        # stage1
        x1, inter_feat = self.enhanceBlock1(x, inter_feat)
        inter_feat.append(x1)  # [1, 32, 256, 256], [1, 32, 256, 256]
        x1 = self.proj1(x1)  # C, H, W  
        x1_out = F.interpolate(x1, scale_factor=0.5, mode='bilinear')  # down 1/2
        
        # stage2
        x1_out, inter_feat = self.enhanceBlock2(x1_out, inter_feat)
        inter_feat.append(x1_out) # [1, 32, 256, 256], [1, 32, 256, 256], [1, 32, 128, 128]
        x2 = self.proj2(x1_out)  # 2C, H/2, W/2
        x2_out = F.interpolate(x2, scale_factor=0.5, mode='bilinear')  # down 1/4
     
        # stage3
        x2_out, inter_feat = self.enhanceBlock3(x2_out, inter_feat)
        inter_feat.append(x2_out) # [1, 32, 256, 256], [1, 32, 256, 256], [1, 32, 128, 128], [1, 64, 64, 64]
        x3 = self.proj3(x2_out)  # 4C, H/4, W/4
        x3_out = F.interpolate(x3, scale_factor=0.5, mode='bilinear')  # down 1/8, 4C, H/8, W/8
        
        feat_list = [x1, x2, x3, x3_out]
        return feat_list, inter_feat
    
class Decoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.patch_size = 4
        self.residual_depth = [1, 1, 1]
        self.recursive_depth = [1, 1, 1]
        self.enhanceBlock1 = URWKV(patch_size=3, in_channels=self.dim*4, embed_dims=self.dim*4, depth=2)
        self.proj1 = nn.Sequential(
            nn.Conv2d(self.dim*4, self.dim*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        
        self.enhanceBlock2 = URWKV(patch_size=3, in_channels=self.dim*2, embed_dims=self.dim*2, depth=2)
        self.proj2 = nn.Sequential(
            nn.Conv2d(self.dim*2, self.dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        
        self.enhanceBlock3 = URWKV(patch_size=3, in_channels=self.dim, embed_dims=self.dim, depth=2)
        self.proj3 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        
        self.tail = nn.Conv2d(self.dim, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.multiscale_fuse1 = SSF(num_feats=3, encode_channels=[self.dim*4, self.dim*2, self.dim], target_channels=self.dim)
        self.multiscale_fuse2 = SSF(num_feats=3, encode_channels=[self.dim*4, self.dim*2, self.dim], target_channels=self.dim*2)
        self.multiscale_fuse3 = SSF(num_feats=3, encode_channels=[self.dim*4, self.dim*2, self.dim], target_channels=self.dim*4)
        self.upSample = nn.Upsample(scale_factor=2, mode="bilinear")
    def forward(self, x, encode_list, inter_feat):
        feat_1s, feat_2s, feat_4s = encode_list[0], encode_list[1], encode_list[2]
        
        
        x1 = self.multiscale_fuse3(self.upSample(x),encode_list[:3])
        x1, inter_feat = self.enhanceBlock1(x1, inter_feat)
        inter_feat.append(x1)
        x1 = self.proj1(x1)
        x2 = self.multiscale_fuse2(self.upSample(x1),encode_list[:3])
        x2, inter_feat = self.enhanceBlock2(x2, inter_feat)
        inter_feat.append(x2)
        x2 = self.proj2(x2)
        x3 = self.multiscale_fuse1(self.upSample(x2),encode_list[:3])
        x3, inter_feat = self.enhanceBlock3(x3, inter_feat)
        inter_feat.append(x3)
        x3 = self.proj3(x3)
        out = self.tail(x3)
        return out
    
# recursive network based on residual units
class LLENet(nn.Module):
    def __init__(self, dim=32): 
        super().__init__()
        self.dim = dim
        self.encoder = Encoder(dim=self.dim) # 3 -> 32 -> 64 -> 128
        self.decoder = Decoder(dim=self.dim)  # 128 -> 64 -> 32 -> 3
        self.apply(self._init_weights)  # Correctly apply init_weights to all submodules

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        outer_shortcut = x
        inter_feat = []
        encode_list, inter_feat = self.encoder(x, inter_feat)
        x = encode_list[-1]
        x = self.decoder(x, encode_list, inter_feat)
        x = torch.add(x, outer_shortcut)
        return x
    
if __name__ == '__main__':
    model = LLENet().cuda()
    x = torch.rand(1, 3, 128, 128).cuda()
    model.eval()
    with torch.no_grad():
        res = model(x)
        print(res.shape)