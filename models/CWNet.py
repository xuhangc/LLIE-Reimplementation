"""
[ICCV 25] CWNet: Causal Wavelet Network for Low-Light Image Enhancement 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numbers
from functools import partial
import pywt


# ============================================================================
# Wavelet Transform Utilities
# ============================================================================

def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


# ============================================================================
# Scale Module for WTConv2d
# ============================================================================

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None
    
    def forward(self, x):
        return torch.mul(self.weight, x)


# ============================================================================
# Wavelet Transform Convolution
# ============================================================================

class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels*4, in_channels*4, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels*4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels*4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride, groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]
            
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for i in range(self.wt_levels-1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0
        
        x = self.base_scale(self.base_conv(x))
        x = x + x_tag
        
        if self.do_stride is not None:
            x = self.do_stride(x)

        return x


# ============================================================================
# Fourier Transform Units
# ============================================================================

class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho'):
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.fft_norm = fft_norm
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
                                          out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.to(torch.float32)
        batch = x.shape[0]
        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.conv_layer(ffted)
        ffted = self.relu(self.bn(ffted.to(torch.float32)))
        ffted = ffted.to(torch.float32)

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)
        return output


class SeparableFourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, kernel_size=3):
        super(SeparableFourierUnit, self).__init__()
        self.groups = groups
        row_out_channels = out_channels // 2
        col_out_channels = out_channels - row_out_channels
        self.row_conv = torch.nn.Conv2d(in_channels=in_channels * 2,
                                        out_channels=row_out_channels * 2,
                                        kernel_size=(kernel_size, 1),
                                        stride=1, padding=(kernel_size // 2, 0),
                                        padding_mode='reflect',
                                        groups=self.groups, bias=False)
        self.col_conv = torch.nn.Conv2d(in_channels=in_channels * 2,
                                        out_channels=col_out_channels * 2,
                                        kernel_size=(kernel_size, 1),
                                        stride=1, padding=(kernel_size // 2, 0),
                                        padding_mode='reflect',
                                        groups=self.groups, bias=False)
        self.row_bn = torch.nn.BatchNorm2d(row_out_channels * 2)
        self.col_bn = torch.nn.BatchNorm2d(col_out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def process_branch(self, x, conv, bn):
        batch = x.shape[0]
        ffted = torch.fft.rfft(x, norm="ortho")
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        ffted = self.relu(bn(conv(ffted)))
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])
        output = torch.fft.irfft(ffted, s=x.shape[-1:], norm="ortho")
        return output

    def forward(self, x):
        rowwise = self.process_branch(x, self.row_conv, self.row_bn)
        colwise = self.process_branch(x.permute(0, 1, 3, 2), self.col_conv, self.col_bn).permute(0, 1, 3, 2)
        out = torch.cat((rowwise, colwise), dim=1)
        return out


class SpectralTransform(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True, separable_fu=False, **fu_kwargs):
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(out_channels // 2, out_channels // 2, groups, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = FourierUnit(out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1), dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)
        return output


# ============================================================================
# FFC (Fast Fourier Convolution) modules
# ============================================================================

class FFC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False, **spectral_kwargs):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) + self.convg2g(x_g)

        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect',
                 enable_lfu=True, **kwargs):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, padding_type=padding_type, **kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l.to(torch.float32)))
        x_g = self.act_g(self.bn_g(x_g.to(torch.float32)))
        return x_l, x_g


class FFCResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, activation_layer=nn.ReLU):
        super(FFCResnetBlock, self).__init__()
        self.ffc1 = FFC_BN_ACT(dim, dim, 3, 0.75, 0.75, stride=1, padding=1, dilation=dilation, groups=1, bias=False,
                               norm_layer=nn.BatchNorm2d, activation_layer=activation_layer, enable_lfu=False)
        self.ffc2 = FFC_BN_ACT(dim, dim, 3, 0.75, 0.75, stride=1, padding=1, dilation=1, groups=1, bias=False,
                               norm_layer=nn.BatchNorm2d, activation_layer=activation_layer, enable_lfu=False)

    def forward(self, x):
        output = x
        _, c, _, _ = output.shape
        output = torch.split(output, [c - int(c * 0.75), int(c * 0.75)], dim=1)
        x_l, x_g = self.ffc1(output)
        output = self.ffc2((x_l, x_g))
        output = torch.cat(output, dim=1)
        output = x + output
        return output


# ============================================================================
# Layer Normalization modules
# ============================================================================

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class LightBlock(nn.Module):
    def __init__(self, dim):
        super(LightBlock, self).__init__()
        self.channel = dim
        self.SIM = nn.Sequential(
            LayerNorm2d(dim),
            FFCResnetBlock(dim),
            nn.Conv2d(dim, dim, kernel_size=5, padding=2, stride=1, bias=True),
            SimpleGate(),
            nn.Conv2d(dim // 2, dim, kernel_size=1, stride=1, bias=True),
        )
        self.CIM = nn.Sequential(
            LayerNorm2d(dim),
            FFCResnetBlock(dim),
            nn.Conv2d(dim, dim * 4, kernel_size=1, stride=1, bias=True),
            SimpleGate(),
            nn.Conv2d(dim * 2, dim, kernel_size=1, stride=1, bias=True),
        )

    def forward(self, x):
        y = self.SIM(x) + x
        y = self.CIM(y) + y
        return y


class Depth_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Depth_conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


# ============================================================================
# DWT/IDWT modules
# ============================================================================

def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]

    min_height = min(x1.size(2), x2.size(2), x3.size(2), x4.size(2))
    min_width = min(x1.size(3), x2.size(3), x3.size(3), x4.size(3))

    x1 = x1[:, :, :min_height, :min_width]
    x2 = x2[:, :, :min_height, :min_width]
    x3 = x3[:, :, :min_height, :min_width]
    x4 = x4[:, :, :min_height, :min_width]

    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return x_LL, x_HL, x_LH, x_HH


class DWT(nn.Module):
    def __init__(self, fuseh=False):
        super(DWT, self).__init__()
        self.requires_grad = False
        self.fuseh = fuseh

    def forward(self, x):
        if self.fuseh:
            x_LL, x_HL, x_LH, x_HH = dwt_init(x)
            return x_LL, torch.cat((x_HL, x_LH, x_HH), dim=0)
        else:
            return dwt_init(x)


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = int(in_batch / (r ** 2)), in_channel, r * in_height, r * in_width
    x1 = x[0:out_batch, :, :] / 2
    x2 = x[out_batch:out_batch * 2, :, :, :] / 2
    x3 = x[out_batch * 2:out_batch * 3, :, :, :] / 2
    x4 = x[out_batch * 3:out_batch * 4, :, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().to(x.device)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
    return h


class IDWT(nn.Module):
    def __init__(self):
        super(IDWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


# ============================================================================
# Utility modules
# ============================================================================

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class FeedForward(nn.Module):
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


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


def to_4d(x, h, w):
    b, hw, c = x.shape
    return x.reshape(b, h, w, c).permute(0, 3, 1, 2)


def to_3d(x):
    b, c, h, w = x.shape
    return x.permute(0, 2, 3, 1).reshape(b, h * w, c)


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
# SS2D6 - Pure PyTorch Implementation (6-direction selective scan)
# ============================================================================

def selective_scan_pytorch(u, delta, A, B, C, D_param, z=None, delta_bias=None, delta_softplus=False, return_last_state=False):
    """
    Pure PyTorch implementation of selective scan.
    
    Arguments:
        u: (B, D, L) - input where D = K * d_inner
        delta: (B, D, L) - time step
        A: (D, N) - state matrix
        B: (B, K*N, L) - input matrix (flattened)
        C: (B, K*N, L) - output matrix (flattened)
        D_param: (D,) - skip connection
        z: (B, D, L) - optional gate
        delta_bias: (D,) - optional bias for delta
        delta_softplus: bool - apply softplus to delta
        return_last_state: bool - return final state
    
    Returns:
        y: (B, D, L) - output
        last_state: (B, D, N) - optional final state
    """
    batch, dim, seqlen = u.shape
    dstate = A.shape[1]
    
    # Apply delta bias and softplus
    if delta_bias is not None:
        delta = delta + delta_bias.unsqueeze(0).unsqueeze(-1)
    if delta_softplus:
        delta = F.softplus(delta)
    
    # Compute deltaA = exp(delta * A)
    # A: (D, N), delta: (B, D, L) -> deltaA: (B, D, L, N)
    deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(-2))  # (B, D, L, N)
    
    # Reshape B and C to match the dimension structure
    # B and C come in as (B, K*N, L), we need to handle them properly
    # Since A is (D, N) where D = K * d_inner, we need to expand B and C
    # B: (B, K*N, L) -> need to broadcast to (B, D, N, L)
    
    # For the SS2D6 case, B and C are (B, K*d_state, L)
    # We need to repeat them to match the D dimension
    K = dim // (dim // 6) if dim >= 6 else 1  # Estimate K from dimensions
    if B.shape[1] != dstate:
        # B is (B, K*N, L), reshape to (B, K, N, L)
        K_actual = B.shape[1] // dstate
        B = B.view(batch, K_actual, dstate, seqlen)
        C = C.view(batch, K_actual, dstate, seqlen)
        
        # Repeat B and C to match each direction's d_inner
        d_inner_per_k = dim // K_actual
        B_expanded = B.unsqueeze(2).expand(-1, -1, d_inner_per_k, -1, -1)  # (B, K, d_inner, N, L)
        B_expanded = B_expanded.reshape(batch, dim, dstate, seqlen)  # (B, D, N, L)
        C_expanded = C.unsqueeze(2).expand(-1, -1, d_inner_per_k, -1, -1)
        C_expanded = C_expanded.reshape(batch, dim, dstate, seqlen)
    else:
        B_expanded = B.unsqueeze(1).expand(-1, dim, -1, -1)  # (B, D, N, L)
        C_expanded = C.unsqueeze(1).expand(-1, dim, -1, -1)
    
    # Compute deltaB_u = delta * B * u
    # delta: (B, D, L), B_expanded: (B, D, N, L), u: (B, D, L)
    deltaB_u = delta.unsqueeze(2) * B_expanded * u.unsqueeze(2)  # (B, D, N, L)
    deltaB_u = deltaB_u.permute(0, 1, 3, 2)  # (B, D, L, N)
    
    C_expanded = C_expanded.permute(0, 1, 3, 2)  # (B, D, L, N)
    
    # Sequential scan
    x = torch.zeros(batch, dim, dstate, device=u.device, dtype=u.dtype)
    ys = []
    
    for i in range(seqlen):
        x = deltaA[:, :, i, :] * x + deltaB_u[:, :, i, :]  # (B, D, N)
        y = (x * C_expanded[:, :, i, :]).sum(dim=-1)  # (B, D)
        ys.append(y)
    
    y = torch.stack(ys, dim=2)  # (B, D, L)
    
    # Add skip connection
    if D_param is not None:
        y = y + u * D_param.unsqueeze(0).unsqueeze(-1)
    
    # Apply gate
    if z is not None:
        y = y * F.silu(z)
    
    if return_last_state:
        return y, x
    return y


class SS2D6(nn.Module):
    """
    SS2D6 - Selective State Space 2D with 6 scanning directions.
    Pure PyTorch implementation without CUDA dependencies.
    
    The 6 directions are:
    1. Horizontal left-to-right
    2. Horizontal right-to-left (inverse)
    3. Vertical top-to-bottom (transposed)
    4. Vertical bottom-to-top (transposed inverse)
    5. Diagonal scan
    6. Diagonal scan inverse
    """
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            scan_type='hl',
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.scantype = scan_type
        
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        # Initialize 6 sets of projections for 6 scanning directions
        K = 6
        self.K = K
        
        # x_proj for each direction
        x_proj_list = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in x_proj_list], dim=0))
        
        # dt_projs for each direction
        dt_projs_list = [
            self._dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs_list], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs_list], dim=0))
        
        # A_logs and Ds
        self.A_logs = self._A_log_init(self.d_state, self.d_inner, copies=K, merge=True)
        self.Ds = self._D_init(self.d_inner, copies=K, merge=True)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def _dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def _A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).unsqueeze(0).expand(d_inner, -1).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = A_log.unsqueeze(0).expand(copies, -1, -1)
            if merge:
                A_log = A_log.reshape(-1, d_state)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def _D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = D.unsqueeze(0).expand(copies, -1)
            if merge:
                D = D.reshape(-1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def diagonal_trans(self, x, H, W):
        """Transform tensor for diagonal scanning."""
        B, K, _, L = x.shape
        assert L == H * W, "L must equal H*W"
        idx = torch.arange(H * W, device=x.device).reshape(H, W)
        i_idx = torch.arange(H, device=x.device).reshape(-1, 1).expand(H, W)
        j_idx = torch.arange(W, device=x.device).reshape(1, -1).expand(H, W)
        diag_mask = (i_idx + j_idx)
        sorted_idx = torch.argsort(diag_mask.view(-1))
        diag_indices = torch.index_select(idx.view(-1), 0, sorted_idx)
        x_flat = x.view(B, K, -1, H * W)
        diag_flat = torch.index_select(x_flat, dim=3, index=diag_indices)
        return diag_flat, diag_indices

    def reverse_diagonal_trans(self, x, diag_indices):
        """Reverse the diagonal transformation."""
        B, K, _, L = x.shape
        x_flat = x.view(B, K, -1, L)
        reverse_indices = torch.argsort(diag_indices)
        original_flat = torch.index_select(x_flat, dim=3, index=reverse_indices)
        return original_flat

    def forward_core(self, x):
        """Core forward with 6 scanning directions using pure PyTorch."""
        B, C, H, W = x.shape
        L = H * W
        K = 6

        # Create 6 scanning patterns
        # 1-2: horizontal and inverse
        # 3-4: vertical (transposed) and inverse
        # 5-6: diagonal and inverse
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (B, 4, D, L)

        # Diagonal scanning
        x_invhh = torch.flip(x, dims=[-1])
        x_hh = torch.stack([x.view(B, -1, L), x_invhh.view(B, -1, L)], dim=1).view(B, 2, -1, L)
        x_hh, diag_indices = self.diagonal_trans(x_hh, H, W)

        xs = torch.cat([xs, x_hh], dim=1)  # (B, 6, D, L)

        # Project x for each direction
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # (B, K*D, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (B, K*D, L)
        Bs = Bs.float().view(B, K, -1, L)  # (B, K, d_state, L)
        Cs = Cs.float().view(B, K, -1, L)  # (B, K, d_state, L)
        Ds = self.Ds.float().view(-1)  # (K*D)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (K*D, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (K*D)

        # Use pure PyTorch selective scan
        out_y = selective_scan_pytorch(
            xs, dts,
            As, Bs.reshape(B, -1, L), Cs.reshape(B, -1, L), Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        # Process outputs from different directions
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        transhh_y = self.reverse_diagonal_trans(out_y[:, 4:6], diag_indices)
        invhh_y = torch.flip(transhh_y[:, 1], dims=[-1])

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y, transhh_y[:, 0], invhh_y

    def forward(self, x, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (B, H, W, D)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (B, D, H, W)
        y1, y2, y3, y4, y5, y6 = self.forward_core(x)
        y = y1 + y2 + y3 + y4 + y5 + y6
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)

        return out


# ============================================================================
# ProcessBlock
# ============================================================================

class ProcessBlock(nn.Module):
    def __init__(self, dims, d_state=16, n_l_block=1, n_h_block=1, LayerNorm_type='WithBias'):
        super(ProcessBlock, self).__init__()
        self.dim = dims
        self.dwt = DWT(fuseh=False)
        self.idwt = IDWT()
        self.lnum = n_l_block
        self.hnum = n_h_block
        self.hhenhance = Depth_conv(self.dim, self.dim)
        self.llenhance = nn.ModuleList()
        for layer in range(2):
            self.llenhance.append(
                WTConv2d(dims, dims, kernel_size=5, wt_levels=3))

        self.hhmamba = nn.ModuleList()
        self.norm2 = LayerNorm(self.dim, LayerNorm_type)

        for layer in range(self.hnum):
            self.hhmamba.append(nn.ModuleList([
                SS2D6(d_model=dims, dropout=0, d_state=d_state, scan_type='lh'),
                PreNorm(dims, FeedForward(dim=dims))
            ]))

        self.horizontal_conv, self.vertical_conv, self.diagonal_conv = self.create_wave_conv()

        self.posenhance = nn.ModuleList()
        for layer in range(self.lnum):
            self.posenhance.append(
                LightBlock(self.dim))

        self.conv_fusechannel = nn.Conv2d(self.dim * 2, self.dim, 1, stride=1, bias=False)

    def create_conv_layer(self, kernel):
        conv = nn.Conv2d(in_channels=self.dim, out_channels=self.dim, kernel_size=3, padding=1, bias=False)
        conv.weight.data = kernel.repeat(self.dim, self.dim, 1, 1)
        return conv

    def create_wave_conv(self):
        horizontal_kernel = torch.tensor([[1, 0, -1],
                                          [1, 0, -1],
                                          [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        vertical_kernel = torch.tensor([[1, 1, 1],
                                        [0, 0, 0],
                                        [-1, -1, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        diagonal_kernel = torch.tensor([[0, 1, 0],
                                        [1, -4, 1],
                                        [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        horizontal_conv = self.create_conv_layer(horizontal_kernel)
        vertical_conv = self.create_conv_layer(vertical_kernel)
        diagonal_conv = self.create_conv_layer(diagonal_kernel)
        return horizontal_conv, vertical_conv, diagonal_conv

    def forward(self, x):
        b, c, h, w = x.shape
        xori = x
        ll, hl, lh, hh = self.dwt(x)
        for layer in self.llenhance:
            ll = layer(ll)
        hh = torch.cat((hl, lh, hh), dim=0)
        hh = self.hhenhance(hh)
        e_hl, e_lh, e_hh = hh[:b, ...], hh[b:2 * b, ...], hh[2 * b:, ...]

        ll_hl = self.horizontal_conv(ll)
        ll_lh = self.vertical_conv(ll)
        ll_hh = self.diagonal_conv(ll)

        e_hl = torch.cat((e_hl, ll_hl), dim=1)
        e_lh = torch.cat((e_lh, ll_lh), dim=1)
        e_hh = torch.cat((e_hh, ll_hh), dim=1)

        e_high = torch.cat((e_hl, e_lh, e_hh), dim=0)
        e_high = self.conv_fusechannel(e_high)
        e_high = self.norm2(e_high)

        for (ss2d, ff) in self.hhmamba:
            y = e_high.permute(0, 2, 3, 1)
            e_high = ss2d(y) + e_high.permute(0, 2, 3, 1)
            e_high = ff(e_high) + e_high
            e_high = e_high.permute(0, 3, 1, 2)
        x_out = self.idwt(torch.cat((ll, e_high), dim=0)) + xori

        for layer in self.posenhance:
            x_out = layer(x_out)

        return x_out


# ============================================================================
# CWNet Main Model
# ============================================================================

class CWNet(nn.Module):
    def __init__(self, nc=16, n_l_blocks=[1,3,3,3,1], n_h_blocks=[1,2,2,2,1]):
        super(CWNet, self).__init__()
        self.conv0 = nn.Conv2d(3, nc, 1, 1, 0)
        self.conv1 = ProcessBlock(nc, d_state=16, n_l_block=n_l_blocks[0], n_h_block=n_h_blocks[0])
        self.downsample1 = nn.Conv2d(nc, nc * 2, stride=2, kernel_size=2, padding=0)
        self.conv2 = ProcessBlock(nc * 2, d_state=16, n_l_block=n_l_blocks[1], n_h_block=n_h_blocks[1])
        self.downsample2 = nn.Conv2d(nc * 2, nc * 3, stride=2, kernel_size=2, padding=0)
        self.conv3 = ProcessBlock(nc * 3, d_state=16, n_l_block=n_l_blocks[2], n_h_block=n_h_blocks[2])
        self.up1 = nn.ConvTranspose2d(nc * 5, nc * 2, 1, 1)
        self.conv4 = ProcessBlock(nc * 2, d_state=16, n_l_block=n_l_blocks[3], n_h_block=n_h_blocks[3])
        self.up2 = nn.ConvTranspose2d(nc * 3, nc * 1, 1, 1)
        self.conv5 = ProcessBlock(nc, d_state=16, n_l_block=n_l_blocks[4], n_h_block=n_h_blocks[4])
        self.convout = nn.Conv2d(nc, 3, 1, 1, 0)

    def forward(self, x):
        x_ori = x
        x = self.conv0(x)
        x01 = self.conv1(x)
        x1 = self.downsample1(x01)
        x12 = self.conv2(x1)
        x2 = self.downsample2(x12)
        x3 = self.conv3(x2)
        x34 = self.up1(torch.cat([F.interpolate(x3, size=(x12.size()[2], x12.size()[3]), mode='bilinear'), x12], 1))
        x4 = self.conv4(x34)
        x4 = self.up2(torch.cat([F.interpolate(x4, size=(x01.size()[2], x01.size()[3]), mode='bilinear'), x01], 1))
        x5 = self.conv5(x4)
        xout = self.convout(x5)
        xout = x_ori + xout

        return xout


# ============================================================================
# Main function for testing
# ============================================================================

if __name__ == '__main__':
    print("Testing CWNet Pure PyTorch Implementation...")
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model with typical configuration
    nc = 32  # number of channels
    n_l_blocks = [1, 1, 1, 1, 1]  # number of light blocks per stage
    n_h_blocks = [1, 1, 1, 1, 1]  # number of high-frequency blocks per stage
    
    model = CWNet().to(device)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    with torch.no_grad():
        # Use 128x128 input for testing
        inputs = torch.randn((1, 3, 128, 128)).to(device)
        print(f"Input shape: {inputs.shape}")
        
        res = model(inputs)
        print(f"Output shape: {res.shape}")
        print("Forward pass successful!")
    
    print("\nCWNet Pure PyTorch implementation is working correctly!")