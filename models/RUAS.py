"""
Retinex-inspired unrolling with cooperative prior architecture search for low-light image enhancement
Liu, Risheng and Ma, Long and Zhang, Jiaao and Fan, Xin and Luo, Zhongxuan
CVPR, 2021
"""
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

IEM_Geno = Genotype(normal=[
    ('skip_connect', 0),
    ('resconv_1x1', 1),
    ('resdilconv_3x3', 2),
    ('conv_3x3', 3), ('conv_3x3', 4),
    ('skip_connect', 5),
    ('conv_3x3', 6)
], normal_concat=None, reduce=None, reduce_concat=None)

NRM_Geno = Genotype(normal=[
    ('resconv_1x1', 0),
    ('resconv_1x1', 1),
    ('resdilconv_3x3', 2),
    ('skip_connect', 3),
    ('resconv_1x1', 4),
    ('resconv_1x1', 5),
    ('skip_connect', 6)
], normal_concat=None, reduce=None, reduce_concat=None)

OPS = {
    'avg_pool_3x3': lambda C_in, C_out: nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C_in, C_out: nn.MaxPool2d(3, stride=1, padding=1),
    'skip_connect': lambda C_in, C_out: Identity(),
    'conv_1x1': lambda C_in, C_out: ConvBlock(C_in, C_out, 1),
    'conv_3x3': lambda C_in, C_out: ConvBlock(C_in, C_out, 3),
    'conv_5x5': lambda C_in, C_out: ConvBlock(C_in, C_out, 5),
    'conv_7x7': lambda C_in, C_out: ConvBlock(C_in, C_out, 7),
    'dilconv_3x3': lambda C_in, C_out: ConvBlock(C_in, C_out, 3, dilation=2),
    'dilconv_5x5': lambda C_in, C_out: ConvBlock(C_in, C_out, 5, dilation=2),
    'dilconv_7x7': lambda C_in, C_out: ConvBlock(C_in, C_out, 7, dilation=2),
    'resconv_1x1': lambda C_in, C_out: ResBlock(C_in, C_out, 1),
    'resconv_3x3': lambda C_in, C_out: ResBlock(C_in, C_out, 3),
    'resconv_5x5': lambda C_in, C_out: ResBlock(C_in, C_out, 5),
    'resconv_7x7': lambda C_in, C_out: ResBlock(C_in, C_out, 7),
    'resdilconv_3x3': lambda C_in, C_out: ResBlock(C_in, C_out, 3, dilation=2),
    'resdilconv_5x5': lambda C_in, C_out: ResBlock(C_in, C_out, 5, dilation=2),
    'resdilconv_7x7': lambda C_in, C_out: ResBlock(C_in, C_out, 7, dilation=2),
}


class ConvBlock(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride=1, dilation=1, groups=1):
        super(ConvBlock, self).__init__()
        padding = int((kernel_size - 1) / 2) * dilation
        self.op = nn.Conv2d(C_in, C_out, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                            groups=groups)

    def forward(self, x):
        return self.op(x)


class ResBlock(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride=1, dilation=1, groups=1):
        super(ResBlock, self).__init__()
        padding = int((kernel_size - 1) / 2) * dilation
        self.op = nn.Conv2d(C_in, C_out, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                            groups=groups)

    def forward(self, x):
        return self.op(x) + x


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SearchBlock(nn.Module):
    def __init__(self, channel, genotype):
        super(SearchBlock, self).__init__()
        self.channel = channel
        op_names, _ = zip(*genotype.normal)
        self.c1_d = OPS[op_names[0]](self.channel, self.channel)
        self.c1_r = OPS[op_names[1]](self.channel, self.channel)
        self.c2_d = OPS[op_names[2]](self.channel, self.channel)
        self.c2_r = OPS[op_names[3]](self.channel, self.channel)
        self.c3_d = OPS[op_names[4]](self.channel, self.channel)
        self.c3_r = OPS[op_names[5]](self.channel, self.channel)
        self.c4 = OPS[op_names[6]](self.channel, self.channel)
        self.act = nn.LeakyReLU(negative_slope=0.05, inplace=False)
        self.c5 = nn.Conv2d(self.channel * 4, self.channel, 1)

    def forward(self, x):
        distilled_c1 = self.act(self.c1_d(x))
        r_c1 = self.act(self.c1_r(x) + x)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = self.act(self.c2_r(r_c1) + r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = self.act(self.c3_r(r_c2) + r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out_fused = self.c5(out)

        return out_fused


class IEM(nn.Module):
    def __init__(self, channel, genetype):
        super(IEM, self).__init__()
        self.channel = channel
        self.genetype = genetype
        self.cell = SearchBlock(self.channel, self.genetype)

    def forward(self, x, u):
        u = F.pad(u, (0, 1, 0, 1), "constant", 0)
        u = F.max_pool2d(u, 2, 1, 0)
        t = 0.5 * (u + x)
        t = self.cell(t)
        t = torch.sigmoid(t)
        t = torch.clamp(t, 0.001, 1.0)
        u = torch.clamp(x / t, 0.0, 1.0)
        return u


class EnhanceNetwork(nn.Module):
    def __init__(self, iteratioin, channel, genotype):
        super(EnhanceNetwork, self).__init__()
        self.iem_nums = iteratioin

        self.iems = nn.ModuleList()
        for i in range(self.iem_nums):
            self.iems.append(IEM(channel, genotype))

    def forward(self, x):
        o = x
        for i in range(self.iem_nums):
            o = self.iems[i](x, o)
        return o


class DenoiseNetwork(nn.Module):
    def __init__(self, layers, channel, genotype):
        super(DenoiseNetwork, self).__init__()
        self.layers = layers
        self.stem = nn.Conv2d(3, channel, 3, 1, 1)
        self.nrms = nn.ModuleList()
        for _ in range(layers):
            self.nrms.append(SearchBlock(channel, genotype))
        self.activate = nn.Sequential(nn.Conv2d(channel, 3, 3, 1, 1))

    def forward(self, x):
        feat = self.stem(x)
        for i in range(self.layers):
            feat = self.nrms[i](feat)
        noise = self.activate(feat)
        output = x - noise
        return output


class RUAS(nn.Module):
    def __init__(self, with_denoise=True):
        super(RUAS, self).__init__()
        self.with_denoise = with_denoise
        self.enhance_net = EnhanceNetwork(iteratioin=3, channel=3, genotype=IEM_Geno)
        self.denoise_net = DenoiseNetwork(layers=3, channel=6, genotype=NRM_Geno)

    def forward(self, x):
        x = self.enhance_net(x)
        if self.with_denoise:
            x = self.denoise_net(x)
        return x


if __name__ == '__main__':
    from thop import profile, clever_format

    model = RUAS().cuda().eval()
    print(model)

    inp = torch.rand(1, 3, 600, 400).cuda()

    macs, params = profile(model, inputs=(inp,))
    macs, params = clever_format([macs, params], "%.3f")
    print(f"MACs: {macs}, Parameters: {params}")

    with torch.no_grad():
        output = model(inp)
        print(output.shape)
