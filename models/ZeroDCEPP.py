import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Learning to enhance low-light image via zero-reference deep curve estimation
Authors: Li, Chongyi and Guo, Chunle and Loy, Chen Change
IEEE transactions on pattern analysis and machine intelligence, 2021
'''


class CSDN_Tem(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CSDN_Tem, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class ZeroDCEPP(nn.Module):
    def __init__(self, scale_factor=12, dim=32):
        super(ZeroDCEPP, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)

        self.e_conv1 = CSDN_Tem(3, dim)
        self.e_conv2 = CSDN_Tem(dim, dim)
        self.e_conv3 = CSDN_Tem(dim, dim)
        self.e_conv4 = CSDN_Tem(dim, dim)
        self.e_conv5 = CSDN_Tem(dim * 2, dim)
        self.e_conv6 = CSDN_Tem(dim * 2, dim)
        self.e_conv7 = CSDN_Tem(dim * 2, 3)

    def forward(self, x):
        x_down = F.interpolate(x, scale_factor=1 / self.scale_factor, mode='bilinear', align_corners=True,
                               recompute_scale_factor=True)

        x1 = self.relu(self.e_conv1(x_down))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        x_r = F.interpolate(x_r, x.shape[2:], mode='bilinear', align_corners=True)
        for _ in range(8):
            x = x + x_r * (torch.pow(x, 2) - x)
        return x


if __name__ == '__main__':
    from thop import profile, clever_format

    model = ZeroDCEPP().cuda().eval()
    print(model)

    inp = torch.rand(1, 3, 256, 256).cuda()

    macs, params = profile(model, inputs=(inp,))
    macs, params = clever_format([macs, params], "%.3f")
    print(f"MACs: {macs}, Parameters: {params}")

    with torch.no_grad():
        output = model(inp)
        print(output.shape)
