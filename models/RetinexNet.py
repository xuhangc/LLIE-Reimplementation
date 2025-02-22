import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Deep Retinex Decomposition for Low-Light Enhancement
Authors: Chen Wei, Wenjing Wang, Wenhan Yang, Jiaying Liu
BMVC, 2018
'''


class DecomNet(nn.Module):

    def __init__(self, layer_num=5, channel=64, kernel_size=3):
        super(DecomNet, self).__init__()
        self.layer_num = layer_num
        self.conv0 = nn.Conv2d(4, channel, kernel_size * 3, padding=4)
        feature_conv = []
        for _ in range(layer_num):
            feature_conv.append(nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size, padding=1),
                nn.ReLU(inplace=True)
            ))
        self.conv = nn.ModuleList(feature_conv)
        self.conv1 = nn.Conv2d(channel, 4, kernel_size, padding=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x_max = torch.max(x, dim=1, keepdim=True)
        x = torch.cat((x, x_max[0]), dim=1)

        out = self.conv0(x)
        for idx in range(self.layer_num):
            out = self.conv[idx](out)
        out = self.conv1(out)
        out = self.sig(out)

        r_part = out[:, 0:3, :, :]
        l_part = out[:, 3:4, :, :]

        return r_part, l_part


class RelightNet(nn.Module):

    def __init__(self, channel=64, kernel_size=3):
        super(RelightNet, self).__init__()
        self.conv0 = nn.Conv2d(4, channel, kernel_size, padding=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.deconv1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size, padding=1),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size, padding=1),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size, padding=1),
            nn.ReLU(inplace=True)
        )
        self.feature_fusion = nn.Conv2d(channel * 3, channel, 1)
        self.output = nn.Conv2d(channel, 1, kernel_size, padding=1)

    def forward(self, r_part, l_part):
        x = torch.cat((r_part, l_part), dim=1)

        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv1(conv1)
        conv3 = self.conv1(conv2)

        up1 = F.interpolate(conv3, scale_factor=2)
        deconv1 = self.deconv1(up1) + conv2
        up2 = F.interpolate(deconv1, scale_factor=2)
        deconv2 = self.deconv2(up2) + conv1
        up3 = F.interpolate(deconv2, scale_factor=2)
        deconv3 = self.deconv3(up3) + conv0

        deconv1_resize = F.interpolate(deconv1, scale_factor=4)
        deconv2_resize = F.interpolate(deconv2, scale_factor=2)

        out = torch.cat((deconv1_resize, deconv2_resize, deconv3), dim=1)
        out = self.feature_fusion(out)
        out = self.output(out)

        return out


class RetinexNet(nn.Module):
    def __init__(self):
        super(RetinexNet, self).__init__()
        self.decom = DecomNet()
        self.enhance = RelightNet()

    def forward(self, x):
        R, L = self.decom(x)
        S = self.enhance(R, L)
        return R * S


if __name__ == '__main__':
    from thop import profile, clever_format

    model = RetinexNet().cuda().eval()
    print(model)

    inp = torch.rand(1, 3, 256, 256).cuda()

    macs, params = profile(model, inputs=(inp,))
    macs, params = clever_format([macs, params], "%.3f")
    print(f"MACs: {macs}, Parameters: {params}")

    with torch.no_grad():
        output = model(inp)
        print(output.shape)
