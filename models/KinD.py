"""
Kindling the darkness: A practical low-light image enhancer
Authors: Zhang, Yonghua and Zhang, Jiawan and Guo, Xiaojie
ACM MM, 2019
Beyond brightening low-light images
Authors: Zhang, Yonghua and Guo, Xiaojie and Ma, Jiayi and Liu, Wei and Zhang, Jiawan
IJCV, 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSIA(nn.Module):
    """
    Multi-scale Illumination Attention Module
    This module performs multi-scale illumination attention to capture the illumination information of the input image.
    """

    def __init__(self, filters, activation='lrelu'):
        super().__init__()
        # Down1
        self.conv_bn_relu1 = ConvBNReLU(filters, activation)

        # Down2
        self.down_2 = MaxPooling2D(2, 2)
        self.conv_bn_relu2 = ConvBNReLU(filters, activation)
        self.deconv_2 = ConvTranspose2D(filters, filters)

        # Down4
        self.down_4 = MaxPooling2D(2, 2)
        self.conv_bn_relu4 = ConvBNReLU(filters, activation, kernel=1)
        # TODO: why are we using kernel = 1 here?
        self.deconv_4_1 = ConvTranspose2D(filters, filters)
        self.deconv_4_2 = ConvTranspose2D(filters, filters)

        # output
        self.out = Conv2DandReLU(filters * 4, filters)

    def forward(self, R, I_att):
        """
        R : Reflectance
        I_att : Illumination attention
        """
        R_att = R * I_att  # WARN: why are we multiplying R and I_att?

        # Down 1
        msia_1 = self.conv_bn_relu1(R_att)

        # Down 2
        down_2 = self.down_2(R_att)
        conv_bn_relu_2 = self.conv_bn_relu2(down_2)
        msia_2 = self.deconv_2(conv_bn_relu_2)

        # Down 4
        down_4 = self.down_4(down_2)
        conv_bn_relu_4 = self.conv_bn_relu4(down_4)
        deconv_4 = self.deconv_4_1(conv_bn_relu_4)
        msia_4 = self.deconv_4_2(deconv_4)

        # Concat
        concat = torch.cat([R, msia_1, msia_2, msia_4], dim=1)
        # NOTE: Revise this part

        out = self.out(concat)
        return out


class ConvBNReLU(nn.Module):
    """
    Convolution + BatchNorm + ReLU
    """

    def __init__(self, channels, activation='lrelu', kernel=3):
        super().__init__()
        self.activation_layer = nn.LeakyReLU(inplace=True) if activation == 'lrelu' else nn.ReLU(inplace=True)
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel, padding=kernel // 2),  # TODO: padding
            nn.BatchNorm2d(channels, momentum=0.99),
            self.activation_layer,
        )

    def forward(self, x):
        return self.conv_bn_relu(x)


class DoubleConv(nn.Module):
    """
    Double Convolution
    This module performs two convolution operations followed by a ReLU activation function.
    """

    def __init__(self, in_channels, out_channels, activation='lrelu'):
        super().__init__()
        self.doubleconv = nn.Sequential(
            Conv2DandReLU(in_channels, out_channels, activation),
            Conv2DandReLU(out_channels, out_channels, activation),
        )

    def forward(self, x):
        return self.doubleconv(x)


class ResConv(nn.Module):
    """
    Residual Convolution
    This module performs a residual convolution operation.
    In residual convolution, the input is added to the output of the convolution operation.
    """

    def __init__(self, in_channels, out_channels, activation='lrelu'):
        super().__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True) if activation == 'lrelu' else nn.ReLU(inplace=True)
        # NOTE: we have used a different slope value for the LeakyReLU activation function

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=0.99)
        self.cbam = CBAM(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.99)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.relu(bn1)
        cbam = self.cbam(relu1)
        conv2 = self.conv2(cbam)
        bn2 = self.bn2(conv2)
        out = x + bn2
        return out


class Conv2DandReLU(nn.Module):
    """
    Convolution + ReLU
    This module performs the downsampling operation
    Kernel size is fixed to 3x3 and stride is 1
    """

    def __init__(self, in_channels, out_channels, activation='lrelu', stride=1):
        super().__init__()
        self.activation_layer = nn.LeakyReLU(inplace=True) if activation == 'lrelu' else nn.ReLU(inplace=True)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            self.activation_layer,
        )

    def forward(self, x):
        return self.conv_relu(x)


class ConvTranspose2D(nn.Module):
    """
    Transposed Convolution + ReLU
    This module performs the upsampling operation
    """

    def __init__(self, in_channels, out_channels, activation='lrelu'):
        super().__init__()
        self.activation_layer = nn.LeakyReLU(inplace=True) if activation == 'lrelu' else nn.ReLU(inplace=True)
        self.deconv_relu = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
            # WARN: removed output padding from orig code
            self.activation_layer,
        )

    def forward(self, x):
        return self.deconv_relu(x)


class MaxPooling2D(nn.Module):
    """
    Max Pooling
    This module perform Max Pooling operation, which is used in the downsampling path
    """

    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return self.maxpool(x)


class AvgPooling2D(nn.Module):
    """
    Average Pooling
    This module perform Average Pooling operation, which is used in the upsampling path
    """

    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return self.avgpool(x)


class ChannelAttention(nn.Module):
    """
    Channel Attention Module
    This implements a channel attention module that adaptively recalibrates channel-wise feature responses
    by modeling interdependencies between channels.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # TODO: AdaptiveAvgPool2d

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.sharedMLP(self.avg_pool(x))
        max_out = self.sharedMLP(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module
    This implements a spatial attention module to perform adaptive spatial feature recalibration.
    """

    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        # TODO: why are input and output channels 2 and 1?
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # TODO: dim=1?
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # TODO: dim=1?
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module

    This module implements a convolutional block attention module that adaptively recalibrates
    channel-wise and spatial-wise feature responses by explicitly modeling interdependencies
    between channels and spatial locations.
    """

    def __init__(self, channels):
        super().__init__()
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x) * x  # TODO: why multiply?
        out = self.spatial_attention(x) * x
        return out


class Concat(nn.Module):
    """
    Concatenation
    This module performs concatenation of two tensors along the channel dimension
    """

    def forward(self, x, y):
        """
        We first calculate the difference in height and width between the two tensors.
        Then we pad the smaller tensor with zeros on all sides so that it has the same height and width as the larger tensor.
        y is always the smaller tensor
        Finally, we concatenate the two tensors along the channel dimension.
        """
        _, _, xH, xW = x.size()
        _, _, yH, yW = y.size()
        diffY = xH - yH
        diffX = xW - yW
        y = F.pad(y, (diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2))
        return torch.cat([x, y], dim=1)


class DecomNet(nn.Module):
    """
    Decomposition Net Class
    This class defines the architecture of the Decomposition Net.
    """

    def __init__(self, filters=32, activation='lrelu'):
        """
        layers are named as _r1, _r2 for reflectance path and _i1, _i2 for illumination path
        """
        super().__init__()

        self.conv_input = Conv2DandReLU(3, filters)
        # Top path for Reflectance
        self.maxpool_r1 = MaxPooling2D()
        self.conv_and_relu_r1 = Conv2DandReLU(filters, filters * 2)
        self.maxpool_r2 = MaxPooling2D()
        self.conv_and_relu_r2 = Conv2DandReLU(filters * 2, filters * 4)
        self.deconv_r1 = ConvTranspose2D(filters * 4, filters * 2)
        self.concat_r1 = Concat()
        self.conv_and_relu_r3 = Conv2DandReLU(filters * 4, filters * 2)
        self.deconv_r2 = ConvTranspose2D(filters * 2, filters)
        self.concat_r2 = Concat()
        self.conv_and_relu_r4 = Conv2DandReLU(filters * 2, filters)
        self.conv_r5 = nn.Conv2d(filters, 3, kernel_size=3, padding=1)  # WARN: padding
        self.sigmoid_r = nn.Sigmoid()

        # Bottom path for Illumination
        self.conv_and_relu_i1 = Conv2DandReLU(filters, filters)
        self.concat_i1 = Concat()
        self.conv_i1 = nn.Conv2d(filters * 2, 1, kernel_size=3, padding=1)
        self.sigmoid_i1 = nn.Sigmoid()

    def forward(self, x):
        decom_conv1 = self.conv_input(x)
        decom_pool1 = self.maxpool_r1(decom_conv1)
        decom_conv2 = self.conv_and_relu_r1(decom_pool1)
        decom_pool2 = self.maxpool_r2(decom_conv2)
        decom_conv3 = self.conv_and_relu_r2(decom_pool2)
        decom_up1 = self.deconv_r1(decom_conv3)
        decom_concat1 = self.concat_r1(decom_up1, decom_conv2)
        decom_conv4 = self.conv_and_relu_r3(decom_concat1)
        decom_up2 = self.deconv_r2(decom_conv4)
        decom_concat2 = self.concat_r2(decom_up2, decom_conv1)
        decom_conv5 = self.conv_and_relu_r4(decom_concat2)
        decom_conv6 = self.conv_r5(decom_conv5)
        decom_R = self.sigmoid_r(decom_conv6)

        decom_i_conv1 = self.conv_and_relu_i1(decom_conv1)
        decom_i_conv2 = self.concat_i1(decom_i_conv1, decom_conv5)
        decom_i_conv3 = self.conv_i1(decom_i_conv2)
        decom_L = self.sigmoid_i1(decom_i_conv3)

        return decom_R, decom_L


class IllumNet(nn.Module):
    """
    Illumination Net Class.
    This class defines the architecture of the Illumination Net.
    """

    def __init__(self, filters=32, activation='lrelu'):
        super().__init__()
        self.concat_1 = Concat()
        self.conv_and_relu_1 = Conv2DandReLU(2, filters)
        self.conv_and_relu_2 = Conv2DandReLU(filters, filters)
        self.conv_and_relu_3 = Conv2DandReLU(filters, filters)
        self.conv_1 = nn.Conv2d(filters, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, L, ratio):
        with torch.no_grad():
            ratio_map = torch.ones_like(L) * ratio
        adjust_concat1 = self.concat_1(L, ratio_map)
        adjust_conv1 = self.conv_and_relu_1(adjust_concat1)
        adjust_conv2 = self.conv_and_relu_2(adjust_conv1)
        adjust_conv3 = self.conv_and_relu_3(adjust_conv2)
        adjust_conv4 = self.conv_1(adjust_conv3)
        adjust_L = self.sigmoid(adjust_conv4)
        # print(f'{adjust_I.requires_grad=}')
        return adjust_L


# NOTE: skipped Restorenet_msia and custom_illum classes

class RestoreNetMSIA(nn.Module):
    def __init__(self, filters=16, activation='relu'):
        super().__init__()
        # Illumination Attention
        self.i_input = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.i_att = nn.Sigmoid()
        # NOTE: Revise this

        # Layers
        self.conv1_1 = Conv2DandReLU(3, filters, activation)
        self.conv1_2 = Conv2DandReLU(filters, filters * 2, activation)
        self.msia1 = MSIA(filters * 2, activation)

        self.conv2_1 = Conv2DandReLU(filters * 2, filters * 4, activation)
        self.conv2_2 = Conv2DandReLU(filters * 4, filters * 4, activation)
        self.msia2 = MSIA(filters * 4, activation)

        self.conv3_1 = Conv2DandReLU(filters * 4, filters * 8, activation)
        # NOTE: no dropout
        self.conv3_2 = Conv2DandReLU(filters * 8, filters * 4, activation)
        self.msia3 = MSIA(filters * 4, activation)

        self.conv4_1 = Conv2DandReLU(filters * 4, filters * 2, activation)
        self.conv4_2 = Conv2DandReLU(filters * 2, filters * 2, activation)
        self.msia4 = MSIA(filters * 2, activation)

        self.conv5_1 = Conv2DandReLU(filters * 2, filters, activation)
        self.conv5_2 = nn.Conv2d(filters, 3, kernel_size=3, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, R, I):
        i_input = self.i_input(I)
        i_att = self.i_att(i_input)

        # Network
        conv1 = self.conv1_1(R)
        conv1 = self.conv1_2(conv1)
        msia1 = self.msia1(conv1, i_att)

        conv2 = self.conv2_1(msia1)
        conv2 = self.conv2_2(conv2)
        msia2 = self.msia2(conv2, i_att)

        conv3 = self.conv3_1(msia2)
        conv3 = self.conv3_2(conv3)
        msia3 = self.msia3(conv3, i_att)

        conv4 = self.conv4_1(msia3)
        conv4 = self.conv4_2(conv4)
        msia4 = self.msia4(conv4, i_att)

        conv5 = self.conv5_1(msia4)
        conv5 = self.conv5_2(conv5)

        R = self.sigmoid(conv5)  # WARN: DIFF
        return R


class RestoreNet_Unet(nn.Module):
    """
    RestoreNet Class
    This class defines the architecture of the RestoreNet.
    """

    def __init__(self, filters=32, activation='lrelu'):
        super().__init__()
        self.concat_1 = Concat()
        self.conv_and_relu_1 = Conv2DandReLU(4, filters)
        self.conv_and_relu_2 = Conv2DandReLU(filters, filters)
        self.maxpool_1 = MaxPooling2D()

        self.conv_and_relu_3 = Conv2DandReLU(filters, filters * 2)
        self.conv_and_relu_4 = Conv2DandReLU(filters * 2, filters * 2)
        self.maxpool_2 = MaxPooling2D()

        self.conv_and_relu_5 = Conv2DandReLU(filters * 2, filters * 4)
        self.conv_and_relu_6 = Conv2DandReLU(filters * 4, filters * 4)
        self.maxpool_3 = MaxPooling2D()

        self.conv_and_relu_7 = Conv2DandReLU(filters * 4, filters * 8)
        self.conv_and_relu_8 = Conv2DandReLU(filters * 8, filters * 8)
        self.maxpool_4 = MaxPooling2D()

        self.conv_and_relu_9 = Conv2DandReLU(filters * 8, filters * 16)
        self.conv_and_relu_10 = Conv2DandReLU(filters * 16, filters * 16)
        # WARN: Dropout removed
        self.deconv_1 = ConvTranspose2D(filters * 16, filters * 8)
        self.concat_2 = Concat()

        self.conv_and_relu_11 = Conv2DandReLU(filters * 16, filters * 8)
        self.conv_and_relu_12 = Conv2DandReLU(filters * 8, filters * 8)
        self.deconv_2 = ConvTranspose2D(filters * 8, filters * 4)
        self.concat_3 = Concat()

        self.conv_and_relu_13 = Conv2DandReLU(filters * 8, filters * 4)
        self.conv_and_relu_14 = Conv2DandReLU(filters * 4, filters * 4)
        self.deconv_3 = ConvTranspose2D(filters * 4, filters * 2)
        self.concat_4 = Concat()

        self.conv_and_relu_15 = Conv2DandReLU(filters * 4, filters * 2)
        self.conv_and_relu_16 = Conv2DandReLU(filters * 2, filters * 2)
        self.deconv_4 = ConvTranspose2D(filters * 2, filters)
        self.concat_5 = Concat()

        self.conv_and_relu_17 = Conv2DandReLU(filters * 2, filters)
        self.conv_and_relu_18 = Conv2DandReLU(filters, filters)
        # WARN: Paper has 256 in output, but implementation has 32 here. We are using 256

        self.conv_1 = nn.Conv2d(filters, 3, kernel_size=3, stride=1, padding=1)  # WARN: padding
        self.sigmoid = nn.Sigmoid()

    def forward(self, R, L):
        """
        R: output decom_conv_5 of DecomNet  #WARN should it be decom_conv_6 in paper?
        L: output decom_i_conv3 of DecomNet
        # WARN: in what order should they be concatenated?
        """

        # x = torch.cat([R, I], dim=1)
        # re_concat1 = self.concat_1(x)
        # WARN: replacing above 2 lines with 1 below
        # print(f'{R.shape=}, {I.shape=}')
        re_concat1 = self.concat_1(R, L)
        # print(re_concat1.shape)
        re_conv1_1 = self.conv_and_relu_1(re_concat1)
        re_conv1_2 = self.conv_and_relu_2(re_conv1_1)
        re_pool1 = self.maxpool_1(re_conv1_2)

        re_conv2_1 = self.conv_and_relu_3(re_pool1)
        re_conv2_2 = self.conv_and_relu_4(re_conv2_1)
        re_pool2 = self.maxpool_2(re_conv2_2)

        re_conv3_1 = self.conv_and_relu_5(re_pool2)
        re_conv3_2 = self.conv_and_relu_6(re_conv3_1)
        re_pool3 = self.maxpool_3(re_conv3_2)

        re_conv4_1 = self.conv_and_relu_7(re_pool3)
        re_conv4_2 = self.conv_and_relu_8(re_conv4_1)
        re_pool4 = self.maxpool_4(re_conv4_2)

        re_conv5_1 = self.conv_and_relu_9(re_pool4)
        re_conv5_2 = self.conv_and_relu_10(re_conv5_1)
        re_up1 = self.deconv_1(re_conv5_2)
        re_concat2 = self.concat_2(re_conv4_2, re_up1)

        re_conv6_1 = self.conv_and_relu_11(re_concat2)
        re_conv6_2 = self.conv_and_relu_12(re_conv6_1)
        re_up2 = self.deconv_2(re_conv6_2)
        re_concat3 = self.concat_3(re_conv3_2, re_up2)

        re_conv7_1 = self.conv_and_relu_13(re_concat3)
        re_conv7_2 = self.conv_and_relu_14(re_conv7_1)
        re_up3 = self.deconv_3(re_conv7_2)
        re_concat4 = self.concat_4(re_conv2_2, re_up3)

        re_conv8_1 = self.conv_and_relu_15(re_concat4)
        re_conv8_2 = self.conv_and_relu_16(re_conv8_1)
        re_up4 = self.deconv_4(re_conv8_2)
        re_concat5 = self.concat_5(re_conv1_2, re_up4)

        re_conv9_1 = self.conv_and_relu_17(re_concat5)
        re_conv9_2 = self.conv_and_relu_18(re_conv9_1)

        re_conv10 = self.conv_1(re_conv9_2)
        re_R = self.sigmoid(re_conv10)
        return re_R


class KinD_noDecom(nn.Module):
    """
    The entire network for KinD without decomposition net
    """

    def __init__(self, filters=32, activation='lrelu'):
        super().__init__()
        self.restore_net = RestoreNet_Unet(filters, activation)  # WARN: params missing in original
        self.illum_net = IllumNet(filters, activation)

    def forward(self, R, L, ratio):
        L_final = self.illum_net(L, ratio)
        R_final = self.restore_net(R, L)  # WARN: should pass I or I_final?, (mostly I)
        L_final_3 = torch.cat([L_final, L_final, L_final], dim=1)  # WARN: why dim=1?
        out = L_final_3 * R_final
        return R_final, L_final, out


class KinD(nn.Module):
    def __init__(self, filters=32, activation='lrelu'):
        super().__init__()

        # WARN: DIFF FROM ORIGINAL CODE
        self.decom_net = DecomNet(filters, activation)
        self.restore_net = RestoreNet_Unet(filters, activation)
        self.illum_net = IllumNet(filters, activation)
        self.KinD_noDecom = KinD_noDecom(filters, activation)
        self.KinD_noDecom.restore_net = self.restore_net  # NOTE: overwrite restore_net and illum_net?
        self.KinD_noDecom.illum_net = self.illum_net
        self.ratio = nn.Parameter(torch.ones(1))

    def forward(self, I):
        R, L = self.decom_net(I)
        R_final, L_final, out = self.KinD_noDecom(R, L, self.ratio)
        return R_final, L_final, out


class KinDPP(nn.Module):
    def __init__(self, filters=32, activation='lrelu'):
        super().__init__()
        self.decom_net = DecomNet(filters, activation)
        self.restore_net = RestoreNetMSIA(filters, activation)
        self.illum_net = IllumNet(filters, activation)
        self.ratio = nn.Parameter(torch.ones(1))

    def forward(self, I):
        R, L = self.decom_net(I)
        L_final = self.illum_net(L, self.ratio)
        R_final = self.restore_net(R, L)  # WARN: should pass L or L_final?
        L_final_3 = torch.cat([L_final, L_final, L_final], dim=1)
        if L_final_3.shape[2:] != R_final.shape[2:]:
            R_final = F.interpolate(R_final, L_final_3.shape[2:])
        out = L_final_3 * R_final
        return R_final, L_final, out


if __name__ == '__main__':
    model = KinD().cuda().eval()
    print(model)

    inp = torch.rand(1, 3, 256, 256).cuda()

    with torch.no_grad():
        _, _, output = model(inp)
        print(output.shape)