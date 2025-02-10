import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Deep Stacked Laplacian Restorer for Low-light Image Enhancement
Authors: Lim, Seokjae and Kim, Wonjun
IEEE Transactions on Multimedia, 2020
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2)
        self.instance = nn.InstanceNorm2d(dim, track_running_stats=False, affine=False)

    def forward(self, x):
        x1 = F.relu(self.instance(self.conv1(x)))
        x = F.relu(self.instance(self.conv2(x1)) + x)
        return x


class LRBLock_3(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super(LRBLock_3, self).__init__()
        self.res_l3 = ResBlock(dim, kernel_size)
        self.res_l2 = ResBlock(dim, kernel_size)
        self.res_l1 = ResBlock(dim, kernel_size)

    def sample(self, x, size=None):
        # Use 'size' instead of 'scale_factor' to avoid shape mismatch
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, x):
        # Downsample by specifying the target size
        x_down2 = self.sample(x, size=(x.size(2) // 2, x.size(3) // 2))  # 128
        x_down4 = self.sample(x_down2, size=(x_down2.size(2) // 2, x_down2.size(3) // 2))  # 64

        # Upsample by specifying the target size
        laplace2 = x_down2 - self.sample(x_down4, size=(x_down2.size(2), x_down2.size(3)))
        laplace1 = x - self.sample(x_down2, size=(x.size(2), x.size(3)))

        scale1 = self.res_l1(x_down4)
        scale2 = self.res_l2(laplace2)
        scale3 = self.res_l3(laplace1)

        output1 = scale1
        output2 = self.sample(scale1, size=(laplace2.size(2), laplace2.size(3))) + scale2
        output3 = self.sample(output2, size=(laplace1.size(2), laplace1.size(3))) + scale3

        return output3


class LRBLock_2(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super(LRBLock_2, self).__init__()
        self.res_l2 = ResBlock(dim, kernel_size)
        self.res_l1 = ResBlock(dim, kernel_size)

    def sample(self, x, size=None):
        # Use 'size' instead of 'scale_factor' to avoid shape mismatch
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, x):
        # Downsample by specifying the target size
        x_down2 = self.sample(x, size=(x.size(2) // 2, x.size(3) // 2))  # 128
        laplace1 = x - self.sample(x_down2, size=(x.size(2), x.size(3)))

        scale1 = self.res_l1(x_down2)
        scale2 = self.res_l2(laplace1)

        output2 = self.sample(scale1, size=(laplace1.size(2), laplace1.size(3))) + scale2

        return output2


class LMSB_Block(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding, lr_block, lr_block_kernel_size=3, up_block=False,
                 last_block=False):
        super(LMSB_Block, self).__init__()
        if not up_block:
            self.model = nn.Sequential(*[
                nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding),
                nn.InstanceNorm2d(out_dim, track_running_stats=False, affine=False),
                nn.ReLU(inplace=True),
                lr_block(out_dim, lr_block_kernel_size)
            ])
        else:
            if last_block:
                self.model = nn.Sequential(*[
                    lr_block(in_dim, lr_block_kernel_size),
                    nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride, padding),
                ])
            else:
                self.model = nn.Sequential(*[
                    lr_block(in_dim, 3),
                    nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride, padding),
                    nn.InstanceNorm2d(out_dim, track_running_stats=False, affine=False),
                    nn.ReLU(inplace=True),
                ])

    def forward(self, x):
        return self.model(x)


class LMSB(nn.Module):
    def __init__(self, dim=64, lr_block=LRBLock_3):
        super(LMSB, self).__init__()
        self.lmsb_block_d1 = LMSB_Block(3, dim * 1, 5, 2, 2, lr_block)
        self.lmsb_block_d2 = LMSB_Block(dim * 1, dim * 2, 5, 2, 2, lr_block)
        self.lmsb_block_d3 = LMSB_Block(dim * 2, dim * 4, 5, 2, 2, lr_block)
        self.lmsb_block_u1 = LMSB_Block(dim * 4, dim * 2, 4, 2, 1, lr_block, 3, True)
        self.lmsb_block_u2 = LMSB_Block(dim * 2, dim * 1, 4, 2, 1, lr_block, 3, True)
        self.lmsb_block_u3 = LMSB_Block(dim * 1, 3, 4, 2, 1, lr_block, 3, True, True)


    def sample(self, x, size):
        # Use 'size' instead of 'scale_factor' to avoid shape mismatch
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, x):
        input_x = x
        x1 = self.lmsb_block_d1(x)
        x2 = self.lmsb_block_d2(x1)
        x3 = self.lmsb_block_d3(x2)
        x = self.lmsb_block_u1(x3)
        x = self.lmsb_block_u2(x + self.sample(x2, size=(x.size(2), x.size(3))))
        x = self.lmsb_block_u3(x + self.sample(x1, size=(x.size(2), x.size(3))))
        return x + self.sample(input_x, size=(x.size(2), x.size(3)))


class DSLR(nn.Module):
    def __init__(self):
        super(DSLR, self).__init__()
        self.stage1 = LMSB(64, LRBLock_3)
        self.stage2 = LMSB(32, LRBLock_2)
        self.stage3 = LMSB(32, LRBLock_2)

    def sample(self, x, size=None):
        # Use 'size' instead of 'scale_factor' to avoid shape mismatch
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, x):
        # Downsample by specifying the target size
        x_down2 = self.sample(x, size=(x.size(2) // 2, x.size(3) // 2))  # 128
        x_down4 = self.sample(x_down2, size=(x_down2.size(2) // 2, x_down2.size(3) // 2))  # 64

        ### Stage 1
        scale1 = self.stage1(x_down4)

        ### Stage 2
        laplace2 = x_down2 - self.sample(x_down4, size=(x_down2.size(2), x_down2.size(3)))
        scale2 = self.stage2(laplace2)
        output2 = self.sample(scale1, size=(scale2.size(2), scale2.size(3))) + scale2

        ### Stage 3
        laplace1 = x - self.sample(x_down2, size=(x.size(2), x.size(3)))
        scale3 = self.stage3(laplace1)
        output3 = self.sample(output2, size=(scale3.size(2), scale3.size(3))) + scale3

        return output3


if __name__ == '__main__':
    from thop import profile, clever_format

    model = DSLR().cuda().eval()
    print(model)
    inp = torch.rand(1, 3, 600, 400).cuda()
    macs, params = profile(model, inputs=(inp,))
    macs, params = clever_format([macs, params], "%.3f")
    print(f"MACs: {macs}, Parameters: {params}")
    with torch.no_grad():
        output = model(inp)
        print(output.shape)