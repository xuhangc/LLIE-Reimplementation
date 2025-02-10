import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Sparse Gradient Regularized Deep Retinex Network for Robust Low-Light Image Enhancement
Yang, Wenhan and Wang, Wenjing and Huang, Haofeng and Wang, Shiqi and Liu, Jiaying
IEEE Transactions on Image Processing, 2021
'''


class RDB_Conv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(RDB_Conv, self).__init__()
        self.conv = nn.Sequential(*[
            nn.Conv2d(in_dim, out_dim, 3, 1, 1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, init_dim, grow_dim, nConvLayers):
        super(RDB, self).__init__()
        convs = []
        for n in range(nConvLayers):
            convs.append(RDB_Conv(init_dim + n * grow_dim, grow_dim))

        self.convs = nn.Sequential(*convs)
        self.LFF = nn.Conv2d(init_dim + nConvLayers *
                             grow_dim, init_dim, 1, 1, 0)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x


class SM_uint(nn.Module):
    def __init__(self, dim=8):
        super(SM_uint, self).__init__()
        self.E = nn.Conv2d(1, dim, 3, 1, 1, 1)
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1, 1)
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 2, 2)
        self.conv3 = nn.Conv2d(dim, dim, 3, 1, 3, 3)
        self.G = nn.Conv2d(dim, 1, 3, 1, 1, 1)

    def forward(self, x):
        out = F.relu(self.E(x))
        out = F.relu(self.conv1(out)) + out
        out = F.relu(self.conv2(out)) + out
        out = F.relu(self.conv3(out)) + out
        out = F.relu(self.G(out))
        return out


class SM(nn.Module):
    def __init__(self):
        super(SM, self).__init__()
        self.u1 = SM_uint()
        self.u2 = SM_uint()
        self.u3 = SM_uint()
        self.u4 = SM_uint()

    def forward(self, x):
        x = self.u1(x.detach())
        x = self.u2(x.detach())
        x = self.u3(x.detach())
        x = self.u4(x.detach())
        return x


class RDBBlock(nn.Module):
    def __init__(self, init_dim=32, grow_dim=16, nConvLayers=6, rdb_nums=4, last_block=False):
        super(RDBBlock, self).__init__()
        self.last_block = last_block
        self.RDBs = nn.ModuleList()
        for i in range(rdb_nums):
            self.RDBs.append(RDB(init_dim, grow_dim, nConvLayers))
        if self.last_block:
            self.GFF = nn.Sequential(*[
                nn.Conv2d(rdb_nums * init_dim, init_dim, 1, 1, 0),
                nn.Conv2d(init_dim, init_dim, 3, 1, 1)
            ])
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(init_dim, grow_dim, 3, 1, 1),
                nn.Conv2d(grow_dim, 3, 3, 1, 1)
            ])
        else:
            self.compress = nn.Conv2d(rdb_nums * init_dim, init_dim, 1, 1, 0)
            self.UpConv = nn.Conv2d(init_dim, init_dim, 3, 1, 1)
            self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x_ = x
        RDBs_out = []
        for i in range(len(self.RDBs)):
            x = self.RDBs[i](x)
            RDBs_out.append(x)
        RDBs_out_feat = torch.cat(RDBs_out, 1)
        if self.last_block:
            x_ = x_ + self.GFF(RDBs_out_feat)
            return self.UPNet(x_)
        else:
            x_ = x_ + self.compress(RDBs_out_feat)
            return self.UpConv(self.Up(x_))


class LRDN(nn.Module):
    def __init__(self, init_dim=32, grow_dim=16, nConvLayers=6, rdb_nums=4):
        super(LRDN, self).__init__()

        self.SFENet1 = nn.Conv2d(3, init_dim, 3, 1, 1)
        self.SFENet2 = nn.Conv2d(init_dim, init_dim, 3, 1, 1)
        self.Down1 = nn.Conv2d(init_dim, init_dim, 3, 2, 1)
        self.Down2 = nn.Conv2d(init_dim, init_dim, 3, 2, 1)
        self.Down3 = nn.Conv2d(init_dim, init_dim, 3, 2, 1)
        self.RDBs1 = RDBBlock(init_dim, grow_dim, nConvLayers, rdb_nums, last_block=True)
        self.RDBs2 = RDBBlock(init_dim, grow_dim, nConvLayers, rdb_nums)
        self.RDBs3 = RDBBlock(init_dim, grow_dim, nConvLayers, rdb_nums)
        self.RDBs4 = RDBBlock(init_dim, grow_dim, nConvLayers, rdb_nums)

    def forward(self, x):
        x = self.SFENet1(x)
        x0 = self.SFENet2(x)
        x1 = self.Down1(x)
        x2 = self.Down2(x1)
        x3 = self.Down2(x2)

        x3_out = self.RDBs4(x3)
        x2_out = self.RDBs3(x2 + x3_out)
        x1_out = self.RDBs2(x1 + x2_out)
        x0_out = self.RDBs1(x0 + x1_out)
        return x0_out


class SGM(nn.Module):
    def __init__(self, ):
        super(SGM, self).__init__()
        self.decom = SM()
        self.restore = LRDN()
        self.enhance = LRDN()

    def forward(self, S_low):
        # 1. Decomposition
        S_low = (S_low + 0.05) / 1.05
        R_low = self.decom(torch.max(S_low, dim=1, keepdim=True)[0])
        torch.clamp_(R_low, 21 ** -1, 1)

        # 2. Adjustment
        I_normal_pred = self.enhance(S_low)
        torch.clamp_(I_normal_pred, 21 ** -1, 1)
        R_normal_pred = self.restore(S_low / R_low.detach())

        # 3. Reconstruct
        S_normal_pred = I_normal_pred * R_normal_pred
        S_normal_pred = S_normal_pred * 1.05 - 0.05
        return S_normal_pred


if __name__ == '__main__':
    from thop import profile, clever_format

    model = SGM().cuda().eval()
    print(model)

    inp = torch.rand(1, 3, 256, 256).cuda()

    macs, params = profile(model, inputs=(inp,))
    macs, params = clever_format([macs, params], "%.3f")
    print(f"MACs: {macs}, Parameters: {params}")

    with torch.no_grad():
        output = model(inp)
        print(output.shape)
