import torch
import torch.nn as nn

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x1 = self.conv1(x)
        return self.sigmoid(x1)

class CAF(nn.Module):
    def __init__(self):
        super(CAF, self).__init__()
        self.sa = SpatialAttention()
    def forward(self, x1_rgb, x1_d):
        x1_add = self.sa(x1_rgb + x1_d) * (x1_rgb + x1_d)
        x1_mul = self.sa(x1_rgb * x1_d) * (x1_rgb * x1_d)
        x1_sub = self.sa(x1_rgb - x1_d) * (x1_rgb - x1_d)
        fuse_C = x1_sub + x1_mul + x1_add
        # fuse_C = x1_mul + x1_add
        return fuse_C

class MAc(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MAc, self).__init__()

        self.r_d1 = nn.Conv2d(in_channel, out_channel, 3, padding=1, dilation=1)
        self.r_d3 = nn.Conv2d(in_channel, out_channel, 3, padding=3, dilation=3)
        self.r_d5 = nn.Conv2d(in_channel, out_channel, 3, padding=5, dilation=5)
        self.r_d7 = nn.Conv2d(in_channel, out_channel, 3, padding=7, dilation=7)

        self.d_d1 = nn.Conv2d(in_channel, out_channel, 3, padding=1, dilation=1)
        self.d_d3 = nn.Conv2d(in_channel, out_channel, 3, padding=3, dilation=3)
        self.d_d5 = nn.Conv2d(in_channel, out_channel, 3, padding=5, dilation=5)
        self.d_d7 = nn.Conv2d(in_channel, out_channel, 3, padding=7, dilation=7)

        self.CAF = CAF()

        self.conv = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=1),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )
    def forward(self, rf, df):

        d1_r = self.r_d1(rf)
        d3_r = self.r_d3(rf)
        d5_r = self.r_d5(rf)
        d7_r = self.r_d7(rf)
        r13_diff = d1_r - d3_r
        r15_diff = d1_r - d5_r
        r17_diff = d1_r - d7_r

        d1_d = self.d_d1(df)
        d3_d = self.d_d3(df)
        d5_d = self.d_d5(df)
        d7_d = self.d_d7(df)
        d13_diff = d1_d - d3_d
        d15_diff = d1_d - d5_d
        d17_diff = d1_d - d7_d

        # fuse_r_diff = torch.cat((r13_diff, r15_diff, r17_diff), dim=1)
        fuse_r_diff = r13_diff * r15_diff * r17_diff

        fuse_d_diff = d13_diff * d15_diff * d17_diff

        fuse_rd = self.CAF(1 - fuse_r_diff, 1 - fuse_d_diff)

        out_fuse = self.conv(fuse_rd)

        return out_fuse


if __name__ == '__main__':
    rgb = torch.randn(2,64,256,256)
    dep = torch.randn(2,64,256,256)
    fuse = MAc(64, 64)
    out = fuse(rgb, dep)
    print(out.shape)