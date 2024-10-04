import torch
import torch.nn as nn
import torch.nn.functional as F
import kmeans1d
affine_par = True

class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=1, s=1, p=0, g=1, d=1, bias=False, bn=False, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)

class DSConv1x1(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv1x1, self).__init__()
        self.conv = nn.Sequential(
                convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu),
                )

    def forward(self, x):
        return self.conv(x)

class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
                convbnrelu(in_channel, out_channel,  k=3, s=stride, p=dilation, d=dilation, g=in_channel, relu=relu),
                )

    def forward(self, x):
        return self.conv(x)
# class MutilPooling(nn.Module):
#     """
#     Reference:
#     """
#     def __init__(self, in_channels, pool_size, norm_layer=nn.BatchNorm2d):
#         super(MutilPooling, self).__init__()
#         self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
#         self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])
#         self.pool3 = nn.AdaptiveAvgPool2d((1, None))
#         self.pool4 = nn.AdaptiveAvgPool2d((None, 1))
#
#         inter_channels = int(in_channels/4)
#         self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
#                                 norm_layer(inter_channels),
#                                 nn.ReLU(True))
#         self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
#                                 norm_layer(inter_channels),
#                                 nn.ReLU(True))
#         self.conv2_0 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
#                                 norm_layer(inter_channels))
#         self.conv2_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
#                                 norm_layer(inter_channels))
#         self.conv2_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
#                                 norm_layer(inter_channels))
#         self.conv2_3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
#                                 norm_layer(inter_channels))
#         self.conv2_4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
#                                 norm_layer(inter_channels))
#         self.conv2_5 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
#                                 norm_layer(inter_channels),
#                                 nn.ReLU(True))
#         self.conv2_6 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
#                                 norm_layer(inter_channels),
#                                 nn.ReLU(True))
#         self.conv3 = nn.Sequential(nn.Conv2d(inter_channels*2, in_channels, 1, bias=False),
#                                 norm_layer(in_channels))
#         # # bilinear interpolate options
#         # self._up_kwargs = up_kwargs
#
#     def forward(self, x):
#         _, _, h, w = x.size()
#         x1 = self.conv1_1(x)
#         x2 = self.conv1_2(x)
#         x2_1 = self.conv2_0(x1)
#         x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), (h, w))
#         x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), (h, w))
#         x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), (h, w))
#         x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), (h, w))
#         x1 = self.conv2_5(F.relu_(x2_1 + x2_2 + x2_3))
#         x2 = self.conv2_6(F.relu_(x2_5 + x2_4))
#         out = self.conv3(torch.cat([x1, x2], dim=1))
#         return F.relu_(x + out)

class SCDC(nn.Module):

    def __init__(self, inplanes, need_cluster=False):
        super(SCDC, self).__init__()
        self.margin = 0
        self.IN = nn.InstanceNorm2d(inplanes, affine=affine_par)
        self.CFR_branches = nn.ModuleList()
        self.CFR_branches.append(nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False))
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.mask_matrix = None
        self.CR1 = DSConv1x1(inplanes, inplanes)
        self.CR3 = DSConv3x3(inplanes, inplanes)
        self.only_con1 = DSConv1x1(inplanes, inplanes, relu=False)
        # self.MP = MutilPooling(inplanes, (h, w))
        self.need_cluster = need_cluster
    def Regional_Normalization(self, region_mask, x):
        masked = x * region_mask
        RN_feature_map = self.IN(masked)
        return RN_feature_map

    def set_class_mask_matrix(self, normalized_map):

        b,c,h,w = normalized_map.size()
        var_flatten = torch.flatten(normalized_map)

        # kmeans1d clustering setting for RN block
        clusters, centroids = kmeans1d.cluster(var_flatten, 5)
        # print('var_flatten.size()[0]',var_flatten.size()[0])
        # print('clusters.count(0)',clusters.count(0))

        num_category = var_flatten.size()[0] - clusters.count(0)  # 1: class-region, 2~5: background
        _, indices = torch.topk(var_flatten, k=int(num_category))
        mask_matrix = torch.flatten(torch.zeros(b, c, h, w).cuda())
        mask_matrix[indices] = 1
        # except:
        #     mask_matrix = torch.ones(var_flatten.size()[0]).cuda()
        #     print('xxxxx')

        mask_matrix = mask_matrix.view(b, c, h, w)

        return mask_matrix

    def forward(self, x):
        outs = []
        idx = 0
        # mask_softmax = F.softmax(mask_input, dim=1)
        x_CR1 = self.CR1(x)
        x_onlycon1 = self.only_con1(x)
        # mask_mean = torch.mean(mask_softmax, dim=1, keepdim=True)
        x_CR1_sin = torch.sin(x_onlycon1)
        x_CR1_cos = torch.cos(x_onlycon1)
        x_sc = x_CR1_sin + x_CR1_cos
        x_sc_sin = self.CR1(torch.sin(x_sc))
        x_sc_cos = self.CR1(torch.cos(x_sc))
        x_scsc = x_sc_sin * x_sc_cos

        x_scsc_ = x_scsc + x
        x_scsc_CBR3 = self.CR3(x_scsc_)

        x_reduct = x_CR1 + x_scsc_CBR3

        mid = x_reduct

        avg_out = torch.mean(mid, dim=1, keepdim=True)
        max_out,_ = torch.max(mid, dim=1, keepdim=True)

        atten = torch.cat([avg_out, max_out], dim=1)
        atten = self.sigmoid(self.CFR_branches[idx](atten))
        out = mid * atten
        heatmap = torch.mean(out, dim=1, keepdim=True)
        if self.need_cluster:
            # print(self.need_cluster)
            class_region = self.set_class_mask_matrix(heatmap)
            out = self.Regional_Normalization(class_region, out)
            # print('ok!')
        else:
            out = self.Regional_Normalization(heatmap, out)
        outs.append(out)
        out_ = sum(outs)
        out_ = self.relu(out_)

        return out_





