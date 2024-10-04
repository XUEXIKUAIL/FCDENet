import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import OrderedDict
from .mix_transformer import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5
# import pywt
import numpy as np
# from model.toolbox.models.A3project.guided_diffusion.script_util import create_gaussian_diffusion
from .ICB import SCDC
from .FCD import MAc
# from pytorch_wavelets import DTCWTForward, DTCWTInverse
from pytorch_wavelets import DWTForward, DWTInverse
# from torch.nn.functional import kl_div

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class MP(nn.Module):
    """
    Reference:
    """
    def __init__(self, in_channels, pool_size, norm_layer=nn.BatchNorm2d):
        super(MP, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
        self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))

        inter_channels = int(in_channels/4)
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_0 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                norm_layer(inter_channels))
        self.conv2_4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                norm_layer(inter_channels))
        self.conv2_5 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_6 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels*2, in_channels, 1, bias=False),
                                norm_layer(in_channels))
        # # bilinear interpolate options
        # self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x2_1 = self.conv2_0(x1)
        x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), (h, w))
        x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), (h, w))
        x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), (h, w))
        x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), (h, w))
        x1 = self.conv2_5(F.relu_(x2_1 + x2_2 + x2_3))
        x2 = self.conv2_6(F.relu_(x2_5 + x2_4))
        out = self.conv3(torch.cat([x1, x2], dim=1))
        return F.relu_(x + out)

class DWT_and_IDWT(nn.Module):
    def __init__(self, inchannel, outchannel, h, w):
        super().__init__()
        # self.DWT = DTCWTForward(J=3, biort='near_sym_b', qshift='qshift_b')
        # self.IWT = DTCWTInverse(biort='near_sym_b', qshift='qshift_b')
        self.DWT_haar = DWTForward(J=1, wave='haar', mode='zero')
        self.IWT_haar = DWTInverse(wave='haar', mode='zero')
        self.conv1 = BasicConv2d(inchannel, outchannel, kernel_size=1, stride=1, padding=0)
        self.conv1x1 = BasicConv2d(int(inchannel//4), inchannel, kernel_size=1, stride=1, padding=0)
        self.conv3 = BasicConv2d(outchannel, outchannel)
        self.local_att = nn.Sequential(
            nn.MaxPool2d(1, padding=0, stride=1),
            nn.Conv2d(inchannel, int(inchannel), kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(int(inchannel)),
            nn.ReLU(inplace=True)
        )
        self.MP = MP(inchannel, (h, w))
        self.sig = nn.Sigmoid()
    def forward(self, x, y):
        xy = self.MP(x + y)
        xy_sig = self.sig(xy)
        Xl, Xh = self.DWT_haar(x)
        Yl, Yh = self.DWT_haar(y)
        x_y = Xl + Yl
        x_m = self.IWT_haar((x_y, Xh))
        y_m = self.IWT_haar((x_y, Yh))
        xy_m = x_m + y_m
        return self.conv1(xy_m.to(torch.float32)) * xy_sig + xy

# def wavelet_transform_2d(image, wavelet='haar'):
#
#     coeffs = pywt.dwt2(image, wavelet)
#     re_image = pywt.idwt2(coeffs, wavelet)
#
#     return np.array(re_image)


class pp_upsample(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inc, outc, 3, padding=1),
            nn.BatchNorm2d(outc),
            nn.PReLU()
        )
    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)


class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=1, s=1, p=0, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)

class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
                convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
                convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu),
                )

    def forward(self, x):
        return self.conv(x)

class up2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.interpolate(input=x, size=(h, w), mode='bilinear')
        return p
class SalHead(nn.Module):
    def __init__(self, in_channel, n_classes):
        super(SalHead, self).__init__()
        self.conv = nn.Sequential(
            nn.Dropout2d(p=0.1),
            nn.Conv2d(in_channel, n_classes, 1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.conv(x)
class EnDecoderModel(nn.Module):
    def __init__(self, n_classes, backbone):
        super(EnDecoderModel, self).__init__()
        if backbone == 'segb4':
            self.backboner = mit_b4()
            self.backboned = mit_b4()
            # with torch.no_grad():
            #     self.backbone = mit_b4().eval()
        #############################################
        self.dscon512 = convbnrelu(512, 320)
        self.dscon320 = convbnrelu(320, 128)
        self.dscon128 = convbnrelu(128, 64)
        self.dscon64 = convbnrelu(64, n_classes)
        self.up2 = up2()

        self.up_conv512_320 = pp_upsample(512, 320)
        self.up_conv320_128 = pp_upsample(320, 128)
        self.up_conv128_64 = pp_upsample(128, 64)
        self.up_conv64_nclass = pp_upsample(64, n_classes)

        self.mac1 = MAc(64, 64)
        self.mac2 = MAc(128, 128)

        self.SCDC_R3 = SCDC(320, True)
        self.SCDC_D3 = SCDC(320)

        self.SCDC_R4 = SCDC(512, True)
        self.SCDC_D4 = SCDC(512)

        self.dwt320 = DWT_and_IDWT(320, 320, 30, 40)
        self.dwt128 = DWT_and_IDWT(128, 128, 60, 80)
        self.dwt64 = DWT_and_IDWT(64, 64, 120, 160)

        self.f4_p = SalHead(320, n_classes)
        self.f3_p = SalHead(128, n_classes)
        self.f2_p = SalHead(64, n_classes)
    def forward(self, rgb, dep):

        features_rgb = self.backboner(rgb)
        features_dep = self.backboned(dep)
        # with torch.no_grad():
        #     prompt_r = self.backbone(rgb)
        #     features_prompt_rlist = prompt_r[0]
        #     pro_rf1 = features_prompt_rlist[0]
        #     pro_rf2 = features_prompt_rlist[1]
        #     pro_rf3 = features_prompt_rlist[2]
        #     pro_rf4 = features_prompt_rlist[3]
        features_rlist = features_rgb[0]
        features_dlist = features_dep[0]

        # features_r_embeding = features_rgb[1]
        # features_d_embeding = features_dep[1]
        # remb1 = features_r_embeding[0]

        rf1 = features_rlist[0]
        rf2 = features_rlist[1]
        rf3 = features_rlist[2]
        rf4 = features_rlist[3]

        df1 = features_dlist[0]
        df2 = features_dlist[1]
        df3 = features_dlist[2]
        df4 = features_dlist[3]


        FD_pervise = []
        cluster_list = []
        #############################################
        # FD1 = rf1 + df1
        FD1 = self.mac1(rf1, df1)
        ############################################
        # FD2 = rf2 + df2
        FD2 = self.mac2(rf2, df2)
        ##############################################
        rf3 = self.SCDC_R3(rf3)
        df3 = self.SCDC_D3(df3)
        cluster_list.append(rf3)
        FD3 = rf3 + df3
        #############################################
        rf4 = self.SCDC_R4(rf4)
        df4 = self.SCDC_D4(df4)
        rf4_align = self.dscon512(rf4)
        cluster_list.append(rf4_align)
        FD4 = rf4 + df4
        ##############################################
        # 使用haar小波变换对图像进行小波变换
        # transformed_image = wavelet_transform_2d(image, wavelet='haar')
        # print(transformed_image)

        # FD4_up2 = self.up2(FD4)
        # FD4_up2_DScon3 = self.dscon512(FD4_up2)

        FD4_up2_DScon3 = self.up_conv512_320(FD4)

        # FD34 = FD3 + FD4_up2_DScon3
        FD34 = self.dwt320(FD3, FD4_up2_DScon3)

        FD34_p = self.f4_p(FD34)
        FD_pervise.append(FD34_p)

        # FD34_up2 = self.up2(FD34)
        # FD34_up2_DScon3 = self.dscon320(FD34_up2)

        FD34_up2_DScon3 = self.up_conv320_128(FD34)
        # FD234 = FD34_up2_DScon3 + FD2
        FD234 = self.dwt128(FD2, FD34_up2_DScon3)

        FD234_p = self.f3_p(FD234)
        FD_pervise.append(FD234_p)

        # FD234_up2 = self.up2(FD234)
        # FD234_up2_DScon3 = self.dscon128(FD234_up2)

        FD234_up2_DScon3 = self.up_conv128_64(FD234)
        # FD1234 = FD234_up2_DScon3 + FD1
        FD1234 = self.dwt64(FD1, FD234_up2_DScon3)

        FD1234_p = self.f2_p(FD1234)
        FD_pervise.append(FD1234_p)

        out = self.up2(FD1234)
        # out_upinit = self.up2(out)
        # out_upinit_DSnumclass = self.dscon64(out_upinit)
        out_upinit_DSnumclass = self.up_conv64_nclass(out)

        return out_upinit_DSnumclass, FD_pervise, cluster_list

    def load_pre(self, pre_model1):
        new_state_dict3 = OrderedDict()
        state_dict = torch.load(pre_model1)['state_dict']
        for k, v in state_dict.items():
            name = k[9:]
            new_state_dict3[name] = v
        self.backboner.load_state_dict(new_state_dict3, strict=False)
        self.backboned.load_state_dict(new_state_dict3, strict=False)
        print('B4.Pth loading')
        # print('self.backbone loading')

if __name__ == '__main__':
     import os
     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
     device = torch.device('cuda')
     rgb = torch.randn(1, 3, 480, 640).to(device)
     dep = torch.randn(1, 3, 480, 640).to(device)
     model = EnDecoderModel(n_classes=41, backbone='segb4').to(device)
     out = model(rgb, dep)
     print('out[0]输出结果：', out[0].shape)
     print('cluster_out', out[2][0].shape, out[2][1].shape)
     print('****************************************')
     print('参数量统计如下------------:')
     from model.toolbox.models.A1project2.FLOP import CalParams
     #
     CalParams(model, rgb, dep)
     print('Total params % .2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
        # 只有BACKBONEr+BACKBONEd####################
        # [Statistics Information]
        # FLOPs: 105.785G
        # Params: 121.685M
        #  ####################
        # Total params  121.69M

        # #######CAF(1 - fuse_r_diff, 1 - fuse_d_diff)+FuP_cluster_r+nocl_d+_dwt_and_idwt############
        # [Statistics Information]
        # FLOPs: 126.258G
        # Params: 127.685M
        #  ####################
        # Total params  127.68M

# model3_SCDC_2####################
# ####################
# [Statistics Information]
# FLOPs: 135.419G
# Params: 128.203M
#  ####################
# Total params  128.75M
#
# Process finished with exit code 0