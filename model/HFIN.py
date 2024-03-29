
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .modules import InvertibleConv1x1
from .refine import Refine
import torch.nn.init as init




def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():

            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, d, relu_slope=0.1):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, dilation=d, padding=d, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, dilation=d, padding=d, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

    def forward(self, x):
        out = self.relu_1(self.conv_1(x))
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, d = 1, init='xavier', gc=8, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = UNetConvBlock(channel_in, gc, d)
        self.conv2 = UNetConvBlock(gc, gc, d)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3], 0.1)
        # initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))

        return x3





class InvBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, d = 1, clamp=0.8):
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2
        # print(channel_num)

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1, d)
        self.G = subnet_constructor(self.split_len1, self.split_len2, d)
        self.H = subnet_constructor(self.split_len1, self.split_len2, d)

        in_channels = channel_num
        self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=True)
        self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    def forward(self, x, rev=False):
        # if not rev:
        # invert1x1conv
        x, logdet = self.flow_permutation(x, logdet=0, rev=False)

        # split to 1 channel and 2 channel.
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
        # print('x1',x1.shape)  channel = 4
        # print('x2',x2.shape)

        y1 = x1 + self.F(x2)  # 1 channel
        self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
        out = torch.cat((y1, y2), 1)

        return out


class Freprocess(nn.Module):
    def __init__(self, channels):
        super(Freprocess, self).__init__()
        self.pre1 = nn.Conv2d(channels,channels,1,1,0)
        self.pre2 = nn.Conv2d(channels,channels,1,1,0)
        self.amp_fuse = nn.Sequential(nn.Conv2d(2*channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(2*channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.post = nn.Conv2d(channels,channels,1,1,0)

    def forward(self, msf, panf):

        _, _, H, W = msf.shape
        msF = torch.fft.rfft2(self.pre1(msf)+1e-8, norm='backward')
        panF = torch.fft.rfft2(self.pre2(panf)+1e-8, norm='backward')
        msF_amp = torch.abs(msF)
        msF_pha = torch.angle(msF)
        panF_amp = torch.abs(panF)
        panF_pha = torch.angle(panF)
        amp_fuse = self.amp_fuse(torch.cat([msF_amp,panF_amp],1))
        pha_fuse = self.pha_fuse(torch.cat([msF_pha,panF_pha],1))

        real = amp_fuse * torch.cos(pha_fuse)+1e-8
        imag = amp_fuse * torch.sin(pha_fuse)+1e-8
        out = torch.complex(real, imag)+1e-8
        out = torch.abs(torch.fft.irfft2(out, s=(H, W), norm='backward'))

        return self.post(out)


class STFreprocess(nn.Module):
    def __init__(self, channels):
        super(STFreprocess, self).__init__()
        self.pre1 = nn.Conv2d(channels,channels,1,1,0)
        self.pre2 = nn.Conv2d(channels,channels,1,1,0)
        self.amp_fuse1 = nn.Sequential(nn.Conv2d(2*channels,8* channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(8 * channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.pha_fuse1 = nn.Sequential(nn.Conv2d(2*channels,8* channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(8 * channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.amp_fuse2 = nn.Sequential(nn.Conv2d(2*channels,8* channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(8 * channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.pha_fuse2 = nn.Sequential(nn.Conv2d(2*channels,8* channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(8 * channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.amp_fuse3 = nn.Sequential(nn.Conv2d(2*channels,8* channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(8 * channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.pha_fuse3 = nn.Sequential(nn.Conv2d(2*channels,8* channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(8 * channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.amp_fuse4 = nn.Sequential(nn.Conv2d(2*channels,8* channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(8 * channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.pha_fuse4 = nn.Sequential(nn.Conv2d(2*channels,8* channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(8 * channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))

        # self.fshif = nn.Conv2d(2 * channels, channels, 3, 1, 1)

        self.post = nn.Conv2d(channels,channels,1,1,0)

    def forward(self, msf, panf):

        _, _, H, W = msf.shape
        s1 = 43
        s2 = 42
        # split
        msfa = msf[:, :, :s1+s2, :s1+s2]
        msfb = msf[:, :, :s1+s2, s1:]
        msfc = msf[:, :, s1:, :s1+s2]
        msfd = msf[:, :, s1:, s1:]
        msf_list = [msfa, msfb, msfc, msfd]


        panfa = panf[:, :, :s1+s2, :s1+s2]
        panfb = panf[:, :, :s1+s2, s1:]
        panfc = panf[:, :, s1:, :s1+s2]
        panfd = panf[:, :, s1:, s1:]
        panf_list = [panfa, panfb, panfc, panfd]

        qH = H // 4
        qW = W // 4


        amp_list = [self.amp_fuse1, self.amp_fuse2, self.amp_fuse3, self.amp_fuse4]
        pha_list = [self.pha_fuse1, self.pha_fuse2, self.pha_fuse3, self.pha_fuse4]

        out_list = [0, 0, 0, 0]
        out_shift = [0, 0, 0, 0]

        for i in range(len(msf_list)):
            msF = torch.fft.rfft2(self.pre1(msf_list[i])+1e-8, norm='backward')
            panF = torch.fft.rfft2(self.pre2(panf_list[i])+1e-8, norm='backward')
            msF_amp = torch.abs(msF)
            msF_pha = torch.angle(msF)
            panF_amp = torch.abs(panF)
            panF_pha = torch.angle(panF)

            # 不共享权重

            amp_fuse = amp_list[i](torch.cat([msF_amp,panF_amp],1))
            pha_fuse = pha_list[i](torch.cat([msF_pha,panF_pha],1))

            real = amp_fuse * torch.cos(pha_fuse)+1e-8
            imag = amp_fuse * torch.sin(pha_fuse)+1e-8
            out_list[i] = torch.complex(real, imag)+1e-8
            out_list[i] = torch.abs(torch.fft.irfft2(out_list[i], s=(s1+s2, s1+s2), norm='backward'))



        outup =  torch.cat((out_list[0][:,:,:s1,:s1],
                            (out_list[0][:,:,:s1,s1:]+out_list[1][:,:,:s1,:s2])/2,
                            out_list[1][:, :, :s1, s2:]
                            ), dim=-1)
        outmid = torch.cat(((out_list[0][:,:,s1:,:s1]+out_list[2][:,:,:s2,:s1])/2,
                             (out_list[0][:,:,s1:,s1:]+out_list[1][:, :, s1:, :s2]+out_list[2][:, :, :s2, s1:]+out_list[3][:, :, :s2, :s2])/4,
                            (out_list[1][:, :, s1:, s2:]+out_list[3][:, :, :s2, s2:])/2
                             ), dim=-1)

        outdown = torch.cat((out_list[2][:, :, s2:, :s1],
                           (out_list[2][:, :, s2:, s1:] + out_list[3][:, :, s2:, :s2]) / 2,
                           out_list[3][:, :, s2:, s2:]
                           ), dim=-1)

        out = torch.cat((outup, outmid, outdown), dim=-2)

        return self.post(out)




class SpaFre(nn.Module):
    def __init__(self, channels):
        super(SpaFre, self).__init__()
        self.panprocess = nn.Conv2d(channels,channels,3,1,1)
        self.panpre = nn.Conv2d(channels,channels,1,1,0)
        self.spa_process = nn.Sequential(InvBlock(DenseBlock, 2*channels, channels),
                                         nn.Conv2d(2*channels,channels,1,1,0))
        self.fre_process = Freprocess(channels)
        self.STfre_process = STFreprocess(channels)
        self.spa_att = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=True),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1, bias=True),
                                     nn.Sigmoid())
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.contrast = stdv_channels
        self.cha_att1 = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=1, padding=0, bias=True),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(channels // 2, channels, kernel_size=1, padding=0, bias=True),
                                     nn.Sigmoid())
        self.cha_att2 = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=1, padding=0, bias=True),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(channels // 2, channels, kernel_size=1, padding=0, bias=True),
                                     nn.Sigmoid())
        self.spa_ST_fusion = nn.Sequential(nn.Conv2d(channels * 2, channels // 2, kernel_size=3, padding=1, bias=True),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1, bias=True),
                                     nn.Sigmoid())
        self.spa_FT_fusion = nn.Sequential(nn.Conv2d(channels * 2, channels // 2, kernel_size=3, padding=1, bias=True),
                                           nn.LeakyReLU(0.1),
                                           nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1, bias=True),
                                           nn.Sigmoid())
        self.ST_to_FT = nn.Conv2d(channels, channels, 3, 1, 1)
        self.FT_to_ST = nn.Conv2d(channels, channels, 3, 1, 1)
        self.post1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.post2 = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, msf, pan):  #, i
        panpre = self.panprocess(pan)
        panf = self.panpre(panpre)

        # MS PAN fusion
        spafuse = self.spa_process(torch.cat([msf,panf],1))
        frefuse = self.fre_process(msf,panf)
        STfuse = self.STfre_process(msf,panf)

        # muti-fusion
        spa_ST = self.spa_ST_fusion(torch.cat([spafuse, STfuse],dim=1))
        ST_fused = STfuse * spa_ST + STfuse

        spa_FT = self.spa_FT_fusion(torch.cat([spafuse, frefuse], dim=1))
        FT_fused = frefuse * spa_FT + frefuse

        # ST FT exchange information
        ms_ST = ST_fused + self.FT_to_ST(FT_fused)
        ms_FT = FT_fused + self.ST_to_FT(ST_fused)
        # ms_ST = torch.cat([ST_fused[:, :2, :, :], FT_fused[:, 2:, :, :]], dim=1)
        # ms_FT = torch.cat([FT_fused[:, :2, :, :], ST_fused[:, 2:, :, :]], dim=1)


        # cat_f = torch.cat([spa_res,frefuse,STfuse],1)  # 为啥是frefuse？不应该是spafuse吗？

        # channel attention
        fin_ST = self.post1(self.cha_att1(self.contrast(ms_ST) + self.avgpool(ms_ST)) * ms_ST)
        fin_FT = self.post2(self.cha_att2(self.contrast(ms_FT) + self.avgpool(ms_FT)) * ms_FT)

        out_ST = fin_ST + msf
        out_FT = fin_FT + msf

        return out_ST, out_FT, panpre


class SpaFremid(nn.Module):
    def __init__(self, channels):
        super(SpaFremid, self).__init__()
        self.panprocess = nn.Conv2d(channels, channels, 3, 1, 1)
        self.panpre = nn.Conv2d(channels, channels, 1, 1, 0)
        self.spa_process = nn.Sequential(InvBlock(DenseBlock, 3 * channels, channels),
                                         nn.Conv2d(3 * channels, channels, 1, 1, 0))
        self.fre_process = Freprocess(channels)
        self.STfre_process = STFreprocess(channels)
        self.spa_att = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=True),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1, bias=True),
                                     nn.Sigmoid())
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.contrast = stdv_channels
        self.cha_att1 = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=1, padding=0, bias=True),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(channels // 2, channels, kernel_size=1, padding=0, bias=True),
                                     nn.Sigmoid())
        self.cha_att2 = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=1, padding=0, bias=True),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(channels // 2, channels, kernel_size=1, padding=0, bias=True),
                                     nn.Sigmoid())
        self.spa_ST_fusion = nn.Sequential(nn.Conv2d(channels * 2, channels // 2, kernel_size=3, padding=1, bias=True),
                                           nn.LeakyReLU(0.1),
                                           nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1, bias=True),
                                           nn.Sigmoid())
        self.spa_FT_fusion = nn.Sequential(nn.Conv2d(channels * 2, channels // 2, kernel_size=3, padding=1, bias=True),
                                           nn.LeakyReLU(0.1),
                                           nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1, bias=True),
                                           nn.Sigmoid())
        self.ST_to_FT = nn.Conv2d(channels, channels, 3, 1, 1)
        self.FT_to_ST = nn.Conv2d(channels, channels, 3, 1, 1)
        self.post1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.post2 = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, msst, msft, pan):  # , i  实际输入是 msst msft
        panpre = self.panprocess(pan)
        panf = self.panpre(panpre)
        msf = torch.cat([msst,msft],dim=1)

        # MS PAN fusion
        spafuse = self.spa_process(torch.cat([msf, panf], 1))
        frefuse = self.fre_process(msst, panf)
        STfuse = self.STfre_process(msft, panf)

        # muti-fusion
        spa_ST = self.spa_ST_fusion(torch.cat([spafuse, STfuse], dim=1))
        ST_fused = STfuse * spa_ST + STfuse

        spa_FT = self.spa_FT_fusion(torch.cat([spafuse, frefuse], dim=1))
        FT_fused = frefuse * spa_FT + frefuse

        # ST FT exchange information
        ms_ST = ST_fused + self.FT_to_ST(FT_fused)
        ms_FT = FT_fused + self.ST_to_FT(ST_fused)
        # ms_ST = torch.cat([ST_fused[:, :2, :, :], FT_fused[:, 2:, :, :]], dim=1)
        # ms_FT = torch.cat([FT_fused[:, :2, :, :], ST_fused[:, 2:, :, :]], dim=1)

        fin_ST = self.post1(self.cha_att1(self.contrast(ms_ST) + self.avgpool(ms_ST)) * ms_ST)
        fin_FT = self.post2(self.cha_att2(self.contrast(ms_FT) + self.avgpool(ms_FT)) * ms_FT)

        out_ST = fin_ST + msft
        out_FT = fin_FT + msst

        return out_ST, out_FT, panpre



class FeatureProcess(nn.Module):
    def __init__(self, channels):
        super(FeatureProcess, self).__init__()

        self.conv_p = nn.Conv2d(4, channels, 3, 1, 1)
        self.conv_p1 = nn.Conv2d(1, channels, 3, 1, 1)
        self.block = SpaFre(channels)
        self.block1 = SpaFremid(channels)
        self.block2 = SpaFremid(channels)
        self.block3 = SpaFremid(channels)
        self.block4 = SpaFremid(channels)
        self.fuse = nn.Conv2d(5*channels,channels,1,1,0)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.contrast = stdv_channels
        self.cha_att = nn.Sequential(nn.Conv2d(channels * 2, channels // 2, kernel_size=1, padding=0, bias=True),
                                      nn.LeakyReLU(0.1),
                                      nn.Conv2d(channels // 2, channels * 2, kernel_size=1, padding=0, bias=True),
                                      nn.Sigmoid())
        self.post = nn.Conv2d(channels * 2, channels, 3, 1, 1)


    def forward(self, ms, pan): #, i
        msf = self.conv_p(ms)
        panf = self.conv_p1(pan)
        msst0, msft0, panf0 = self.block(msf, panf) #,i
        msst1, msft1, panf1 = self.block1(msst0, msft0, panf0)
        msst2, msft2, panf2 = self.block2(msst1, msft1, panf1)
        msst3, msft3, panf3 = self.block3(msst2, msft2, panf2)
        msst4, msft4, panf4 = self.block4(msst3, msft3, panf3)


        fin_all_fusion = torch.cat([msst4, msft4],dim=1)

        fin_FT = self.post(self.cha_att(self.contrast(fin_all_fusion) + self.avgpool(fin_all_fusion)) * fin_all_fusion)

        return fin_FT


def upsample(x, h, w):
    return F.interpolate(x, size=[h, w], mode='bicubic', align_corners=True)


class Net(nn.Module):
    def __init__(self, channels):
        super(Net, self).__init__()
        self.process = FeatureProcess(channels)
        # self.cdc = nn.Sequential(nn.Conv2d(1, 4, 1, 1, 0), cdcconv(4, 4), cdcconv(4, 4))
        self.refine = Refine(channels, 4)

    def forward(self, ms, pan): # i
        # ms  - low-resolution multi-spectral image [N,C,h,w]
        # pan - high-resolution panchromatic image [N,1,H,W]
        if type(pan) == torch.Tensor:
            pass
        elif pan == None:
            raise Exception('User does not provide pan image!')
        _, _, m, n = ms.shape
        _, _, M, N = pan.shape

        mHR = upsample(ms, M, N)
        HRf = self.process(mHR, pan) # i
        HR = self.refine(HRf)+ mHR

        return HR


class Net1(nn.Module):
    def __init__(self, num_channels, base_filter, args):
        super(Net1, self).__init__()
        self.process = FeatureProcess(num_channels)
        # self.cdc = nn.Sequential(nn.Conv2d(1, 4, 1, 1, 0), cdcconv(4, 4), cdcconv(4, 4))
        self.refine = Refine(num_channels, 4)

    def forward(self,l_ms,bms,pan):
        if type(pan) == torch.Tensor:
            pass
        elif pan == None:
            raise Exception('User does not provide pan image!')
        _, _, m, n = l_ms.shape
        _, _, M, N = pan.shape

        # mHR = upsample(l_ms, M, N)
        mHR=bms
        HRf = self.process(mHR, pan)
        HR = self.refine(HRf)+ mHR

        return HR




def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)



import os
import cv2

def feature_save(tensor,name,i):
    # tensor = torchvision.utils.make_grid(tensor.transpose(1,0))
    tensor = torch.mean(tensor,dim=1)
    inp = tensor.detach().cpu().numpy().transpose(1,2,0)
    inp = inp.squeeze(2)
    inp = (inp - np.min(inp)) / (np.max(inp) - np.min(inp))
    if not os.path.exists(name):
        os.makedirs(name)
    # for i in range(tensor.shape[1]):
    #     inp = tensor[:,i,:,:].detach().cpu().numpy().transpose(1,2,0)
    #     inp = np.clip(inp,0,1)
    # # inp = (inp-np.min(inp))/(np.max(inp)-np.min(inp))
    #
    #     cv2.imwrite(str(name)+'/'+str(i)+'.png',inp*255.0)
    inp = cv2.applyColorMap(np.uint8(inp * 255.0),cv2.COLORMAP_JET)
    cv2.imwrite(name + '/' + str(i) + '.png', inp)
