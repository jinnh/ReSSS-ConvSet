import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as f
from option import opt


class BasicConv3d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding=(0, 0, 0)):
        super(BasicConv3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channel, out_channel,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channel, out_channel,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


class ReConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ReConv, self).__init__()
        self.reconv = nn.ModuleList([
            nn.Conv3d(in_channel, out_channel, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=False),
            nn.Conv3d(in_channel, out_channel, kernel_size=(1, 3, 1), stride=1, padding=(0, 1, 0), bias=False),
            nn.Conv3d(in_channel, out_channel, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1), bias=False)
        ])
        self.aggregation = nn.Conv3d(3 * out_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        hsi_feas = []
        for layer in self.reconv:
            fea = layer(x)
            hsi_feas.append(fea)
        hsi_feas = torch.cat(hsi_feas, dim=1)
        x = self.aggregation(hsi_feas)

        return x


class ReConvblock(nn.Module):
    def __init__(self, cin, cout):
        super(ReConvblock, self).__init__()

        self.Conv_mixdence = nn.Conv3d(cin, cout, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.reconv = nn.ModuleList([
            BasicConv3d(cout, cout, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
            BasicConv3d(cout, cout, kernel_size=(1, 3, 1), stride=1, padding=(0, 1, 0)),
            BasicConv3d(cout, cout, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1))
        ])
        self.aggregation1 = nn.Conv3d(3 * cout, cout, kernel_size=1, stride=1, padding=0, bias=False)
        self.aggregation2 = BasicConv3d(cout, cout, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))

    def forward(self, x):
        x = torch.cat(x, 1)
        x = self.relu(self.Conv_mixdence(x))

        hsi_feas = []
        for layer in self.reconv:
            fea = layer(x)
            hsi_feas.append(fea)
        hsi_feas = torch.cat(hsi_feas, dim=1)
        output = self.aggregation1(hsi_feas) + x
        output = self.aggregation2(output) + output

        return output


class ReConvNet(nn.Module):
    def __init__(self):
        super(ReConvNet, self).__init__()

        n_feats = 64
        self.head = ReConv(1, n_feats)

        self.ReConvLayers = nn.ModuleList([
            ReConvblock(cin=1 * n_feats, cout=n_feats * 1),
            ReConvblock(cin=1 * n_feats, cout=n_feats * 1),
            ReConvblock(cin=2 * n_feats, cout=n_feats * 1),
            ReConvblock(cin=3 * n_feats, cout=n_feats * 1),
            ReConvblock(cin=4 * n_feats, cout=n_feats * 1),
            ReConvblock(cin=5 * n_feats, cout=n_feats * 1),
            ReConvblock(cin=6 * n_feats, cout=n_feats * 1),
            ReConvblock(cin=7 * n_feats, cout=n_feats * 1),
            ReConvblock(cin=8 * n_feats, cout=n_feats * 1),
            ReConvblock(cin=9 * n_feats, cout=n_feats * 1),
            ReConvblock(cin=10 * n_feats, cout=n_feats * 1),
            # ReConvblock(cin=11 * n_feats, cout=n_feats * 1),
            # ReConvblock(cin=12 * n_feats, cout=n_feats * 1)
        ])

        scale = opt.upscale_factor
        self.tail1 = nn.ConvTranspose3d(n_feats, 16, kernel_size=(3, 2 + scale, 2 + scale), stride=(1, scale, scale),
                                        padding=(1, 1, 1), bias=False)
        self.tail2 = ReConv(16, 1)

        # self.downscale = nn.Conv2d(1, 1, kernel_size=(5, 5), stride=(4, 4), padding=(2, 2))

    def forward(self, LHSI):

        LHSI = LHSI.unsqueeze(1)
        T = self.head(LHSI)

        x = [T]
        x1 = []
        for layer in self.ReConvLayers:
            x_ = layer(x)
            x1.append(x_)
            x = x1
        x = x[-1] + T

        x = self.tail1(x)
        x = self.tail2(x)

        x = x.squeeze(1)

        return x


class reconnetHRHSI(nn.Module):
    def __init__(self):
        super(reconnetHRHSI, self).__init__()

        self.upsamper = nn.Upsample(scale_factor=opt.upscale_factor, mode='bicubic', align_corners=False)

        self.reconvNet = ReConvNet()

    def forward(self, LHSI):

        LHSI = [LHSI]
        recon_out = self.upsamper(LHSI[-1])

        recon = self.reconvNet(LHSI[-1])
        recon_out = recon_out + recon

        return recon_out
