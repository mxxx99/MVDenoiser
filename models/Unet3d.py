import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io

from models.modules import ConvBlock3d
from models.pytorch_pwc.utils import warp2center_5frame
import argparse, json, glob, os, sys
from models.register import register

from pathlib import Path
_script_dir = Path( __file__ ).parent
_root_dir = _script_dir.parent

@register
class Unet3d(nn.Module):
    def __init__(self, n_channel_in=1, n_channel_out=1, residual=False, down='conv', up='tconv', activation='selu'):
        super(Unet3d, self).__init__()

        self.residual = residual
        self.n_channel_out=n_channel_out

        if down == 'maxpool':
            self.down1 = nn.MaxPool3d(kernel_size=2)
            self.down2 = nn.MaxPool3d(kernel_size=2)
            self.down3 = nn.MaxPool3d(kernel_size=2)
            self.down4 = nn.MaxPool3d(kernel_size=2)
        elif down == 'avgpool':
            self.down1 = nn.AvgPool3d(kernel_size=2)
            self.down2 = nn.AvgPool3d(kernel_size=2)
            self.down3 = nn.AvgPool3d(kernel_size=2)
            self.down4 = nn.AvgPool3d(kernel_size=2)
        elif down == 'conv':
            self.down1 = nn.Conv3d(32, 32, kernel_size=2, stride=2, groups=32)
            self.down2 = nn.Conv3d(64, 64, kernel_size=2, stride=2, groups=64)
            self.down3 = nn.Conv3d(128, 128, kernel_size=2, stride=2, groups=128)
            self.down4 = nn.Conv3d(256, 256, kernel_size=2, stride=2, groups=256)

            self.down1.weight.data = 0.01 * self.down1.weight.data + 0.25
            self.down2.weight.data = 0.01 * self.down2.weight.data + 0.25
            self.down3.weight.data = 0.01 * self.down3.weight.data + 0.25
            self.down4.weight.data = 0.01 * self.down4.weight.data + 0.25

            self.down1.bias.data = 0.01 * self.down1.bias.data + 0
            self.down2.bias.data = 0.01 * self.down2.bias.data + 0
            self.down3.bias.data = 0.01 * self.down3.bias.data + 0
            self.down4.bias.data = 0.01 * self.down4.bias.data + 0

        if up == 'bilinear' or up == 'nearest':
            self.up1 = lambda x: nn.functional.interpolate(x, mode=up, scale_factor=2)
            self.up2 = lambda x: nn.functional.interpolate(x, mode=up, scale_factor=2)
            self.up3 = lambda x: nn.functional.interpolate(x, mode=up, scale_factor=2)
            self.up4 = lambda x: nn.functional.interpolate(x, mode=up, scale_factor=2)
        elif up == 'tconv':
            self.up1 = nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2, groups=256)
            self.up2 = nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2, groups=128)
            self.up3 = nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2, groups=64)
            self.up4 = nn.ConvTranspose3d(32, 32, kernel_size=2, stride=2, groups=32)

            self.up1.weight.data = 0.01 * self.up1.weight.data + 0.25
            self.up2.weight.data = 0.01 * self.up2.weight.data + 0.25
            self.up3.weight.data = 0.01 * self.up3.weight.data + 0.25
            self.up4.weight.data = 0.01 * self.up4.weight.data + 0.25

            self.up1.bias.data = 0.01 * self.up1.bias.data + 0
            self.up2.bias.data = 0.01 * self.up2.bias.data + 0
            self.up3.bias.data = 0.01 * self.up3.bias.data + 0
            self.up4.bias.data = 0.01 * self.up4.bias.data + 0

        self.conv1 = ConvBlock3d(n_channel_in, 32, residual, activation)
        self.conv2 = ConvBlock3d(32, 64, residual, activation)
        self.conv3 = ConvBlock3d(64, 128, residual, activation)
        self.conv4 = ConvBlock3d(128, 128, residual, activation)

        self.conv6 = ConvBlock3d(2 * 256, 128, residual, activation)
        self.conv7 = ConvBlock3d(2 * 128, 64, residual, activation)
        self.conv8 = ConvBlock3d(2 * 64, 32, residual, activation)
        self.conv9 = ConvBlock3d(2 * 32, n_channel_out, residual, activation)

        if self.residual:
            self.convres = ConvBlock3d(n_channel_out, n_channel_out, residual, activation)

    def forward(self, x, flow=None):
        # if self.noise_cat:
        #     noise_map=x[1]#(b,1,1,1,1)
        #     x=x[0]

        if not flow==None:
            x_ori=x.clone().detach()
            x=warp2center_5frame(x,flow)

        c0 = x
        c1 = self.conv1(x)#4->32
        x = self.down1(c1)
        c2 = self.conv2(x)#32->64
        x = self.down2(c2)
        c3 = self.conv3(x)#64->128
        x = self.down3(c3)
        x = self.conv4(x)#128->128

        x = self.up2(x)
        x = torch.cat([x, c3], 1)  # x[:,0:64]*x[:,64:128],
        x = self.conv7(x)
        x = self.up3(x)
        x = torch.cat([x, c2], 1)  # x[:,0:32]*x[:,32:64],
        x = self.conv8(x)
        x = self.up4(x)
        x = torch.cat([x, c1], 1)  # x[:,0:16]*x[:,16:32],
        inter_feat=x.clone().detach()
        x = self.conv9(x)
        if self.residual:
            x = torch.add(x, self.convres(c0[:,:self.n_channel_out]))

        return x,inter_feat


