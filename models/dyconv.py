#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" TAdaConv. """

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _triple
from torchvision.models.resnet import resnet18

# an example using dynamic conditional layer
class condition_idfcn_basis_comb_resnet(nn.Module):
  def __init__(self, resnet, fcn):
    super(condition_idfcn_basis_comb_resnet, self).__init__()
    self.resnet = resnet18()
    self.id_fc = nn.Linear(459558, fcn)
    self.id_tanh = nn.Tanh()
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x1, x2):
    x1 = self.resnet.conv1(x1)
    x1 = self.resnet.bn1(x1)
    x1 = self.resnet.relu(x1)
    x1 = self.resnet.maxpool(x1)

    x1 = self.resnet.layer1(x1)
    x1 = self.resnet.layer2(x1)
    x1 = self.resnet.layer3(x1)
    x1 = self.resnet.layer4(x1)

    x2 = self.id_fc(x2)
    x2 = self.id_tanh(x2)

    return x2

class RouteFuncMLP(nn.Module):
    """
    The routing function for generating the calibration weights.
    """

    def __init__(self, c_in, ratio, out_channels, kernels, bn_eps=1e-5, bn_mmt=0.1):
        """
        Args:
            c_in (int): number of input channels.
            ratio (int): reduction ratio for the routing function.
            kernels (list): temporal kernel size of the stacked 1D convolutions
        """
        super(RouteFuncMLP, self).__init__()
        self.c_in = c_in
        self.out_channels=out_channels
        self.avgpool = nn.AdaptiveAvgPool3d((None,1,1))
        self.globalpool = nn.AdaptiveAvgPool3d(1)
        self.g = nn.Conv3d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=1,
            padding=0,
        )
        self.a = nn.Conv3d(
            in_channels=c_in,
            out_channels=int(c_in//ratio),
            kernel_size=[kernels[0],1,1],
            padding=[kernels[0]//2,0,0],
        )
        # self.bn = nn.BatchNorm3d(int(c_in//ratio), eps=bn_eps, momentum=bn_mmt)
        self.relu = nn.ReLU(inplace=True)#这个relu会不会导致梯度消失，所以iso最大的时候会有问题
        self.b = nn.Conv3d(
            in_channels=int(c_in//ratio),
            out_channels=out_channels,
            kernel_size=[kernels[1],1,1],
            padding=[kernels[1]//2,0,0],
            bias=False
        )
        self.b.skip_init=True
        self.b.weight.data.zero_() # to make sure the initial values 
                                   # for the output is 1.
        
    def forward(self, x):
        g = self.globalpool(x)#(b,c,t,h,w)->(b,c,1,1,1)?
        x = self.avgpool(x)#->(b,c,t,1,1)
        x = self.a(x + self.g(g))#->(b,c//r,t,1,1)
        # x = self.bn(x)
        # x = self.relu(x)
        x = self.b(x) + 1#out:(b,1,t,1,1)
        x = x.permute(0,2,1,3,4).contiguous().view(-1,self.out_channels,1,1)#(b,t,1,1,1)
        # 这里输出的应该是一个attention图吧
        # 嗯输出的是calibration weight
        # shot torch.Size([1, 4, 16, 1, 1])
        return x
    
class RouteFuncMLP2d(nn.Module):
    """
    The routing function for generating the calibration weights.
    """

    def __init__(self, c_in, ratio, out_channels, kernels, bn_eps=1e-5, bn_mmt=0.1):
        """
        Args:
            c_in (int): number of input channels.
            ratio (int): reduction ratio for the routing function.
            kernels (list): temporal kernel size of the stacked 1D convolutions
        """
        super(RouteFuncMLP2d, self).__init__()
        self.c_in = c_in
        self.out_channels=out_channels
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.a = nn.Conv2d(
            in_channels=c_in,
            out_channels=int(c_in//ratio),
            kernel_size=kernels[0],
            padding=kernels[0]//2,
            bias=False
        )
        # self.bn = nn.BatchNorm3d(int(c_in//ratio), eps=bn_eps, momentum=bn_mmt)
        self.relu = nn.ReLU(inplace=True)#这个relu会不会导致梯度消失，所以iso最大的时候会有问题
        self.b = nn.Conv2d(
            in_channels=int(c_in//ratio),
            out_channels=out_channels,
            kernel_size=kernels[1],
            padding=kernels[1]//2,
            bias=True
        )
        self.b.skip_init=True
        self.b.weight.data.zero_() # to make sure the initial values 
                                   # for the output is 1.
        
    def forward(self, x):
        # 由于是2d版本，参考其他的dyconv，去掉globalpool只要avgpool
        # 2层conv+relu（基本都是这么写的）
        x = self.avgpool(x.float())#->(b,c,1,1)
        x = self.a(x)#->(b,c//r,1,1)
        # x = self.bn(x)
        # x = self.relu(x)
        x = self.b(x) + 1#out:(b,1,1,1)
        # 这里输出的应该是一个attention图吧
        # 嗯输出的是calibration weight
        return x

class TAdaConv2d(nn.Module):
    """
    Performs temporally adaptive 2D convolution.
    Currently, only application on 5D tensors is supported, which makes TAdaConv2d 
        essentially a 3D convolution with temporal kernel size of 1.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 cal_dim="cin"):
        super(TAdaConv2d, self).__init__()
        """
        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            kernel_size (list): kernel size of TAdaConv2d. 
            stride (list): stride for the convolution in TAdaConv2d.
            padding (list): padding for the convolution in TAdaConv2d.
            dilation (list): dilation of the convolution in TAdaConv2d.
            groups (int): number of groups for TAdaConv2d. 
            bias (bool): whether to use bias in TAdaConv2d.
        """

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        assert kernel_size[0] == 1
        assert stride[0] == 1
        assert padding[0] == 0
        assert dilation[0] == 1
        assert cal_dim in ["cin", "cout"]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.cal_dim = cal_dim

        # base weights (W_b)
        self.weight = nn.Parameter(
            torch.Tensor(1, 1, out_channels, in_channels // groups, kernel_size[1], kernel_size[2])
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, alpha):
        """
        Args:
            x (tensor): feature to perform convolution on.
            alpha (tensor): calibration weight for the base weights.
                W_t = alpha_t * W_b
        """
        _, _, c_out, c_in, kh, kw = self.weight.size()
        b, c_in, t, h, w = x.size()
        x = x.permute(0,2,1,3,4).reshape(1,-1,h,w)

        if self.cal_dim == "cin":
            # w_alpha: B, C, T, H(1), W(1) -> B, T, C, H(1), W(1) -> B, T, 1, C, H(1), W(1)
            # corresponding to calibrating the input channel
            weight = (alpha.permute(0,2,1,3,4).unsqueeze(2) * self.weight).reshape(-1, c_in//self.groups, kh, kw)
        elif self.cal_dim == "cout":
            # w_alpha: B, C, T, H(1), W(1) -> B, T, C, H(1), W(1) -> B, T, C, 1, H(1), W(1)
            # corresponding to calibrating the input channel
            weight = (alpha.permute(0,2,1,3,4).unsqueeze(3) * self.weight).reshape(-1, c_in//self.groups, kh, kw)

        bias = None
        if self.bias is not None:
            # in the official implementation of TAda2D, 
            # there is no bias term in the convs
            # hence the performance with bias is not validated
            bias = self.bias.repeat(b, t, 1).reshape(-1)
        output = F.conv2d(
            x, weight=weight, bias=bias, stride=self.stride[1:], padding=self.padding[1:],
            dilation=self.dilation[1:], groups=self.groups * b * t)

        output = output.view(b, t, c_out, output.size(-2), output.size(-1)).permute(0,2,1,3,4)

        return output
        
    def __repr__(self):
        return f"TAdaConv2d({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, " +\
            f"stride={self.stride}, padding={self.padding}, bias={self.bias is not None}, cal_dim=\"{self.cal_dim}\")"