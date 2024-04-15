import math
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm

from mmcv.ops import ModulatedDeformConv2dPack, modulated_deform_conv2d


class ResBlocks(nn.Module):
    def __init__(self, input_channels, out_channels, num_resblocks, num_channels,pytorch_init=False):
        super(ResBlocks, self).__init__()
        self.input_channels = input_channels
        self.num_channels=num_channels
        self.first_conv = nn.Conv2d(in_channels=self.input_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1)

        modules = []
        for _ in range(num_resblocks):
            modules.append(ResidualBlockNoBN(num_feat=num_channels))
        self.resblocks = nn.Sequential(*modules)
        self.conv_out=nn.Conv2d(in_channels=num_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        
        if not pytorch_init:
            default_init_weights([self.first_conv, self.conv_out], 0.0)

    def forward(self, h):
        shallow_feature = self.first_conv(h)
        inter_h=self.resblocks(shallow_feature)
        new_h = self.conv_out(inter_h)
        return new_h


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.0)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale



@torch.no_grad()
def default_init_weights(module_list, scale=0, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)