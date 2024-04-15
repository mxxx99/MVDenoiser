import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import warnings
import math
import torch
import torchvision
import torch.utils.checkpoint as checkpoint
from distutils.version import LooseVersion
from torch.nn.modules.utils import _pair, _single
import numpy as np
from functools import reduce, lru_cache
from operator import mul
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class Basic_Stage(nn.Module):
    def __init__(self,num_layers=2,depth=3,in_chans=4,embed_dim=96,img_sz=224,patch_sz=1,
                 num_heads=6,window_size=7,norm_layer=nn.LayerNorm):
        super(Basic_Stage, self).__init__()
        self.layers=[]
        num_out_ch=in_chans
        self.window_size=window_size

        for i in range(num_layers):
            # 一个layer是一个RSTB；一个RSTB包括多个个SwinTransformerBlock
            # 一个SwinTransformerBlock是一个Attn+FFN
            self.layers.append(RSTB(dim=embed_dim,
                         depth=depth,
                         num_heads=num_heads,
                         window_size=window_size))
            
        # self.head = nn.Conv3d(in_channels=in_chans,out_channels=embed_dim,
        #                       kernel_size=3,stride=1,padding=1)
        # self.conv_after_body = nn.Conv3d(in_channels=embed_dim,out_channels=embed_dim,
        #                                  kernel_size=3,stride=1,padding=1)
        # self.tail = nn.Conv3d(in_channels=embed_dim,out_channels=num_out_ch,
        #                       kernel_size=3,stride=1,padding=1)
        self.head = nn.Conv2d(in_channels=in_chans,out_channels=embed_dim,
                              kernel_size=3,stride=1,padding=1)
        self.conv_after_body = nn.Conv2d(in_channels=embed_dim,out_channels=embed_dim,
                                         kernel_size=3,stride=1,padding=1)
        self.tail = nn.Conv2d(in_channels=embed_dim,out_channels=num_out_ch,
                              kernel_size=3,stride=1,padding=1)
        self.norm = norm_layer(embed_dim)
    
    def check_image_size(self, x):
        # B,C,T,H,W = x.shape
        B,C,H,W = x.shape
        mod_pad_h = (self.window_size - H % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - W % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x,[H+mod_pad_h,W+mod_pad_w]

    def forward(self, x):
        # B,C,T,H,W = x.shape
        B,C,H,W = x.shape
        x,x_sz_new = self.check_image_size(x)#补全成能被window partition的大小
        # 这里原来rgb的图还有一个减去均值的操作，想下咋加上去
        x_first = self.head(x)

        x_body = patch_embed(x_first)
        for layer in self.layers:
            x_body=layer(x_body,x_sz_new)

        res = self.conv_after_body(
            patch_unembed(self.norm(x_body),x_sz_new)
            ) + x_first
        x = x + self.tail(res)
        # x = x / self.img_range + self.mean
        return x[...,:H,:W]
    

class RSTB(nn.Module):#layer层attn+ffn
    """Residual Swin Transformer Block (RSTB)."""
    def __init__(self, dim, depth, num_heads, window_size,
                 drop_path=0., norm_layer=nn.LayerNorm, embed_dim=96):
        super(RSTB, self).__init__()
        self.dim = dim
        self.embed_dim=embed_dim
        self.layers=[]
        for i in range(depth):
            self.layers.append(SwinAttention(dim=dim,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer))
        # self.residual_group=nn.Sequential(*layers)
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self,x,x_size):
        for layer in self.layers:
            x=layer(x,x_size)
        return patch_embed(self.conv(patch_unembed(x, x_size))) + x
    

class SwinAttention(nn.Module):#一层attn+ffn
    """ Swin Transformer Block."""
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 drop_path=0.,act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads)
        
        self.norm2 = norm_layer(dim)
        self.depthconv = nn.Conv2d(dim, dim, 3, 1, 1,groups=dim)

        # 为啥shift_size>0就要算mask
        # if self.shift_size > 0:
        #     attn_mask = self.calculate_mask(self.input_resolution)
        # else:
        #     attn_mask = None

        # self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask # nW, window_size^2, window_size^2

    def forward(self, x,x_size):
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        # nW：总共划分得到的window个数
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        # 如果前面是shift之后的window算attn，现在要shift回来，也就是window attn有两种（shift或者不shift）
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B,-1,C)
        x = shortcut + x

        # FFN
        x = x.view(B,H,W,C).permute(0,3,1,2) + \
            self.depthconv(self.norm2(x).view(B,H,W,C).permute(0,3,1,2))
        x=x.permute(0,2,3,1).view(B,-1,C)#(bchw)->(bhwc)
        return x
    

class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window."""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape  #B_:window的个数：（504/8）^2
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # q,k,v:(B_,head,window_sz^2,C//head)，window_sz^2相当于是embedding的个数，即每个window内像素间算attn
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))    #@矩阵乘法运算 (B_,head,window_sz^2,window_sz^2)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # head, window_sz^2, window_sz^2
        attn = attn + relative_position_bias.unsqueeze(0)   
        #每个window都加一个相同的bias，bias似乎还是可学习的

        if mask is not None:
            nW = mask.shape[0]  #nW：window的个数
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition(x, window_size): #x: (B, H, W, C); window_size (int)
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows #(num_windows*B, window_size, window_size, C)


def window_reverse(windows, window_size, H, W):#windows:(num_windows*B, window_size, window_size, C)
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)  #(b,h,w,c)
    return x    #(B, H, W, C)


def patch_embed(x):
    """默认patch_sz=1时把整张图(bchw)->(b,h*w,c)"""
    B,C,H,W=x.shape
    return x.flatten(2).transpose(1, 2) # B Ph*Pw C   #(1,504*504,180)

def patch_unembed(x,x_size):
    """把(b,h*w,c)->(b,c,h,w),x_size是图片的大小"""
    B, HW, C = x.shape
    x = x.transpose(1, 2).view(B,-1,x_size[0],x_size[1])  # B Ph*Pw C
    return x


if __name__ == '__main__':
    upscale = 4
    window_size = 8
    height = 200
    width = 300
    model = Basic_Stage(num_layers=2,depth=6,in_chans=3,embed_dim=96,img_sz=224,patch_sz=1,
                 num_heads=6,window_size=7,norm_layer=nn.LayerNorm)
    # print(model)

    x = torch.randn((1, 3, height, width))
    x = model(x)
    print(x.shape)
