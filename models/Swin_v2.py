import os
# os.environ["CUDA_VISIBLE_DEVICES"]='7'
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
from models.register import register
from models.pytorch_pwc.utils import warp2center_5frame
from models.wave_transform import DWT,IWT

# from register import register
# from pytorch_pwc.utils import warp2center_5frame
# from wave_transform import DWT,IWT
import numbers

from pathlib import Path
_script_dir = Path( __file__ ).parent
_root_dir = _script_dir.parent

@register
class Swin_v2(nn.Module):
    '''Swin_args={'num_layers':2,'depth':2,'in_chans':n_channel_in,'embed_dim':112,'num_heads':7,
            'window_size':7,'t_length':t_length,'noise_cat':noise_cat,'device':self.device}'''
    def __init__(self,num_layers=2,depth=2,in_chans=4,embed_dim=96,
                 num_heads=6,window_size=(3,7,7),t_length=5,noise_cat=False,
                 norm_layer=nn.LayerNorm,device="cuda",shuffle=True,scale=2,wave_t=True):
        '''in_chans是不包括noisemap在内的通道'''
        super(Swin_v2, self).__init__()
        self.shuffle=shuffle
        self.num_layers=num_layers
        self.noise_cat=noise_cat
        self.wave_t=wave_t
        if self.shuffle:
            self.scale=scale
            if self.wave_t:#用小波变换下采样
                assert self.scale==2#只支持2倍
                self.down=DWT()
                self.up=IWT()
            else:#用shuffle下采样
                self.down=nn.PixelUnshuffle(downscale_factor=self.scale)
                self.up=nn.PixelShuffle(upscale_factor=self.scale)
            in_chans=in_chans*(self.scale**2)
        num_out_ch=in_chans
        self.window_size=window_size

        for i in range(num_layers):
            # 一个layer是一个RSTB；一个RSTB包括多个个SwinTransformerBlock
            # 一个SwinTransformerBlock是一个Attn+FFN
            setattr(self,'layer_%d'%i,RSTB(dim=embed_dim,
                         depth=depth,t_length=t_length,
                         num_heads=num_heads,
                         window_size=window_size))
            setattr(self,'conv_after_body_%d'%i,nn.Conv3d(in_channels=embed_dim,out_channels=embed_dim,
                            kernel_size=3,stride=1,padding=1,device=device))
            setattr(self,'norm_%d'%i ,norm_layer(embed_dim,device=device))
            if i==num_layers-1:
                setattr(self,'refine_%d'%i,Refine2(dim=embed_dim,t_length=t_length,
                        norm_layer=norm_layer,device=device))
            else:
                setattr(self,'refine_%d'%i,Refine(dim=embed_dim,t_length=t_length,
                            norm_layer=norm_layer,device=device))
            
        self.head = nn.Conv3d(in_channels=in_chans+1 if noise_cat else in_chans,
                              out_channels=embed_dim,kernel_size=3,stride=1,padding=1,device=device)
        self.tail = nn.Sequential(nn.Conv3d(in_channels=embed_dim,out_channels=embed_dim,
                                                        kernel_size=3,stride=1,padding=1,device=device),
                                                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                nn.Conv3d(in_channels=embed_dim,out_channels=embed_dim,
                                                        kernel_size=3,stride=1,padding=1,device=device),
                                                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                nn.Conv3d(in_channels=embed_dim,out_channels=num_out_ch,
                                                        kernel_size=3,stride=1,padding=1,device=device))
        # self.refine=Refine2(dim=embed_dim,t_length=t_length,
        #                     norm_layer=norm_layer,device=device)
        # self.norm=norm_layer(embed_dim*t_length,device=device)
        

    def forward(self, x, flow=None):
        if self.noise_cat:
            noise_map=x[1]#(b,1,1,1,1)
            x=x[0]

        B,C,T,H0,W0 = x.shape
        if not flow==None:
            x_ori=x.clone().detach()
            x=warp2center_5frame(x,flow)

        if self.shuffle:
            x=self.down(x.transpose(1,2).contiguous().view(-1,C,H0,W0))
            _,C,H,W = x.shape
            x=x.view(B,-1,C,H,W).transpose(1,2).contiguous()
        
        if self.noise_cat:
            noise_map=noise_map.expand(B,1,T,H,W)

        x_first = self.head(torch.cat([x,noise_map],dim=1) if self.noise_cat else x)#head要不要也加上reslearn
        
        for i in range(self.num_layers):
            # 一个body包括一个swin layer和一个refine
            x_body = x_first
            x_body=getattr(self,'layer_%d'%i)(x_body)
            res = getattr(self,'conv_after_body_%d'%i)(
                        (getattr(self,'norm_%d'%i)(x_body.permute(0,2,3,4,1).contiguous()))
                                    .permute(0,4,1,2,3).contiguous()
                        )
                # getattr(self,'norm_%d'%i)(x_body),x_sz_new)
            res=getattr(self,'refine_%d'%i)(res) + x_first
            x_first=res

        x = x + self.tail(res)#最后一个stage这里可以只对一帧处理
        x = x[...,:H,:W].contiguous()

        if self.shuffle:
            B,C,T,H,W = x.shape
            x=self.up(x.transpose(1,2).contiguous().view(-1,C,H,W))
            _,C,H,W = x.shape
            x=x.view(B,-1,C,H,W).transpose(1,2).contiguous()
        # x = x+x_mean
        
        return x
    

class RSTB(nn.Module):#layer层attn+ffn
    """Residual Swin Transformer Block (RSTB)."""
    def __init__(self, dim, depth, num_heads, window_size,t_length=5,
                 drop_path=0.0, norm_layer=nn.LayerNorm, embed_dim=96, device="cuda"):
        super(RSTB, self).__init__()
        self.dim = dim
        self.embed_dim=embed_dim
        self.num_blocks=depth
        self.window_size=window_size
        self.shift_size=list(j // 2 for j in window_size)

        for i in range(depth):
            setattr(self,'block_%d'%i,SwinAttention3D(dim=dim,num_heads=num_heads, 
                                window_size=window_size,t_length=t_length,
                                shift_size=(0,0,0) if (i % 2 == 0) else self.shift_size,
                                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                norm_layer=norm_layer,device=device))

    def forward(self,x):
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D,H,W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)

        for i in range(self.num_blocks):
            x=getattr(self,'block_%d'%i)(x,attn_mask)

        x = rearrange(x, 'b d h w c -> b c d h w')
        return x
    

class SwinAttention3D(nn.Module):#一层attn+ffn
    """ Swin Transformer Block."""
    def __init__(self, dim, num_heads, window_size=(3,7,7), shift_size=(0,0,0),t_length=5,
                 drop_path=0.0,act_layer=nn.GELU, norm_layer=nn.LayerNorm,device="cuda"):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        
        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim,device=device)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,device=device)
        
        self.norm2 = norm_layer(dim,device=device)
        self.depthconv = nn.Sequential(
                            nn.Conv3d(dim, dim, 1, 1, 0,device=device),
                            nn.Conv3d(dim, dim, 3, 1, 1,groups=dim,device=device))
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def check_seq_size(self, x, window_size):
        B,C,D,H,W = x.shape
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]#D
        pad_h = (window_size[1] - H % window_size[1]) % window_size[1]#H
        pad_w = (window_size[2] - W % window_size[2]) % window_size[2]#W
        x = F.pad(x.contiguous().view(B,-1,H,W), (pad_l, pad_w, pad_t, pad_h), mode='reflect').view(B,C,D,H+pad_h,W+pad_w)
        if pad_d1>0:
            x=torch.cat([x,torch.flip(x[:,:,-1-pad_d1:-1],dims=[2])],dim=2)
        # x = F.pad(x.contiguous().view(B,-1,H,W),(0, mod_pad_w, 0, mod_pad_h), 'reflect')\
        #     .view(B,C,T,H+mod_pad_h,W+mod_pad_w)
        return x,pad_d1,pad_h,pad_w


    def forward(self, x, attn_mask):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        # 把constant改成reflect应该ok

        shortcut = x
        x,pad_d1,pad_h,pad_w=self.check_seq_size(x.permute(0,4,1,2,3).contiguous(),window_size)
        x=x.permute(0,2,3,4,1).contiguous()#(b,c,d,h,w)->(b,d,h,w,c)
        x = self.norm1(x)

        _, Dp, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            # 用传过来的attn_mask
        else:
            shifted_x = x
            attn_mask=None

        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # nW*B, window_size, window_size, C
        # nW：总共划分得到的window个数

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, self.window_size, B, Dp, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        # 如果前面是shift之后的window算attn，现在要shift回来，也就是window attn有两种（shift或者不shift）
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_w > 0 or pad_h > 0:
            x = x[:, :D, :H, :W, :]

        x = shortcut + self.drop_path(x)    #(b,t,h,w,c)

        # FFN
        x = x + \
            self.depthconv(self.norm2(x).permute(0,4,1,2,3).contiguous()).permute(0,2,3,4,1).contiguous()

        del attn_windows,shortcut,x_windows
        return x
    

class WindowAttention3D(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window."""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, 
                 attn_drop=0.0, proj_drop=0.0,device="cuda"):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads,device=device))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", self.get_position_index(window_size))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias,device=device)#这个linear如果换成卷积会更省权重
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop>0. else nn.Identity()

        self.proj = nn.Linear(dim, dim, device=device)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop>0. else nn.Identity()

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)


    def get_position_index(self, window_size):
        ''' Get pair-wise relative position index for each token inside the window. '''
        coords_d = torch.arange(window_size[0])
        coords_h = torch.arange(window_size[1])
        coords_w = torch.arange(window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 2] += window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww

        return relative_position_index
    

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape  #B_:window的个数：（504/8）^2
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)#(3,B_,heads,N,C//heads)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # q,k,v:(B_,head,window_sz^2,C//head)，window_sz^2相当于是embedding的个数，即每个window内像素间算attn
        attn = (q * self.scale @ k.transpose(-2, -1).contiguous())    #@矩阵乘法运算 (B_,head,window_sz^2,window_sz^2)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:N, :N].reshape(-1)].reshape(N,N,-1)
        # head, window_sz^2, window_sz^2
        attn = attn + relative_position_bias.permute(2, 0, 1).unsqueeze(0).contiguous()
        #每个window都加一个相同的bias，bias似乎还是可学习的

        if mask is not None:
            nW = mask.shape[0]  #nW：window的个数
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask[:, :N, :N].unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).contiguous().reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        del attn,relative_position_bias,q,k,v
        return x


def window_partition(x, window_size): #x: (B, H, W, C); window_size (int)
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2],
               window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """ Reverse windows back to the original input. Attention was conducted within the windows.

    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1],
                     window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)

    return x


# class Refine(nn.Module):
#     def __init__(self, dim, t_length=5, norm_layer=nn.LayerNorm,device="cuda",bias=False):
#         super(Refine, self).__init__()
#         self.conv=nn.Conv3d(dim, dim, 3, 1, 1,device=device, groups=dim)
#         self.conv_gate=nn.Sequential(nn.Conv3d(dim,dim,3,1,1,device=device),
#                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
#                                 nn.Conv3d(dim,dim,3,1,1,device=device))
#         self.act=nn.Sigmoid()
#     def forward(self, x):#, x_size):
#         B,C,T,H,W=x.shape
#         feat=self.act(self.conv_gate(x))#(b,c,t,h,w)
#         return feat*self.conv(x)

class Refine(nn.Module):
    def __init__(self, dim, t_length=5, norm_layer=nn.LayerNorm,device="cuda",bias=False):
        super(Refine, self).__init__()
        self.pool=nn.AdaptiveAvgPool3d((None,1,1))#(b,c,t,h=1,w=1)
        self.conv=nn.Conv3d(dim, dim, 1, 1, 0,device=device)
        # self.conv=nn.Sequential(nn.Conv3d(dim,dim,1,1,0,device=device),
        #                         nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #                         nn.Conv3d(dim,dim,1,1,0,device=device))
        self.act=nn.Sigmoid()
    def forward(self, x):#, x_size):
        B,C,T,H,W=x.shape
        feat=self.act(self.conv(self.pool(x)))
        return feat*x
    
    
# class Refine(nn.Module):
#     def __init__(self, dim, t_length=5, norm_layer=nn.LayerNorm,device="cuda",bias=False):
#         super(Refine, self).__init__()
#         self.pool=nn.AdaptiveAvgPool3d((1,1,1))#(b,t,c=1,h,w)
#         # self.conv=nn.Conv3d(dim, 1, 3, 1, 1,device=device)
#         self.conv=nn.Sequential(nn.Conv3d(dim,dim,3,1,1,device=device),
#                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
#                                 nn.Conv3d(dim,1,3,1,1,device=device))
#         self.act=nn.Sigmoid()
#     def forward(self, x):#, x_size):
#         B,C,T,H,W=x.shape
#         feat=self.act(self.conv(x))#(b,c=1,t,h,w)
#         return feat*x
    

class Refine2(nn.Module):
    def __init__(self, dim, t_length=5, embed_dim=64, norm_layer=nn.LayerNorm,device="cuda",norm=False):
        '''mode:pad或者nopad'''
        super(Refine2, self).__init__()
        self.t_length=t_length
        self.embed = nn.Conv3d(dim, embed_dim*2, 1, 1, 0)#embed的dim可以少一些
        self.embed_dwconv = nn.Conv3d(embed_dim*2, embed_dim*2, 3, 1, 1, groups=embed_dim*2)
        self.project = nn.Conv3d(1, 1, 3, 1, 1)
        self.norm=norm
        self.sigma2=0.05
        if self.norm:
            self.norm=norm_layer(dim,device=device)
        self.act=nn.Sigmoid()

    def forward(self, x):#, x_size):
        B,C,T,H,W=x.shape
        if self.norm:
            embed = self.embed_dwconv(self.embed(
                self.norm(x.permute(0,2,3,4,1).contiguous()).permute(0,4,1,2,3).contiguous()#(b,c,t,h,w)->(b,t,h,w,c)->(b,c,t,h,w)
                ))#(b,c,t,h,w)->(b,3c,t,h,w)
        else:
            embed = self.embed_dwconv(self.embed(x))#(b,c,t,h,w)->(b,3c,t,h,w)
        nbr,ref = embed.chunk(2, dim=1)

        out=[]
        for i in range(self.t_length):
            # 之前的版本，有sigmoid无mask，不改了，怎么改都不如原来的好
            sim_i=self.act(self.project(
                torch.mean(torch.mul(nbr,ref[:,:,i,...].unsqueeze(2)),dim=1,keepdim=True)
                ))
            out.append(torch.mean(x*sim_i,dim=2,keepdim=True))
        out=torch.cat(out,dim=2)
        return out


def get_window_size(x_size, window_size, shift_size=None):
    """ Get the window size and the shift size """

    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)
    

@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    """ Compute attnetion mask for input of size (D, H, W). @lru_cache caches each stage results. """

    img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask

if __name__ == '__main__':

    window_size = 7
    height = 200
    width = 300
    device = torch.device('cuda:0')
    
    model = Swin_v2(num_layers=1,depth=2,in_chans=3,embed_dim=64,
                 num_heads=4,window_size=(3,7,7),t_length=5,norm_layer=nn.LayerNorm,device=device)
    model=model.to(device)
    # print(model)
    
    x = torch.randn((1, 3, 5,height, width)).to(device)
    with torch.no_grad():
        x = model(x)
    print(x.shape)


