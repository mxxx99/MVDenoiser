import os
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
from models.resblocks import ResBlocks
from models.pytorch_pwc.utils import warp2center_5frame
from models.wave_transform import DWT,IWT
from models.mask_attentions import mask_att_map_guide,rescale,mask_att_noi_guide
from models.partialconv.codes.partialconv2d import PartialConv2d,PConvBlock
from torch.utils.tensorboard import SummaryWriter
import random
import numbers

# from resblocks import ResBlocks
# from pytorch_pwc.utils import warp2center_5frame
# from wave_transform import DWT,IWT
# from mask_attentions import mask_att_map_guide,rescale,mask_att_noi_guide
# from partialconv.codes.partialconv2d import PartialConv2d

from pathlib import Path
_script_dir = Path( __file__ ).parent
_root_dir = _script_dir.parent


@register
class Swin(nn.Module):
    '''Swin_args={'num_layers':2,'depth':2,'in_chans':n_channel_in,'embed_dim':112,'num_heads':7,
            'window_size':7,'t_length':t_length,'noise_cat':noise_cat,'device':self.device}'''
    def __init__(self,num_layers=2,depth=2,in_chans=4,embed_dim=96,
                 num_heads=6,window_size=7,t_length=5,noise_cat=True,
                 norm_layer=nn.LayerNorm,device="cuda",shuffle=True,scale=2,
                 wave_t=True,mask_ratio1=[20,30],mask_ratio2=[20,30],res=False,
                 input_mask=None,att_mask=False):
        '''in_chans是不包括noisemap在内的通道，
        mask_ratio1 for input mask &
        mask_ratio2 for att mask'''
        super(Swin, self).__init__()
        self.shuffle=shuffle
        self.num_layers=num_layers
        self.noise_cat=noise_cat
        self.wave_t=wave_t
        self.res=res

        self.mask_ratio1=mask_ratio1
        print(self.mask_ratio1)
        self.input_mask=input_mask

        self.in_chans=in_chans
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
        num_out_ch=4*(self.scale**2)
        self.window_size=window_size

        for i in range(num_layers):
            # 一个layer是一个RSTB；一个RSTB包括多个个SwinTransformerBlock
            # 一个SwinTransformerBlock是一个Attn+FFN
            setattr(self,'layer_%d'%i,RSTB(dim=embed_dim*t_length,
                         depth=depth,t_length=t_length,
                         num_heads=num_heads*t_length,
                         window_size=window_size,
                         use_mask=att_mask,
                         mask_ratio2=mask_ratio2))
            setattr(self,'conv_after_body_%d'%i,nn.Conv3d(in_channels=embed_dim,out_channels=embed_dim,
                            kernel_size=3,stride=1,padding=1,device=device))
            setattr(self,'norm_%d'%i ,norm_layer(embed_dim*t_length,device=device))
            getattr(self,'norm_%d'%i).requires_grad_=False
            if i==num_layers-1:
                setattr(self,'refine_%d'%i,Refine2(dim=embed_dim,t_length=t_length,
                        norm_layer=norm_layer,device=device))
            else:
                setattr(self,'refine_%d'%i,Refine(dim=embed_dim,t_length=t_length,
                            norm_layer=norm_layer,device=device))
        
        in_chan_with_noimap=in_chans+1 if noise_cat else in_chans
        self.head = nn.Conv3d(in_channels=in_chan_with_noimap,
                              out_channels=embed_dim,kernel_size=3,stride=1,padding=1,device=device)

        self.tail = nn.Sequential(nn.Conv3d(in_channels=embed_dim,out_channels=embed_dim,kernel_size=3,
                                        stride=1,padding=1,device=device),
                                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                        nn.Conv3d(in_channels=embed_dim,out_channels=embed_dim,
                                                kernel_size=3,stride=1,padding=1,device=device),
                                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                        nn.Conv3d(in_channels=embed_dim,out_channels=num_out_ch,
                                                kernel_size=3,stride=1,padding=1,device=device)
                                                )
        if self.res:
            self.res_cat=nn.Conv3d(in_channels=num_out_ch*2,out_channels=num_out_ch,
                                kernel_size=3,stride=1,padding=1,device=device)
    
    def check_image_size(self, x):
        B,C,T,H,W = x.shape
        mod_pad_h = (self.window_size - H % self.window_size) % self.window_size + self.window_size
        mod_pad_w = (self.window_size - W % self.window_size) % self.window_size + self.window_size
        # x = F.pad(x.contiguous().view(B,-1,H,W), (0, mod_pad_w, 0, mod_pad_h), 'reflect')\
        x = F.pad(x.contiguous().view(B,-1,H,W),(0, mod_pad_w, 0, mod_pad_h), 'reflect')\
            .view(B,C,T,H+mod_pad_h,W+mod_pad_w)
        return x,[T,H+mod_pad_h,W+mod_pad_w]

    def forward(self, x, flow=None):
        if self.noise_cat:
            noise_map=x[1]#(b,1,1,1,1)
            x=x[0]

        B,C,T,H0,W0 = x.shape
        if not flow==None:
            x_ori=x.clone().detach()
            x=warp2center_5frame(x,flow)

        if not self.input_mask==None:
            x,mask=self.add_input_mask(x,mask_mode=self.input_mask,r=self.mask_ratio1)

        if self.shuffle:
            x=self.down(x.transpose(1,2).contiguous().view(-1,C,H0,W0))
            _,C,H,W = x.shape
            x=x.view(B,-1,C,H,W).transpose(1,2)
        
        x,x_sz_new = self.check_image_size(x)#补全成能被window partition的大小

        if self.noise_cat:
            noise_map=noise_map.expand(B,1,T,x_sz_new[1],x_sz_new[2])

        x_input=torch.cat([x,noise_map],dim=1) if self.noise_cat else x

        x_first = self.head(x_input)
        x=x[:,:16]
        
        for i in range(self.num_layers):
            # 一个body包括一个swin layer和一个refine
            x_body = patch_embed(x_first)#(b,hw,ct)
            x_body=getattr(self,'layer_%d'%i)(x_body,x_sz_new)
            res = getattr(self,'conv_after_body_%d'%i)(
                patch_unembed(getattr(self,'norm_%d'%i)(x_body),x_sz_new)
                # getattr(self,'norm_%d'%i)(x_body),x_sz_new)
                )
            res=getattr(self,'refine_%d'%i)(res) + x_first
            x_first=res

        if self.res:
            x = self.res_cat(torch.cat(x,self.tail(res)))#最后一个stage这里可以只对一帧处理
        else:
            x = x + self.tail(res)#最后一个stage这里可以只对一帧处理

        inter=res.clone()
        for i in range(len(self.tail)):
            inter=self.tail[i](inter)
            if i==3:
                inter_feat=inter.clone().detach()
        x = x[...,:H,:W]


        if self.shuffle:
            B,C,T,H,W = x.shape
            x=self.up(x.transpose(1,2).contiguous().view(-1,C,H,W))
            _,C,H,W = x.shape
            x=x.view(B,-1,C,H,W).transpose(1,2)

        # x = x+x_mean
        
        return x,inter_feat,mask if self.input_mask else None

    
    def add_input_mask(self,seq,mask_mode='maskall_singlechannel',r=[30,80]):
        '''return:seq_masked:(b,c,t,h,w);  mask:(b,c,h,w)
        mask_mode: maskall/maskcenter/tube: 同时mask输入的5帧/1帧/5帧用同样的mask；
        multichannel/singlechannel: 对输入的4通道产生不同的mask'''
        B,C,T,H,W=seq.shape
        only_mask_center=True if 'maskcenter' in mask_mode else False
        tube_masking=True if 'tube' in mask_mode else False
        mask_channel= self.in_chans if 'multichannel' in mask_mode else 1

        if only_mask_center or tube_masking:
            size_=(B,mask_channel,H,W)
        else:
            size_=(B,mask_channel,T,H,W)

        prob = random.randint(r[0],r[1]) / 100
        mask = np.random.choice([0, 1], size=size_, p=[prob, 1 - prob])
        mask = torch.from_numpy(mask).to(seq.device)

        if only_mask_center:
            seq_masked=torch.cat(
                [seq[:,:,:T//2],torch.mul(seq[:,:,T//2],mask).unsqueeze(2),seq[:,:,(T//2+1):]],dim=2)
        elif tube_masking:
            seq_masked=torch.mul(seq,mask.unsqueeze(2))
        else:
            seq_masked=torch.mul(seq,mask)
            mask=mask[:,:,T//2]
        return seq_masked,mask


class RSTB(nn.Module):#layer层attn+ffn
    """Residual Swin Transformer Block (RSTB)."""
    def __init__(self, dim, depth, num_heads, window_size,t_length=5,
                 drop_path=0., norm_layer=nn.LayerNorm, embed_dim=96,
                 use_mask=True,mask_ratio2=0.7,device="cuda"):
        super(RSTB, self).__init__()
        self.dim = dim
        self.embed_dim=embed_dim
        self.num_blocks=depth
        for i in range(depth):
            setattr(self,'block_%d'%i,SwinAttention(dim=dim,num_heads=num_heads, 
                                window_size=window_size,t_length=t_length,
                                shift_size=0 if (i % 2 == 0) else window_size // 2,
                                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                use_mask=use_mask,mask_ratio2=mask_ratio2,
                                norm_layer=norm_layer,device=device))
        if t_length==1:
            self.conv = nn.Conv3d(dim//t_length, dim//t_length, (1,3,3), 1, (0,1,1),device=device)
        else:
            self.conv = nn.Conv3d(dim//t_length, dim//t_length, 3, 1, 1,device=device)

    def forward(self,x,x_size):
        for i in range(self.num_blocks):
            x=getattr(self,'block_%d'%i)(x,x_size)
        return patch_embed(self.conv(patch_unembed(x, x_size))) + x
    

class SwinAttention(nn.Module):#一层attn+ffn
    """ Swin Transformer Block."""
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,t_length=5,
                 drop_path=0.,act_layer=nn.GELU, mask_ratio2=[30,80],
                 use_mask=True,norm_layer=nn.LayerNorm,device="cuda"):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mask_ratio2=mask_ratio2
        self.use_mask=use_mask
        
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim,device=device)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,device=device)
        
        self.norm2 = norm_layer(dim,device=device)
        if t_length==1:
            self.depthconv = nn.Conv3d(dim//t_length, dim//t_length, (1,3,3), 1, (0,1,1),groups=dim//t_length,device=device)
        else:
            self.depthconv = nn.Conv3d(dim//t_length, dim//t_length, 3, 1, 1,groups=dim//t_length,device=device)
        # self.depthconv = nn.Sequential(
        #                     nn.Conv3d(dim//t_length, dim//t_length, 1, 1, 0,device=device),
        #                     nn.Conv3d(dim//t_length, dim//t_length, 3, 1, 1,groups=dim//t_length,device=device))

        # 为啥shift_size>0就要算mask
        # if self.shift_size > 0:
        #     attn_mask = self.calculate_mask(self.input_resolution)
        # else:
        #     attn_mask = None

        # self.register_buffer("attn_mask", attn_mask)


    def mask_feature(self, image, x_size=None,r=[30,80]):
        # attention mask

        # if not self.mask_is_diff:
        prob_ = random.randint(r[0],r[1]) / 100
        mask1 = np.random.choice([0, 1], size=(image.shape[0], image.shape[1]), p=[prob_, 1 - prob_])
        mask1 = torch.from_numpy(mask1).to(image.device).unsqueeze(-1)
        noise_image1 = torch.mul(image, mask1)
        #这个mask是在一个位置上mask掉所有通道，我们能不能考虑把通道分成几份mask（和multi-head一起考虑）
        return noise_image1

        # elif self.mask_is_diff:
        #     mask_images = []
        #     for i in range(self.num_heads):
        #         prob_ = random.randint(self.mask_ratio1, self.mask_ratio2) / 100
        #         mask1 = np.random.choice([0, 1], size=(image.shape[0], image.shape[1]), p=[prob_, 1 - prob_])
            #     mask1 = torch.from_numpy(mask1).to(image.device).unsqueeze(-1)
            #     noise_image1 = torch.mul(image, mask1)

            #     mask_images.append(noise_image1)
            # return mask_images


    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        T, H, W = x_size
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
        T, H, W = x_size
        B, L, C = x.shape

        # attention mask
        if self.use_mask:
            x = self.mask_feature(x,r=self.mask_ratio2)
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
            self.depthconv(self.norm2(x).view(B,H,W,C//T,T).permute(0,3,4,1,2)).view(B,-1,H,W)
        x=x.permute(0,2,3,1).view(B,-1,C)#(bchw)->(bhwc)
        del attn_windows,shortcut,x_windows
        return x
    

class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window."""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, 
                 attn_drop=0., proj_drop=0.,device="cuda"):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads,device=device))  # 2*Wh-1 * 2*Ww-1, nH

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

        # self.qkv = nn.Conv2d(dim,dim*3,1,padding=0,bias=qkv_bias,device=device)
        # self.qkv_dwconv=nn.Conv2d(dim*3,dim*3,3,padding=1,groups=dim*3,bias=qkv_bias,device=device)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias,device=device)#这个linear如果换成卷积会更省权重

        self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Conv2d(dim,dim,1,padding=0,bias=qkv_bias,device=device)
        self.proj = nn.Linear(dim, dim, device=device)

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
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)#(3,B_,heads,N,C//heads)
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
        del attn,relative_position_bias,q,k,v
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
    """默认patch_sz=1时把整张图(bcthw)->(b,h*w,c*t)"""
    B,C,T,H,W=x.shape
    return x.flatten(3).permute(0,3,1,2).view(B,H*W,-1) # B Ph*Pw C*T   #(1,504*504,180)

def patch_unembed(x,x_size):
    """把(b,h*w,c)->(b,c,t,h,w),x_size是图片的大小
    x_size:(t,h,w)"""
    B, HW, CT = x.shape
    x = x.transpose(1, 2).view(B,-1,x_size[0],x_size[1],x_size[2])
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
        # self.pool=nn.AdaptiveAvgPool3d((1,1,1))#(b,t,c=1,h,w)
        # self.conv=nn.Conv3d(dim, dim, 3, 1, 1,device=device)
        self.conv=nn.Sequential(nn.Conv3d(dim,dim,3,1,1,device=device),
                                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                nn.Conv3d(dim,1,3,1,1,device=device))
        self.act=nn.Sigmoid()
    def forward(self, x):#, x_size):
        B,C,T,H,W=x.shape
        feat=self.act(self.conv(x))#(b,c=1,t,h,w)
        return feat*x

# class Refine(nn.Module):
#     def __init__(self, dim, t_length=5, norm_layer=nn.LayerNorm,device="cuda",bias=False):
#         super(Refine, self).__init__()
#         self.pool=nn.AdaptiveAvgPool3d((None,1,1))#(b,c,t,h=1,w=1)
#         # self.conv=nn.Conv3d(dim, 1, 3, 1, 1,device=device)
#         self.conv=nn.Sequential(nn.Conv3d(dim,dim,1,1,0,device=device),
#                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
#                                 nn.Conv3d(dim,dim,1,1,0,device=device))
#         self.act=nn.Sigmoid()
#     def forward(self, x):#, x_size):
#         B,C,T,H,W=x.shape
#         feat=self.act(self.conv(self.pool(x)))
#         return feat*x
    
# class Refine(nn.Module):
#     def __init__(self, dim, t_length=5, norm_layer=nn.LayerNorm,device="cuda",bias=False):
#         super(Refine, self).__init__()
#         self.pool=nn.AdaptiveAvgPool3d((None,1,1))#(b,c,t,h=1,w=1)
#         # self.conv=nn.Conv3d(dim, 1, 3, 1, 1,device=device)
#         self.conv=nn.Sequential(nn.Conv3d(dim,dim,1,1,0,device=device),
#                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
#                                 nn.Conv3d(dim,dim,1,1,0,device=device))
#         self.act=nn.Sigmoid()
#     def forward(self, x):#, x_size):
#         B,C,T,H,W=x.shape
#         feat=self.act(self.conv(self.pool(x)))
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
        

class MDTA(nn.Module):
    def __init__(self, dim, num_heads, t_length=5, norm_layer=nn.LayerNorm,device="cuda",bias=False):
        super(MDTA, self).__init__()

        # self.norm1_channel = norm_layer(dim,device=device)
        self.norm1_time = norm_layer(t_length,device=device)
        # self.attn_channel = Attention(dim, num_heads, bias)
        self.attn_time = Attention(t_length, 1, bias)
        # self.fusion=nn.Conv3d(dim*2,dim,1,1,0)
        self.norm2 = norm_layer(dim*t_length,device=device)
        self.ffn = nn.Conv3d(dim, dim, 3, 1, 1,groups=dim,device=device)

    def forward(self, x):#, x_size):
        B,C,T,H,W=x.shape

        # 再仔细研究一下norm的定义
        # if 1==0:
        res=self.attn_time(rearrange(
                self.norm1_time(rearrange(x, 'b c t h w -> (b c) (h w) t'))
                ,'bc (h w) t -> bc t h w',h=H,w=W)).view(B,C,T,H,W)
        # else:
        #     res=self.fusion(torch.cat([
        #         self.attn_channel(rearrange(
        #                         self.norm1_channel(rearrange(x, 'b c t h w -> (b t) (h w) c'))
        #                         ,'bt (h w) c -> bt c h w',h=H,w=W)).view(B,T,C,H,W).transpose(1,2).contiguous(),
        #         self.attn_time(rearrange(
        #                         self.norm1_time(rearrange(x, 'b c t h w -> (b c) (h w) t'))
        #                         ,'bc (h w) t -> bc t h w',h=H,w=W)).view(B,C,T,H,W)
        #     ],dim=1))
        x = x + res
        x = x + self.ffn(patch_unembed(self.norm2(patch_embed(x)),[T,H,W]))

        return x


## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))#(b,c,h,w)->(b,3c,h,w)
        q,k,v = qkv.chunk(3, dim=1)#q,k,v=(b,c,h,w)
        
        # 在c上做att时，算相似度时要不要考虑t？就是输入的是(bt head c (h w))还是(b head c (t h w))
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        del q,k,v,qkv,attn
        return out


if __name__ == '__main__':

    window_size = 7
    height = 130
    width = 200
    device = torch.device('cuda:0')

    model = Swin(num_layers=1,depth=2,in_chans=4,embed_dim=64,is_last=False,
                 num_heads=4,window_size=7,t_length=5,norm_layer=nn.LayerNorm,device=device,
                 shuffle=True,scale=2,wave_t=False,input_mask=True,att_mask=True,pconv=True,
                 mask_att='noi_guide')
    model=model.to(device)
    writer = SummaryWriter("logs")
    writer.add_graph(model, torch.empty([2, 10]))
    # print(model)
    
    x = [torch.randn((2, 4, 5,height, width)).to(device),torch.zeros((1,1,1,1,1)).to(device)]
    with torch.no_grad():
        x,_ = model(x)
    print(x.shape)


