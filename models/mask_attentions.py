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
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import random
import numbers

def rescale(input,ratio_range=[0.3,0.7]):
    x=(input-input.min())/(input.max()-input.min())
    out=x*(ratio_range[1]-ratio_range[0])+ratio_range[0]
    return out

# def mask_att_max_channel(img,ratio_range=[0.3,0.7]):
#     '''img:(b,c,h,w);  ratio_range:[min,max]'''
#     # G = torch.Generator()
#     # G.manual_seed(random.randint(1,1000))
#     r_min,r_max=ratio_range
#     max_channel=torch.max(img,dim=1,keepdim=True).values
#     max_channel=(max_channel-max_channel.min())/(max_channel.max()-max_channel.min())
#     mask=torch.bernoulli(
#                     (1-max_channel)*(r_max-r_min)+r_min#,generator=G
#         ).to(img.device)
#     return mask

def mask_att_map_guide(attn_map,device='cuda:0'):
    '''img:(b,c,h,w);  ratio_range:[min,max]'''
    mask=torch.bernoulli((1-attn_map)).to(device)
    return mask


def mask_att_noi_guide(noi_map,ratio=0.8,device='cuda:0'):
    '''img:(b,c,h,w);  ratio_range:[min,max]'''
    # G = torch.Generator()
    # G.manual_seed(random.randint(1,1000))  
    B,C,H,W=noi_map.shape
    noi_map_=noi_map.clone().detach()
    mask=[]
    for i in range(B):
        map_=torch.norm(torch.abs(noi_map_[i]),p=2,dim=0,keepdim=True)
        value,_=torch.kthvalue(map_.view(-1),k=math.ceil(ratio*(H*W)))
        map_[map_<value]=0;map_[map_>=value]=1;map_=1-map_
        mask.append(map_)
    return torch.stack(mask,dim=0)


if __name__ == '__main__':
    x=Image.open('/data3/mxx/Noise_generate/Starlight_ours/scripts/ll_out.png')
    # x=Image.open('./mountain.png')
    x=transforms.ToTensor()(x)
    print(torch.max(x),torch.min(x))
    x=torch.unsqueeze(x,0)
    # mask=mask_att_max_channel(x,[0.0,0.7])
    # plt.figure()
    # plt.imshow(mask[0].permute(1,2,0))
    # plt.savefig('/data3/mxx/Noise_generate/Starlight_ours/scripts/mask3.png')