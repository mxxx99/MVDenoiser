import os
os.environ['CUDA_VISIBLE_DEVICES']='5'

import sys
sys.path.append('..')

import torchvision.models
import torch
from thop import profile
from thop import clever_format
import numpy as np
from models.stage_denoiser import Stage_denoise3
from models.Swin import Swin,RSTB,SwinAttention,WindowAttention
# from models.Swin_v2 import Swin_v2
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import nn

# unet_opts='residualFalse_conv_tconv_selu'
# keys=['shot','read', 'uniform']#, 'fixed']
# res_opt = bool(unet_opts.split('_')[0].split('residual')[-1])
# model = Stage_denoise3(n_channel_in=4, 
#             residual=res_opt, 
#             down=unet_opts.split('_')[1], 
#             up=unet_opts.split('_')[2], 
#             activation=unet_opts.split('_')[3],keys=keys[::-1])

# model = Swin(num_layers=2,depth=2,in_chans=4,embed_dim=112,
#                 num_heads=7,window_size=7,t_length=5,norm_layer=nn.LayerNorm)
# model = Swin(num_layers=2,depth=2,in_chans=4,embed_dim=112,
#                 num_heads=7,window_size=(3,7,7),t_length=5,norm_layer=nn.LayerNorm)

# model=RSTB(dim=112*5,depth=2,t_length=5,num_heads=7*5,window_size=7)

model=SwinAttention(dim=112*5,num_heads=7*5,window_size=7,t_length=5,shift_size=0,# else window_size // 2,
                                drop_path=0)

# model=WindowAttention(112*5, window_size=to_2tuple(7), num_heads=7*5)

# 统计各部分参数量
_dict = {}
total=0
for _,param in enumerate(model.named_parameters()):
    # print(param[0])
    # print(param[1])
    total_params = param[1].numel()
    # print(f'{total_params:,} total parameters.')
    k = param[0].split('.')[0]
    if k in _dict.keys():
        _dict[k] += total_params
    else:
        _dict[k] = 0
        _dict[k] += total_params
    # print('----------------')
for k,v in _dict.items():
    print(k)
    print(v)
    print("%3.3fM parameters" %  (v / (1024*1024)))
    total+=(v / (1024*1024))
    print('--------')

print("%3.3fM parameters in total" %  total)