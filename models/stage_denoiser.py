import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io

from models.modules import ConvBlock3d
from models.Swin import Swin,Refine2    #swin和unet3d里有装饰器，所以即便在这里不用也要import一下
from models.Swin_v2 import Swin_v2
from models.Unet3d import Unet3d
from models.register import get_class
from models.pytorch_pwc.pwc import PWCNet
from models.pytorch_pwc.extract_flow import extract_flow_torch
from models.pytorch_pwc.utils import warp,demosaic
import argparse, json, glob, os, sys

from pathlib import Path
_script_dir = Path( __file__ ).parent
_root_dir = _script_dir.parent



class Stage_denoise3(nn.Module):
    # 针对每个噪声类型用不同的网络去噪
    def __init__(self, n_channel_in=1, device="cuda:0", residual=False, down='conv', up='tconv', 
                 activation='selu',keys=[],res_learn=False,noise_cat=True,base_model='Swin',\
                    t_length=5,co_training=False,input_mask=None,mask_ratio=[20,30]):
        super(Stage_denoise3, self).__init__()
        self.keys=keys
        self.basemodel={}
        self.res_learn=res_learn#残差学习
        self.predicted_noise={}#每个去噪阶段去除的噪声
        self.device = device
        self.dtype=torch.float32
        self.noise_cat=noise_cat
        self.noise_levels=5
        self.warp=True
        if self.warp:
            self.pwcnet = PWCNet(train=co_training)
            self.co_training=co_training
            # self.pwcnet.requires_grad_(False)

        # 这个args后续改成从json文件导入
        Swin_args={'num_layers':2,'depth':2,'in_chans':n_channel_in,'embed_dim':112,'num_heads':7,
                   'window_size':7,'t_length':t_length,'noise_cat':noise_cat,'device':self.device}
        Swin_args_first={'num_layers':2,'depth':2,'in_chans':n_channel_in,'embed_dim':112,'num_heads':7,
                   'window_size':7,'t_length':t_length,'noise_cat':noise_cat,
                   'input_mask':input_mask,'mask_ratio1':mask_ratio,'device':self.device}
        
        if 'shot' in self.keys:#generator生成的噪声是用一个随机分布乘一个可学习的参数组成的
            # 我们可以根据噪声图估计噪声参数（这部分的参数可以用generator指导去学习），然后再去噪
            self.basemodel_shot = get_class()[base_model](**Swin_args)
        if 'read' in self.keys:
            self.basemodel_read = get_class()[base_model](**Swin_args)
        if 'uniform' in self.keys:
            self.basemodel_uniform = get_class()[base_model](**Swin_args_first)
        if 'fixed' in self.keys:#(b,c,t,h,w)->(b,c,1,h,w)
            C,H,W=self.fixed_shape
            init_fixed_coeff=0.0
            self.fixed_coeff = torch.nn.Parameter(torch.tensor(init_fixed_coeff, dtype = self.dtype, device = device).repeat(self.noise_levels)
                                        , requires_grad = True)

    def trainable_parameters(self):
        params=[]
        for idx,key in enumerate(self.keys):
            params.append(getattr(self,'basemodel_%s'%key).parameters())
        # model=getattr(self,'basemodel_shot')
        # params.append(model.tail.parameters())
        return params

    def denoise_stage(self,x_input,key,flow=None):
        # 我们可以根据噪声图估计噪声参数（这部分的参数可以用generator指导去学习），然后再去噪
        x_out,inter_feat,mask=eval('self.basemodel_%s'%key)(x_input,flow)
        predicted_noise=((x_input[0]-x_out) if self.noise_cat else (x_input-x_out)).detach()
        # x_out:去噪后的结果；denoised_inter:去噪前的结果；predicted_noise:key阶段分离出的噪声
        return x_out,predicted_noise,inter_feat,mask
    

    def extract_flow_5frame(self,x,downscale=1):
        B,C,T,H,W=x.shape
        srgb_seqn = demosaic(x.transpose(1,2))
        flow = {}
        flow['backward']=torch.empty((B, T//2, 2, H, W), device=self.device)
        flow['forward']=torch.empty((B, T//2, 2, H, W), device=self.device)
        for i in range(0,T//2):#i=0,1
            # flow['forward'][:,i] = extract_flow_torch(self.pwcnet, srgb_seqn[:, i+1], srgb_seqn[:, i])#i向i+1对齐
            # flow['backward'][:,i] = extract_flow_torch(self.pwcnet, srgb_seqn[:, T-i-2], srgb_seqn[:, T-i-1])#T-i-1向T-i-2对齐
            flow['forward'][:,i] = extract_flow_torch(self.pwcnet, srgb_seqn[:, T//2], srgb_seqn[:, i],\
                                                      downscale,co_training=self.co_training)#forward:0:0->2; 1:1->2
            flow['backward'][:,i] = extract_flow_torch(self.pwcnet, srgb_seqn[:, T//2], srgb_seqn[:, T-i-1],\
                                                       downscale,co_training=self.co_training)#0:4->2; 1:3->2
        return flow


    def forward(self, x, clean, noise_level=None, noise_ind=0):
        # i0是裁剪图片的位置，和fix噪声的内容有关，不允许输入为空
        # noise_level是合成的噪声级别，是一个dict？
        # noise_level的每种噪声都是(B,1,1,1,1)
        # assert pos is not None
        B,C,T,H,W=x.shape
        if self.warp:
            flow=self.extract_flow_5frame(x)

        denoised_inter={}
        predicted_noise={}
        inter_feat={}
        x_mean=torch.mean(x,dim=[2,3,4],keepdim=True)#每个seq每个channel算一个均值
        clean_=x
        x_mean_clean=torch.mean(clean_,dim=[2,3,4],keepdim=True)
        x=x-x_mean# 这里原来rgb的图还有一个减去均值的操作
        
                
            # 如果把上一个state的model的中间状态也输进来结果应该会更好
            # fixed这里会报错，似乎是一个函数修改了inplace的一个元素，导致梯度反传出现问题
            
        # row和rowt的unet对应的之后的
        if 'uniform' in self.keys:
            denoised_inter['uniform']=x.clone().detach()+x_mean_clean
            x,predicted_noise['uniform'],inter_feat['uniform'],mask=\
                self.denoise_stage([x,noise_level['uniform']] if self.noise_cat else x,'uniform',\
                                   flow if self.warp else None)
        
        if 'read' in self.keys:
            denoised_inter['read']=x.clone().detach()+x_mean_clean
            x,predicted_noise['read'],inter_feat['read'],_=\
                self.denoise_stage([x,noise_level['read']] if self.noise_cat else x,'read')
            
        if 'shot' in self.keys:#generator生成的噪声是用一个随机分布乘一个可学习的参数组成的
            denoised_inter['shot']=x.clone().detach()+x_mean_clean
            x,predicted_noise['shot'],inter_feat['shot'],_=\
                self.denoise_stage([x,noise_level['shot']] if self.noise_cat else x,'shot')#,\
                                #    flow if self.warp else None)
        # x+denoised_inter[shot]+denoised_inter[read]+denoised_inter[uniform]+denoised_inter[fixed]=input

        x_mean_clean=torch.mean(clean_,dim=[3,4],keepdim=True)
        fixed_cali=0 if not 'fixed' in self.keys else self.fixed_coeff[noise_ind].view(B,1,1,1,1)
        x = x-torch.mean(x,dim=[3,4],keepdim=True)+x_mean_clean#+fixed_cali
        # x = x+x_mean_clean
        x=x[:,:,2:-2,...]
        return torch.clip(x,0,1),denoised_inter,predicted_noise,inter_feat,mask


if __name__ == '__main__':

    window_size = 8
    height = 1000
    width = 1000
    device = torch.device('cuda:0')
    
    model = Stage_denoise3(n_channel_in=4,n_channel_out=4)
    model=model.to(device)
    # print(model)
    
    x = torch.randn((1, 4, 5,height, width)).to(device)
    noise_level={}
    keys=['shot','read','uniform']
    for key in keys:
        noise_level[key]=torch.zeros((1,1,1,1,1)).to(device)
    with torch.no_grad():
        x = model(x,)#noise_map=x[1]#(b,1,1,1,1)
    print(x.shape)
