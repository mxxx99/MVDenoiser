import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io
from models.flornn_utils.components import ResBlocks, D
import argparse, json, glob, os, sys
from models.pytorch_pwc.pwc import PWCNet
from models.pytorch_pwc.extract_flow import extract_flow_torch
from models.pytorch_pwc.utils import warp,demosaic
from models.Birnn import Birnn

from pathlib import Path
_script_dir = Path( __file__ ).parent
_root_dir = _script_dir.parent


class Stage_denoise3(nn.Module):
    # 针对每个噪声类型用不同的网络去噪
    def __init__(self, n_channel_in=1, device="cuda:0", residual=False, down='conv', up='tconv', 
                 activation='selu',keys=[],res_learn=False,noise_cat=True,base_model='Swin',t_length=5):
        super(Stage_denoise3, self).__init__()
        self.keys=keys
        self.denoised_inter={}
        self.predicted_noise={}#每个去噪阶段去除的噪声
        self.device = device
        self.dtype=torch.float32
        self.noise_cat=noise_cat
        self.pwcnet = PWCNet()
        
        num_resblocks=4
        num_channels=64
        self.in_chans=n_channel_in
        self.num_channels=num_channels
    
        if 'fixed' in self.keys:#(b,c,t,h,w)->(b,c,1,h,w)
            self.basemodel_fixed = ResBlocks(input_channels=n_channel_in, num_resblocks=num_resblocks, num_channels=num_channels)
            self.fixed_pool=nn.AdaptiveAvgPool3d((1,1,1))#(b,c,1,1,1)
            print('using learned fixed noise from noise generator')
            self.fixed_dir='/data/fixed_pattern_noise_crvd.mat'
            if os.path.exists(str(_root_dir) + self.fixed_dir):
                mean_noise = scipy.io.loadmat(str(_root_dir) + self.fixed_dir)['mean_pattern']
                fixed_noise = mean_noise.astype('float32')/2**16#(h,w,c)->(c,h,w)->(c,1,h,w)
                fixednoiset = torch.tensor(fixed_noise.transpose(2,0,1), dtype = self.dtype, device = self.device).unsqueeze(1)#(c,1,h,w)
                self.fixednoiset=torch.nn.Parameter(fixednoiset, requires_grad = True)
            else:
                C,H,W=(4,1080,1920)
                fixednoiset=torch.zeros(C,H,W).unsqueeze(1)#(c,1,h,w)
                self.fixednoiset = torch.nn.Parameter(fixednoiset, requires_grad = True)

        for idx,key in enumerate(self.keys):
            noisecat=1 if self.noise_cat else 0
            c_in=n_channel_in+noisecat+num_channels if idx==0 else n_channel_in+noisecat
            c_out=num_channels if idx==len(self.keys)-1 else n_channel_in
            setattr(self,'basemodel_%s_forward'%key,ResBlocks(input_channels=c_in, out_channels=c_out, num_resblocks=num_resblocks, num_channels=num_channels))
            setattr(self,'basemodel_%s_backward'%key,ResBlocks(input_channels=c_in, out_channels=c_out,num_resblocks=num_resblocks, num_channels=num_channels))
            if idx!=0:
                setattr(self,'d_%s'%key,D(in_channels=n_channel_in*2, mid_channels=num_channels, out_channels=n_channel_in))
        setattr(self,'d',D(in_channels=num_channels*2, mid_channels=num_channels, out_channels=n_channel_in))

    def trainable_parameters(self):
        params=[]
        for idx,key in enumerate(self.keys):
            params.append({'params':getattr(self,'basemodel_%s_forward'%key).parameters()})
            params.append({'params':getattr(self,'basemodel_%s_backward'%key).parameters()})
            if idx!=0:
                params.append({'params':getattr(self,'d_%s'%key).parameters()})
            params.append({'params':getattr(self,'d').parameters()})
        return params
        # return [{'params':self.forward_rnn.parameters()}, {'params':self.backward_rnn.parameters()}, {'params':self.d.parameters()}]

    def denoise_stage(self,x_input,key,noise_level=None,type='forward'):
        B,C,T,H,W=x_input.shape
        x_input=x_input.transpose(1,2).view(-1,C,H,W)
        noisemap=noise_level[key].expand(B,1,H,W)
        x_out=eval('self.basemodel_%s_%s'%(key,type))\
            (torch.cat([x_input,noisemap],dim=1) if self.noise_cat else x_input)
        # x_out:去噪后的结果；denoised_inter:去噪前的结果；predicted_noise:key阶段分离出的噪声
        return x_out.view(B,T,C,H,W).transpose(1,2)
    
    def denoise_all(self,x_input,noise_level=None,type='forward',t_ind=0):
        x_out=x_input
        B,C,H,W=x_input.shape
        for idx,key in enumerate(self.keys):
            (self.denoised_inter[key])[:,:,t_ind]=x_out[:,:4]
            if type=='forward' and idx!=0:
                (self.denoised_inter[key])[:,:,t_ind]=((self.denoised_inter[key])[:,:,t_ind]+x_out)/2
            noisemap=(noise_level[key])[:,:,0].expand(B,1,H,W)
            x_out=eval('self.basemodel_%s_%s'%(key,type))\
                        (torch.cat([x_out,noisemap],dim=1) if self.noise_cat else x_input)
            if type=='forward' and idx!=len(self.keys)-1:
                (self.predicted_noise[key])[:,:,t_ind]=(self.denoised_inter[key])[:,:,t_ind]-x_out
        return x_out
    
    def forward(self, seqn, pos=None, noise_level=None):
        seqn_mean=torch.mean(seqn,dim=[2,3,4],keepdim=True)#每个seq每个channel算一个均值
        seqn=seqn-seqn_mean# 这里原来rgb的图还有一个减去均值的操作

        # noise_level的每种噪声都是(B,1,1,1,1)
        seqn=seqn.transpose(1,2)
        B,T,C,H,W=seqn.shape
        forward_hs = torch.empty((B, T, self.num_channels, H, W), device=seqn.device)
        backward_hs = torch.empty((B, T, self.num_channels, H, W), device=seqn.device)
        self.denoised_inter={}
        for key in self.keys:#去除key噪声前的结果
            self.denoised_inter[key]=torch.empty((B,self.in_chans,T,H,W),device=seqn.device)
            self.predicted_noise[key]=torch.empty((B,self.in_chans,T,H,W),device=seqn.device)
        seqdn = torch.empty_like(seqn)
        srgb_seqn = demosaic(seqn)

        # backward features
        init_backward_h = torch.zeros((B, self.num_channels, H, W), device=seqn.device)
        backward_h = self.denoise_all(torch.cat((seqn[:,-1],init_backward_h),dim=1),noise_level,'backward',0)
        backward_hs[:, -1] = backward_h
        for i in range(2, T+1):
            flow = extract_flow_torch(self.pwcnet, srgb_seqn[:, T-i], srgb_seqn[:, T-i+1])
            aligned_backward_h,_=warp(backward_h,flow)#T-i+1向T-i对齐
            # aligned_backward_h=backward_h
            backward_h = self.denoise_all(torch.cat((seqn[:,T-i],aligned_backward_h),dim=1),noise_level,'backward',T-i)
            backward_hs[:,T-i] = backward_h

        # forward features
        # and generate final result
        init_forward_h = torch.zeros((B, self.num_channels, H, W), device=seqn.device)
        forward_h = self.denoise_all(torch.cat((seqn[:,0],init_forward_h),dim=1),noise_level,'forward',0)
        forward_hs[:, 0] = forward_h
        seqdn[:,0]=eval('self.d')(torch.cat((forward_hs[:, 0], backward_hs[:, 0]), dim=1))

        for i in range(1, T):
            flow = extract_flow_torch(self.pwcnet, srgb_seqn[:, i], srgb_seqn[:, i-1])
            aligned_forward_h,_=warp(forward_h,flow)#i-1向i对齐
            # aligned_forward_h=forward_h
            forward_h = self.denoise_all(torch.cat((seqn[:,i],aligned_forward_h),dim=1),noise_level,'forward',i)
            forward_hs[:, i] = forward_h
            
            # get results
            seqdn[:,i]=eval('self.d')(torch.cat((forward_hs[:, i], backward_hs[:, i]), dim=1))
            self.predicted_noise[self.keys[-1]][:,:,i]=self.denoised_inter[self.keys[-1]][:,:,i]-seqdn[:,i]
        
        for key in self.keys:#去除key噪声前的结果
            self.denoised_inter[key]=self.denoised_inter[key]+seqn_mean

        return torch.clip(seqdn.transpose(1,2)+seqn_mean,0,1)[:,:,1:-1],self.denoised_inter,self.predicted_noise


class Stage_denoise32(nn.Module):
    # 针对每个噪声类型用不同的网络去噪
    def __init__(self, n_channel_in=1, device="cuda:0", residual=False, down='conv', up='tconv', 
                 activation='selu',keys=[],res_learn=False,noise_cat=True,base_model='Swin',t_length=5):
        super(Stage_denoise32, self).__init__()
        self.keys=keys
        self.denoised_inter={}
        self.predicted_noise={}#每个去噪阶段去除的噪声
        self.device = device
        self.dtype=torch.float32
        self.noise_cat=noise_cat
        self.pwcnet = PWCNet()
        
        num_channels=64
        self.in_chans=n_channel_in
        self.num_channels=num_channels
    

        for idx,key in enumerate(self.keys):
            num_blocks=5 if idx==len(self.keys)-1 else 3
            setattr(self,'basemodel_%s'%key,Birnn(n_channel_in=n_channel_in, device=device,num_channels=num_channels,num_resblocks=num_blocks))

    def trainable_parameters(self):
        params=[]
        for idx,key in enumerate(self.keys):
            params.append({'params':getattr(self,'basemodel_%s'%key).parameters(),'initial_lr':0.0002})
        return params

    # def denoise_stage(self,x_input,key,noise_level=None,type='forward'):
    #     B,C,T,H,W=x_input.shape
    #     x_input=x_input.transpose(1,2).view(-1,C,H,W)
    #     noisemap=noise_level[key].expand(B,1,H,W)
    #     x_out=eval('self.basemodel_%s_%s'%(key,type))\
    #         (torch.cat([x_input,noisemap],dim=1) if self.noise_cat else x_input)
    #     # x_out:去噪后的结果；denoised_inter:去噪前的结果；predicted_noise:key阶段分离出的噪声
    #     return x_out.view(B,T,C,H,W).transpose(1,2)
    
    def forward(self, seqn, pos=None, noise_level=None):
        seqn_mean=torch.mean(seqn,dim=[2,3,4],keepdim=True)#每个seq每个channel算一个均值
        seqn=seqn-seqn_mean# 这里原来rgb的图还有一个减去均值的操作

        # noise_level的每种噪声都是(B,1,1,1,1)
        seqn=seqn.transpose(1,2)
        B,T,C,H,W=seqn.shape
        denoised_inter={}
        predicted_noise={}

        srgb_seqn = demosaic(seqn)
        flow = {}
        flow['backward']=torch.empty((B, T, 2, H, W), device=seqn.device)
        flow['forward']=torch.empty((B, T, 2, H, W), device=seqn.device)
        for i in range(1,T):
            flow['forward'][:,i] = extract_flow_torch(self.pwcnet, srgb_seqn[:, i], srgb_seqn[:, i-1])
        for i in range(2, T+1):
            flow['backward'][:,T-i] = extract_flow_torch(self.pwcnet, srgb_seqn[:, T-i], srgb_seqn[:, T-i+1])
                # row和rowt的unet对应的之后的
                
        seqdn=seqn
        if 'uniform' in self.keys:
            seqdn,denoised_inter['uniform'],predicted_noise['uniform']=\
                getattr(self,'basemodel_uniform')(seqdn,flow,noise_level['uniform'])
            denoised_inter['uniform']+=seqn_mean
        
        if 'read' in self.keys:
            seqdn,denoised_inter['read'],predicted_noise['read']=\
                getattr(self,'basemodel_read')(seqdn,flow,noise_level['read'])
            denoised_inter['read']+=seqn_mean
            
        if 'shot' in self.keys:#generator生成的噪声是用一个随机分布乘一个可学习的参数组成的
            seqdn,denoised_inter['shot'],predicted_noise['shot']=\
                getattr(self,'basemodel_shot')(seqdn,flow,noise_level['shot'])
            denoised_inter['shot']+=seqn_mean

        seqdn=(seqdn.transpose(1,2)+seqn_mean)[:,:,1:-1]
        return torch.clip(seqdn,0,1),denoised_inter,predicted_noise


if __name__ == '__main__':

    window_size = 8
    height = 1000
    width = 1000
    device = torch.device('cuda:0')
    keys=['shot','read','uniform']
    B=1
    noise_level={
            'shot':torch.zeros((B,1,1,1,1),device=device),
            'read':torch.zeros((B,1,1,1,1),device=device),
            'uniform':torch.zeros((B,1,1,1,1),device=device),
            'fixed':torch.zeros((B,1,1,1,1),device=device),
        }
    
    model = Stage_denoise3(n_channel_in=4,keys=keys[::-1])
    model=model.to(device)
    # print(model)
    
    x = torch.randn((1, 4, 5,height, width)).to(device)
    with torch.no_grad():
        x,_,_ = model(x,noise_level=noise_level)
    print(x.shape)
