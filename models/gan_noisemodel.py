import torch.nn as nn
import torch
import numpy as np
from models.spectral_normalization import SpectralNorm
Tensor = torch.cuda.FloatTensor 
import scipy.io
import argparse, json, glob, os, sys
from models.unet import Unet
from models.dyconv import RouteFuncMLP,RouteFuncMLP2d

from pathlib import Path
_script_dir = Path( __file__ ).parent
_root_dir = _script_dir.parent


class NoiseGenerator2d_distributed_ablation(nn.Module):
    # 一组param+修正参数的二维形式
    def __init__(self, net, unet_opts = 'noUnet', device = 'cuda:0', noise_list = 'shot_read_row',res_learn=False,dynamic=True):
        super(NoiseGenerator2d_distributed_ablation, self).__init__()
        
        print('generator device', device)
        self.device = device
        self.dtype = torch.float32
        self.noise_list = noise_list
        self.keys=[]
        self.net = net
        self.unet_opts = unet_opts
        self.keep_track = True
        self.res_learn=res_learn#返回的是加噪声的结果还是每步加上去的噪声
        self.dynamic=dynamic
        self.all_noise = {}
        
        if 'shot' in noise_list:
            self.shot_noise = torch.nn.Parameter(torch.tensor(0.00002*10000, dtype = self.dtype, device = device), 
                                                 requires_grad = True)
        if 'read' in noise_list:     
            self.read_noise = torch.nn.Parameter(torch.tensor(0.000002*10000, dtype = self.dtype, device = device), 
                                                 requires_grad = True)
        if 'row1' in noise_list:         
            self.row_noise = torch.nn.Parameter(torch.tensor(0.000002*1000, dtype = self.dtype, device = device), 
                                                requires_grad = True)
        if 'rowt' in noise_list:
            self.row_noise_temp = torch.nn.Parameter(torch.tensor(0.000002*1000, dtype = self.dtype, device = device), 
                                                     requires_grad = True)
        if 'uniform' in noise_list:    
            self.uniform_noise = torch.nn.Parameter(torch.tensor(0.00001*10000, dtype = self.dtype, device = device), requires_grad = True)
        if 'fixed1' in noise_list:
            mean_noise = scipy.io.loadmat(str(_root_dir) + '/data/fixed_pattern_noise.mat')['mean_pattern']
            fixed_noise = mean_noise.astype('float32')/2**16
            self.fixednoiset = torch.tensor(fixed_noise.transpose(2,0,1), dtype = self.dtype, device = device).unsqueeze(0)
        if 'learnedfixed' in noise_list:
            print('using learned fixed noise')
            self.fixed_dir='/data/fixed_pattern_noise_crvd.mat'
            if os.path.exists(str(_root_dir) + self.fixed_dir):
                mean_noise = scipy.io.loadmat(str(_root_dir) + self.fixed_dir)['mean_pattern']
                fixed_noise = mean_noise.astype('float32')/2**16#(h,w,c)->(c,h,w)->(1,1,c,h,w)
                fixednoiset = torch.tensor(fixed_noise.transpose(2,0,1), dtype = self.dtype, device = device).unsqueeze(0)#(1,c,h,w)
            else:
                C,H,W=(4,1080,1920)
                fixednoiset=torch.zeros(C,H,W).unsqueeze(0)#(1,c,h,w)

            self.fixednoiset = torch.nn.Parameter(fixednoiset, requires_grad = True)

        if self.dynamic:
            self.get_keys()
            self.dynamic_param=RouteFuncMLP2d(c_in=2,ratio=1,out_channels=6,kernels=[3,3])

        
        
    def save_fixed(self):#(c,h,w)->(h,w,c)
        fixed=self.fixednoiset[0].permute(1,2,0).cpu().detach().numpy()
        scipy.io.savemat(str(_root_dir) + self.fixed_dir, {'mean_pattern':fixed}) 

        
    def get_keys(self):#获得当前合成模型的噪声类型
        self.keys=[]
        if 'shot' in self.noise_list and 'read' in self.noise_list:
            self.shot_noise.requires_grad=False
            self.read_noise.requires_grad=False
            self.keys.append('shot_read')
        elif 'shot' not in self.noise_list and 'read' in self.noise_list:
            self.read_noise.requires_grad=False
            self.keys.append('read')
        if 'uniform' in self.noise_list:
            self.uniform_noise.requires_grad=False
            self.keys.append('uniform')
        if 'row1' in self.noise_list:
            self.row_noise.requires_grad=False
            self.keys.append('row')
        if 'rowt' in self.noise_list:
            self.row_noise_temp.requires_grad=False
            self.keys.append('rowt')
        if 'fixed1' in self.noise_list or 'learnedfixed' in self.noise_list:
            self.keys.append('fixed')
        if 'periodic' in self.noise_list:
            self.keys.append('periodic')
        return self.keys


    def forward(self, x, split_into_patches = False, pos=None, noise_level=None):
        # i0是裁剪图片的位置，和fix噪声的内容有关，不允许输入为空
        assert pos is not None

        B,C,H,W=x.shape#gen直接二维输入

        if self.dynamic:
            noise_level_a = noise_level[0].view(-1,1,1,1).expand((B, 1, H, W)).cuda()
            noise_level_b = noise_level[1].view(-1,1,1,1).expand((B, 1, H, W)).cuda()
            dynamic_input=torch.cat([noise_level_a,noise_level_b],dim=1)
            params=self.dynamic_param(dynamic_input)#(b,c,1,1)
            # print(params[9].shape,params[9])
    

        if self.unet_opts == 'Unet_first':
            x  = self.net(x)
        
        noise = torch.zeros_like(x)
        if 'shot' in self.noise_list and 'read' in self.noise_list:
            if self.dynamic:
                # 似乎对于动态的情况就不需要再用固定的可学习噪声参数了
                # shot torch.Size([1, 4, 16, 1, 1])
                variance = params[:,0,...].unsqueeze(1)*x*self.shot_noise + params[:,1,...].unsqueeze(1)*self.read_noise
            else:
                variance = x*self.shot_noise + self.read_noise
            shot_noise = torch.randn(x.shape, requires_grad= True, device = self.device)*variance
            noise += shot_noise
            if self.keep_track == True:
                if self.res_learn:
                    self.all_noise['shot_read'] = shot_noise
                else:
                    self.all_noise['shot_read'] = noise+x

        elif 'read' in self.noise_list and 'shot' not in self.noise_list:
            variance =self.read_noise
            if self.dynamic:
                variance = params[:,1,...].unsqueeze(1)*variance
            else:
                variance = variance
            read_noise = torch.randn(x.shape, requires_grad= True, device = self.device)*variance
            noise+=read_noise
            if self.keep_track == True:
                if self.res_learn:
                    self.all_noise['read'] = read_noise
                else:
                    self.all_noise['read'] = noise+x

        if 'uniform' in self.noise_list:    
            uniform_noise = self.uniform_noise*torch.rand(x.shape, requires_grad= True, device = self.device)
            if self.dynamic:
                uniform_noise=params[:,2,...].unsqueeze(1)*uniform_noise
            else:
                uniform_noise=uniform_noise
            noise += uniform_noise
            if self.keep_track == True:
                if self.res_learn:
                    self.all_noise['uniform'] = uniform_noise
                else:
                    self.all_noise['uniform'] = noise+x

        if 'row1' in self.noise_list: #(b,c,1,w)
            row_noise = self.row_noise*torch.randn([*x.shape[0:-2],x.shape[-1]],requires_grad= True, device = self.device).unsqueeze(-2)
            if self.dynamic:
                row_noise=params[:,3,...].unsqueeze(1)*row_noise
            else:
                row_noise=row_noise
            noise += row_noise
            if self.keep_track == True:
                if self.res_learn:
                    self.all_noise['row'] = row_noise.repeat(1,1,H,1)
                else:
                    self.all_noise['row'] = noise+x

        if 'rowt' in self.noise_list:#(b*t,1,1,w)
            row_noise_temp = self.row_noise_temp*torch.randn([*x.shape[0:-3],x.shape[-1]],requires_grad= True, device = self.device).unsqueeze(-2).unsqueeze(-2)
            if self.dynamic:
                row_noise_temp=params[:,4,...].unsqueeze(1)*row_noise_temp
            else:
                row_noise_temp=row_noise_temp
            noise += row_noise_temp
            if self.keep_track == True:
                if self.res_learn:
                    self.all_noise['rowt'] = row_noise_temp.repeat(1,C,H,1)
                else:
                    self.all_noise['rowt']=noise+x

        if 'fixed1' in self.noise_list or 'learnedfixed' in self.noise_list:
            pos_h = pos[0].view(-1)#(b*t)
            pos_w = pos[1].view(-1)#(b*t)
            #(1,c,h,w)->(1,1,c,pts,pts)
            fixed_noise=[]
            for i in range(B):#这个有没有串行写法啊#(1,c,h,w)->(1,c,pts,pts)
                fixed_noise.append(self.fixednoiset[...,pos_h[i]:pos_h[i]+x.shape[-2], pos_w[i]:pos_w[i] + x.shape[-1]])
            fixed_noise=torch.cat(fixed_noise,dim=0)#(1,c,pts,pts)->(b,c,pts,pts)

            if self.dynamic:
                # 现在是对不同的T都会有不同的fixed noise，需要改
                fixed_noise=params[:,5,...].unsqueeze(1)*fixed_noise
            else:
                fixed_noise=fixed_noise
            
            noise = noise+fixed_noise
            
            if self.keep_track == True:
                if self.res_learn:
                    self.all_noise['fixed'] = fixed_noise
                else:
                    self.all_noise['fixed']=noise+x
            
        noisy = x + noise
        
        # if split_into_patches== True:
        #     noisy = split_into_patches2d(noisy)
        #     x = split_into_patches2d(x)
        
        if self.unet_opts == 'Unet':
            noisy  = self.net(noisy)
        elif self.unet_opts == 'Unet_cat':
            noisy  = self.net(torch.cat((x, noisy),1))
            
        noisy = torch.clip(noisy, 0, 1)

        return noisy,self.all_noise


class NoiseGenerator2d_v2(nn.Module):
    # 多组param+动态参数
    def __init__(self, net, unet_opts = 'noUnet', device = 'cuda:0', noise_list = 'shot_read_row',res_learn=False,dynamic=True,noise_levels=5):
        super(NoiseGenerator2d_v2, self).__init__()
        
        print('generator device', device)
        self.device = device
        self.dtype = torch.float32
        self.noise_list = noise_list
        self.keys=[]
        self.net = net
        self.unet_opts = unet_opts
        self.keep_track = True
        self.res_learn=res_learn#返回的是加噪声的结果还是每步加上去的噪声
        self.dynamic=dynamic
        self.all_noise = {}
        self.noise_levels=noise_levels
        
        # 最简单粗暴的策略：为每个噪声级别计算一个参数
        if 'shot' in noise_list:
            self.shot_noise = torch.nn.Parameter(torch.tensor(0.00002*10000, dtype = self.dtype, device = device).repeat(self.noise_levels), 
                                                 requires_grad = True)
        if 'read' in noise_list:     
            self.read_noise = torch.nn.Parameter(torch.tensor(0.000002*10000, dtype = self.dtype, device = device).repeat(self.noise_levels), 
                                                 requires_grad = True)
        if 'row1' in noise_list:         
            self.row_noise = torch.nn.Parameter(torch.tensor(0.000002*1000, dtype = self.dtype, device = device).repeat(self.noise_levels), 
                                                requires_grad = True)
        if 'rowt' in noise_list:
            self.row_noise_temp = torch.nn.Parameter(torch.tensor(0.000002*1000, dtype = self.dtype, device = device).repeat(self.noise_levels), 
                                                     requires_grad = True)
        if 'uniform' in noise_list:    
            self.uniform_noise = torch.nn.Parameter(torch.tensor(0.00001*10000, dtype = self.dtype, device = device).repeat(self.noise_levels)
                                                    , requires_grad = True)
        if 'fixed1' in noise_list:
            mean_noise = scipy.io.loadmat(str(_root_dir) + '/data/fixed_pattern_noise.mat')['mean_pattern']
            fixed_noise = mean_noise.astype('float32')/2**16
            self.fixednoiset = torch.tensor(fixed_noise.transpose(2,0,1), dtype = self.dtype, device = device).unsqueeze(0)
            self.fixed_coeff = torch.nn.Parameter(torch.tensor(1.0, dtype = self.dtype, device = device).repeat(self.noise_levels)
                                                    , requires_grad = True)
        if 'learnedfixed' in noise_list:
            print('using learned fixed noise')
            self.fixed_dir='/data/fixed_pattern_noise_crvd.mat'
            if os.path.exists(str(_root_dir) + self.fixed_dir):
                mean_noise = scipy.io.loadmat(str(_root_dir) + self.fixed_dir)['mean_pattern']
                fixed_noise = mean_noise.astype('float32')/2**16#(h,w,c)->(c,h,w)->(1,1,c,h,w)
                fixednoiset = torch.tensor(fixed_noise.transpose(2,0,1), dtype = self.dtype, device = device).unsqueeze(0)#(1,c,h,w)
            else:
                C,H,W=(4,1080,1920)
                fixednoiset=torch.zeros(C,H,W).unsqueeze(0)#(1,c,h,w)

            self.fixednoiset = torch.nn.Parameter(fixednoiset, requires_grad = True)
            self.fixed_coeff = torch.nn.Parameter(torch.tensor(1.0, dtype = self.dtype, device = device).repeat(self.noise_levels)
                                                    , requires_grad = True)

        if self.dynamic:
            self.get_keys()
            self.dynamic_param=RouteFuncMLP2d(c_in=1,ratio=1,out_channels=noise_levels,kernels=[3,3])

        
        
    def save_fixed(self):#(c,h,w)->(h,w,c)
        fixed=self.fixednoiset[0].permute(1,2,0).cpu().detach().numpy()
        scipy.io.savemat(str(_root_dir) + self.fixed_dir, {'mean_pattern':fixed}) 

        
    def get_keys(self):#获得当前合成模型的噪声类型
        self.keys=[]
        if 'shot' in self.noise_list and 'read' in self.noise_list:
            # self.shot_noise.requires_grad=False
            # self.read_noise.requires_grad=False
            self.keys.append('shot_read')
        elif 'shot' not in self.noise_list and 'read' in self.noise_list:
            # self.read_noise.requires_grad=False
            self.keys.append('read')
        if 'uniform' in self.noise_list:
            # self.uniform_noise.requires_grad=False
            self.keys.append('uniform')
        if 'row1' in self.noise_list:
            # self.row_noise.requires_grad=False
            self.keys.append('row')
        if 'rowt' in self.noise_list:
            # self.row_noise_temp.requires_grad=False
            self.keys.append('rowt')
        if 'fixed1' in self.noise_list or 'learnedfixed' in self.noise_list:
            self.keys.append('fixed')
        if 'periodic' in self.noise_list:
            self.keys.append('periodic')
        return self.keys

    def weight_params(self,noise,params):
        # noise:->(1,5,1,1);  params:(b,5,1,1)  --> (b,1,1,1)
        # noise=noise.view(1,-1,1,1)
        return torch.mean(noise.view(1,-1,1,1)*params,dim=1,keepdim=True)

    def forward(self, x, split_into_patches = False, pos=None, noise_level=None):
        # 这版的noise_level是一个整数（即params的ind）
        # i0是裁剪图片的位置，和fix噪声的内容有关，不允许输入为空
        assert pos is not None
        noise_ind=noise_level

        B,C,H,W=x.shape#gen直接二维输入

        if self.dynamic:
            # dynamic是训练的第二个阶段：用一个attention把四组权重组合起来
            noise_level_input = noise_level.view(-1,1,1,1).expand((B, 1, H, W)).cuda()
            dynamic_input=torch.cat([noise_level_input],dim=1)
            params=self.dynamic_param(dynamic_input)#(b,c,1,1)
            # print(params)
    

        if self.unet_opts == 'Unet_first':
            x  = self.net(x)
        
        noise = torch.zeros_like(x)
        if 'shot' in self.noise_list and 'read' in self.noise_list:
            if self.dynamic:
                # 似乎对于动态的情况就不需要再用固定的可学习噪声参数了
                # shot torch.Size([1, 4, 16, 1, 1])
                # (b,c,h,w)*(b,5,h,w)
                variance = x*self.weight_params(self.shot_noise,params) + self.weight_params(self.read_noise,params)
            else:
                variance_shot=self.shot_noise[noise_ind].view(B,1,1,1)
                variance_read=self.read_noise[noise_ind].view(B,1,1,1)
                # print('variance_shot:',variance_shot,'variance_read:',variance_read)
                variance = x*variance_shot + variance_read

            shot_noise = torch.randn(x.shape, requires_grad= True, device = self.device)*variance
            noise += shot_noise
            if self.keep_track == True:
                if self.res_learn:
                    self.all_noise['shot_read'] = shot_noise
                else:
                    self.all_noise['shot_read'] = noise+x

        elif 'read' in self.noise_list and 'shot' not in self.noise_list:
            if self.dynamic:
                variance=self.weight_params(self.read_noise,params)
            else:
                variance = self.read_noise[noise_ind].view(B,1,1,1)

            read_noise = torch.randn(x.shape, requires_grad= True, device = self.device)*variance
            noise+=read_noise
            if self.keep_track == True:
                if self.res_learn:
                    self.all_noise['read'] = read_noise
                else:
                    self.all_noise['read'] = noise+x

        if 'uniform' in self.noise_list:
            if self.dynamic:
                uniform_variance=self.weight_params(self.uniform_noise,params)
            else:
                uniform_variance=self.uniform_noise[noise_ind].view(B,1,1,1)
                # print('uniform_variance:',uniform_variance)

            uniform_noise=torch.rand(x.shape, requires_grad= True, device = self.device)*uniform_variance
            noise += uniform_noise
            if self.keep_track == True:
                if self.res_learn:
                    self.all_noise['uniform'] = uniform_noise
                else:
                    self.all_noise['uniform'] = noise+x

        if 'row1' in self.noise_list: #(b,c,1,w)
            if self.dynamic:
                row_variance=self.weight_params(self.row_noise,params)
            else:
                row_variance=self.row_noise[noise_ind].view(B,1,1,1)
                # print('row_variance:',row_variance)

            row_noise = row_variance*\
                torch.randn([*x.shape[0:-2],x.shape[-1]],requires_grad= True, device = self.device).unsqueeze(-2)
            noise += row_noise
            if self.keep_track == True:
                if self.res_learn:
                    self.all_noise['row'] = row_noise.repeat(1,1,H,1)
                else:
                    self.all_noise['row'] = noise+x

        if 'rowt' in self.noise_list:#(b*t,1,1,w)
            if self.dynamic:
                rowt_variance=self.weight_params(self.row_noise_temp,params)
            else:
                rowt_variance=self.row_noise_temp[noise_ind].view(B,1,1,1)
                # print('rowt_variance:',rowt_variance)

            row_noise_temp = rowt_variance*\
                torch.randn([*x.shape[0:-3],x.shape[-1]],requires_grad= True, device = self.device).unsqueeze(-2).unsqueeze(-2)
            noise += row_noise_temp
            if self.keep_track == True:
                if self.res_learn:
                    self.all_noise['rowt'] = row_noise_temp.repeat(1,C,H,1)
                else:
                    self.all_noise['rowt']=noise+x

        if 'fixed1' in self.noise_list or 'learnedfixed' in self.noise_list:
            pos_h = pos[0]#(b)
            pos_w = pos[1]#(b)
            #(1,c,h,w)->(1,1,c,pts,pts)
            if self.dynamic:
                # 现在是对不同的T都会有不同的fixed noise，需要改
                fixed_variance=self.weight_params(self.fixed_coeff,params)
            else:
                fixed_variance=self.fixed_coeff[noise_ind].view(B,1,1,1)
                # print('fixed_variance:',fixed_variance)
                
            fixed_noise=[]
            for i in range(B):#这个有没有并行写法啊#(1,c,h,w)->(1,c,pts,pts)
                fixed_noise.append(self.fixednoiset[...,pos_h[i]:pos_h[i]+x.shape[-2], pos_w[i]:pos_w[i] + x.shape[-1]])
            fixed_noise=torch.cat(fixed_noise,dim=0)#(1,c,pts,pts)->(b,c,pts,pts)
            fixed_noise=fixed_noise*fixed_variance
            noise+=fixed_noise
            
            if self.keep_track == True:
                if self.res_learn:
                    self.all_noise['fixed'] = fixed_noise
                else:
                    self.all_noise['fixed']=noise+x
            
        noisy = x + noise
        
        # if split_into_patches== True:
        #     noisy = split_into_patches2d(noisy)
        #     x = split_into_patches2d(x)
        
        if self.unet_opts == 'Unet':
            noisy  = self.net(noisy)
        elif self.unet_opts == 'Unet_cat':
            noisy  = self.net(torch.cat((x, noisy),1))
            
        noisy = torch.clip(noisy, 0, 1)

        return noisy,self.all_noise
    

class NoiseGenerator2d_v3(nn.Module):
    '''多组param+动态参数'''
    def __init__(self, net, unet_opts = 'noUnet', device = 'cuda:0', \
                 noise_list = 'shot_read_row',res_learn=False,dynamic=False,noise_levels=5,\
                    fixed_shape=(4,1080,1920)):
        super(NoiseGenerator2d_v3, self).__init__()
        
        print('generator device', device)
        self.device = device
        self.dtype = torch.float32
        self.noise_list = noise_list
        self.fixed_shape=fixed_shape
        self.keys=[]
        self.net = net
        self.unet_opts = unet_opts
        self.keep_track = True
        self.res_learn=res_learn#返回的是加噪声的结果还是每步加上去的噪声
        self.dynamic=dynamic
        self.all_noise = {}
        self.noise_levels=noise_levels
        self.fixed_pos=False#crvd用的是true，rnvd是false
        
        # 最简单粗暴的策略：为每个噪声级别计算一个参数
        if 'shot' in noise_list:
            self.shot_noise = torch.nn.Parameter(torch.tensor(0.000002*10000, dtype = self.dtype, device = device).repeat(self.noise_levels), 
                                                 requires_grad = True)
        if 'read' in noise_list:     
            # self.read_noise = torch.nn.Parameter(torch.tensor(0.000002*10000, dtype = self.dtype, device = device).repeat(self.noise_levels), 
            #                                      requires_grad = True)
            self.read_noise = torch.nn.Parameter(torch.tensor(0.0, dtype = self.dtype, device = device).repeat(self.noise_levels), 
                                                 requires_grad = True)
        if 'uniform' in noise_list:    
            # self.uniform_noise = torch.nn.Parameter(torch.tensor(0.00001*10000, dtype = self.dtype, device = device).repeat(self.noise_levels)
            #                                         , requires_grad = True)
            self.uniform_noise = torch.nn.Parameter(torch.tensor(0.0, dtype = self.dtype, device = device).repeat(self.noise_levels)
                                                    , requires_grad = True)
        if 'fixed' in noise_list:
            if self.fixed_pos:#每个pos都有对应的fix噪声
                print('using learned fixed noise')
                self.fixed_dir='/data/fixed_pattern_noise_rnvd.mat'
                if os.path.exists(str(_root_dir) + self.fixed_dir):
                    mean_noise = scipy.io.loadmat(str(_root_dir) + self.fixed_dir)['mean_pattern']
                    fixed_noise = mean_noise.astype('float32')/2**16#(h,w,c)->(c,h,w)->(1,1,c,h,w)
                    fixednoiset = torch.tensor(fixed_noise.transpose(2,0,1), dtype = self.dtype, device = device).unsqueeze(0)#(1,c,h,w)
                else:
                    C,H,W=self.fixed_shape
                    fixednoiset=torch.zeros(C,H,W).unsqueeze(0)#(1,c,h,w)
                init_fixed_coeff=1.0

                self.fixednoiset = torch.nn.Parameter(fixednoiset, requires_grad = True)
            else:
                init_fixed_coeff=0.0

            self.fixed_coeff = torch.nn.Parameter(torch.tensor(init_fixed_coeff, dtype = self.dtype, device = device).repeat(self.noise_levels)
                                                    , requires_grad = True)

        if self.dynamic:
            self.get_keys()
            self.dynamic_param=RouteFuncMLP2d(c_in=1,ratio=1,out_channels=noise_levels,kernels=[3,3])

        
    def save_fixed(self):#(c,h,w)->(h,w,c)
        fixed=self.fixednoiset[0].permute(1,2,0).cpu().detach().numpy()
        scipy.io.savemat(str(_root_dir) + self.fixed_dir, {'mean_pattern':fixed}) 

        
    def get_keys(self):#获得当前合成模型的噪声类型
        self.keys=[]
        if 'shot' in self.noise_list:
            # self.shot_noise.requires_grad=False
            # self.read_noise.requires_grad=False
            self.keys.append('shot')
        if 'read' in self.noise_list:
            # self.read_noise.requires_grad=False
            self.keys.append('read')
        if 'uniform' in self.noise_list:
            # self.uniform_noise.requires_grad=False
            self.keys.append('uniform')
        if 'fixed' in self.noise_list:
            self.keys.append('fixed')
        return self.keys

    def weight_params(self,noise,params):
        # noise:->(1,5,1,1);  params:(b,5,1,1)  --> (b,1,1,1)
        # noise=noise.view(1,-1,1,1)
        return torch.mean(noise.view(1,-1,1,1)*params,dim=1,keepdim=True)
    
    def get_noise_level(self,noise_ind):
        '''B,1,1,1,1，这里的B是对seq而言的'''
        B=noise_ind.shape[0]
        return {
            'shot':self.shot_noise[noise_ind].view(B,1,1,1,1),
            'read':self.read_noise[noise_ind].view(B,1,1,1,1),
            'uniform':self.uniform_noise[noise_ind].view(B,1,1,1,1),
            'fixed':self.fixed_coeff[noise_ind].view(B,1,1,1,1)
        }
        

    def forward(self, x, split_into_patches = False, pos=None, noise_level=None):
        # 这版的noise_level是一个整数（即params的ind）
        # i0是裁剪图片的位置，和fix噪声的内容有关，不允许输入为空
        assert pos is not None
        noise_ind=noise_level#(b,t)

        B,C,H,W=x.shape#gen直接二维输入

        if self.dynamic:
            # dynamic是训练的第二个阶段：用一个attention把四组权重组合起来
            noise_level_input = noise_level.view(-1,1,1,1).expand((B, 1, H, W)).cuda()
            dynamic_input=torch.cat([noise_level_input],dim=1)
            params=self.dynamic_param(dynamic_input)#(b,c,1,1)
            # print(params)

        if self.unet_opts == 'Unet_first':
            x  = self.net(x)

        noise = torch.zeros_like(x)
        if 'shot' in self.noise_list:
            shot_variance = self.weight_params(self.shot_noise,params) if self.dynamic \
                else self.shot_noise[noise_ind].view(B,1,1,1)#shot不能是负数，否则梯度会变大
            shot_noise = torch.poisson(x/torch.abs(shot_variance))*torch.abs(shot_variance)-x#这个possion是在input基础上加的，所以减去x才是最后的噪声
            # shot_noise = torch.randn(x.shape, requires_grad= True, device = self.device)*shot_variance
            noise += shot_noise
            if self.keep_track == True:
                self.all_noise['shot'] = noise if self.res_learn else noise+x

        if 'read' in self.noise_list:
            read_variance=self.weight_params(self.read_noise,params) if self.dynamic \
                else self.read_noise[noise_ind].view(B,1,1,1)
            read_noise = torch.randn(x.shape, requires_grad= True, device = self.device)*read_variance
            noise+=read_noise
            if self.keep_track == True:
                self.all_noise['read'] = noise if self.res_learn else noise+x

        if 'uniform' in self.noise_list:
            uniform_variance=self.weight_params(self.uniform_noise,params) if self.dynamic \
                else self.uniform_noise[noise_ind].view(B,1,1,1)
            uniform_noise=torch.rand(x.shape, requires_grad= True, device = self.device)*uniform_variance
            noise += uniform_noise
            if self.keep_track == True:
                self.all_noise['uniform'] = noise if self.res_learn else noise+x

        if 'fixed' in self.noise_list:
            pos_h = pos[0].view(-1)#(b)
            pos_w = pos[1].view(-1)#(b)
            #(1,c,h,w)->(1,1,c,pts,pts)
            # 现在是对不同的T都会有不同的fixed noise，需要改
            fixed_variance=self.weight_params(self.fixed_coeff,params) if self.dynamic \
                else self.fixed_coeff[noise_ind].view(B,1,1,1)
            
            if self.fixed_pos:#每个pos都有对应的fix噪声
                fixed_noise=[]
                for i in range(B):#这个有没有并行写法啊#(1,c,h,w)->(1,c,pts,pts)
                    fixed_noise.append(self.fixednoiset[...,pos_h[i]:pos_h[i]+x.shape[-2], pos_w[i]:pos_w[i] + x.shape[-1]])
                fixed_noise=torch.cat(fixed_noise,dim=0)#(1,c,pts,pts)->(b,c,pts,pts)
                fixed_noise=fixed_noise*fixed_variance
            else:
                fixed_noise=fixed_variance
            
            noise+=fixed_noise
            
            if self.keep_track == True:
                self.all_noise['fixed'] = noise if self.res_learn else noise+x
            
        noisy = x + noise
        
        if self.unet_opts == 'Unet':
            noisy  = self.net(noisy)
        elif self.unet_opts == 'Unet_cat':
            noisy  = self.net(torch.cat((x, noisy),1))
            
        noisy = torch.clip(noisy, 0, 1)

        return noisy,self.all_noise

    
channels = 4
leak = 0.1
w_g = 8
   

class DiscriminatorS2d_sig(nn.Module):
    def __init__(self, channels = 4):
        super(DiscriminatorS2d_sig, self).__init__()
        self.conv1 = SpectralNorm(nn.Conv2d(channels, 64, 3, stride=1, padding=(1,1)))
        self.conv2 = SpectralNorm(nn.Conv2d(64, 128, 4, stride=2, padding=(1,1)))
        self.conv3 = SpectralNorm(nn.Conv2d(128, 256, 4, stride=2, padding=(1,1)))
        self.conv4 = SpectralNorm(nn.Conv2d(256, 512, 4, stride=2, padding=(1,1)))
        self.conv5 = SpectralNorm(nn.Conv2d(512, 512*2, 3, stride=2, padding=(1,1)))

        self.classifier = nn.Sequential(
            nn.Sigmoid()
        )
        self.fc = SpectralNorm(nn.Linear(1024*4*4, 1))

    def forward(self, x):
        m=x
        m = nn.LeakyReLU(leak)(self.conv1(m))
        m = nn.LeakyReLU(leak)(self.conv2(m))
        m = nn.LeakyReLU(leak)(self.conv3(m))
        m = nn.LeakyReLU(leak)(self.conv4(m))
        m = nn.LeakyReLU(leak)(self.conv5(m))
        out = m.view(m.shape[0],-1)
        out = self.fc(out)
        out = self.classifier(out)
        #out = self.fc(m.view(-1,w_g * w_g * 512*2)).view(-1)
        #print('out', out.shape)
        return out