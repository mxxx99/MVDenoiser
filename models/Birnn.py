import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io
from models.flornn_utils.components import ResBlocks, D
import argparse, json, glob, os, sys
from models.pytorch_pwc.pwc import PWCNet
from models.pytorch_pwc.extract_flow import extract_flow_torch
from models.pytorch_pwc.utils import warp,demosaic


class Birnn(nn.Module):
    # 针对每个噪声类型用不同的网络去噪
    def __init__(self, n_channel_in=1, device="cuda:0",num_channels=64,num_resblocks=3):
        super(Birnn, self).__init__()

        self.device = device
        self.in_chans=n_channel_in+num_channels+1
        self.num_channels=num_channels

        self.basemodel_forward=ResBlocks(input_channels=self.in_chans, out_channels=num_channels,
                                         num_resblocks=num_resblocks, num_channels=num_channels)
        self.basemodel_backward=ResBlocks(input_channels=self.in_chans, out_channels=num_channels,
                                         num_resblocks=num_resblocks, num_channels=num_channels)
        self.d=D(in_channels=num_channels*2, mid_channels=num_channels, out_channels=n_channel_in)


    def forward(self, seqn, flow=None, noise_level=None):
        B,T,C,H,W=seqn.shape
        denoised_inter=seqn.clone()

        # noise_level的每种噪声都是(B,1,1,1,1)
        forward_hs = torch.empty((B, T, self.num_channels, H, W), device=seqn.device)
        backward_hs = torch.empty((B, T, self.num_channels, H, W), device=seqn.device)
        seqdn = torch.empty_like(seqn)
        noisemap=(noise_level)[:,:,0].expand(B,1,H,W)

        # backward features
        init_backward_h = torch.zeros((B, self.num_channels, H, W), device=seqn.device)
        backward_h = self.basemodel_backward(
                                torch.cat((seqn[:,-1],init_backward_h,noisemap),dim=1))
        backward_hs[:, -1] = backward_h
        for i in range(2, T+1):
            aligned_backward_h,_=warp(backward_h,flow['backward'][:,T-i])#T-i+1向T-i对齐
            backward_h = self.basemodel_backward(
                                    torch.cat((seqn[:,T-i],aligned_backward_h, noisemap),dim=1))
            backward_hs[:,T-i] = backward_h

        # forward features
        # and generate final result
        init_forward_h = torch.zeros((B, self.num_channels, H, W), device=seqn.device)
        forward_h = self.basemodel_forward(
                                    torch.cat((seqn[:,0],init_forward_h, noisemap),dim=1))
        forward_hs[:, 0] = forward_h
        seqdn[:,0]=self.d(torch.cat((forward_hs[:, 0], backward_hs[:, 0]), dim=1))

        for i in range(1, T):
            aligned_forward_h,_=warp(forward_h,flow['forward'][:,i])#i-1向i对齐
            # aligned_forward_h=forward_h
            forward_h = self.basemodel_forward(
                                    torch.cat((seqn[:,i],aligned_forward_h, noisemap),dim=1))
            forward_hs[:, i] = forward_h
            
            # get results
            seqdn[:,i]=self.d(torch.cat((forward_hs[:, i], backward_hs[:, i]), dim=1))
        predicted_noise=denoised_inter-seqdn

        return seqdn,denoised_inter.transpose(1,2),predicted_noise.transpose(1,2)