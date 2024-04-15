import torch
import lpips
from torch.nn import MSELoss, L1Loss
import numpy as np
import torch.nn as nn


def tv_loss(x, beta = 0.5):
    dh = torch.pow(x[...,1:] - x[...,:-1], 2)
    dw = torch.pow(x[...,1:,:] - x[...,:-1,:], 2)
    dt = torch.pow(x[...,1:,:,:] - x[...,:-1,:,:], 2)
    return torch.sum(dh[..., :-1, :-1,:] + dw[..., :-1, :, :-1] + dt[...,:-1,:-1] )


class KLD_loss(nn.Module):
    def __init__(self, nbins = 1000, xrange = (0,1)):
        super(KLD_loss, self).__init__()
        self.nbins = nbins
        self.xrange = xrange

    def forward(self, x1, x2):
        sz = np.prod(list(x1.shape))
        p = torch.histc(x1, bins = self.nbins, min =self.xrange[0], max = self.xrange[1])/sz
        q = torch.histc(x2, bins = self.nbins, min =self.xrange[0], max = self.xrange[1])/sz
        idx = (p > 0) & (q > 0)
        p = p[idx]
        q = q[idx]
        logp = torch.log(p)
        logq = torch.log(q)
        kl_fwd = torch.sum(p * (logp - logq))
        kl_inv = torch.sum(q * (logq - logp))
        kl_sym = (kl_fwd + kl_inv) / 2.0
        return kl_sym


# Charbonnier Loss
class CharbonnierLoss(nn.Module):
    def __init__(self,epsilon=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.epsilon2=epsilon*epsilon

    def forward(self,x1,x2):
        diff=torch.add(x1,-x2)
        value=torch.sqrt(torch.pow(diff,2)+self.epsilon2)
        return torch.mean(value)


def define_loss(args, gpu):
    print('using lpips loss')
    loss_fn_alex = lpips.LPIPS(net='alex').to(gpu)
    def gen_loss(in1, in2): 
        
        total_loss = 0
        if in1.shape[1]==8:
            total_loss+=torch.mean(loss_fn_alex(in1[:,0:3],
                                                in2[:,0:3],0,1))
            total_loss+=torch.mean(loss_fn_alex(in1[:,4:7],
                                                in2[:,4:7],0,1))
        else:
            total_loss+=torch.mean(loss_fn_alex(in1[:,0:3],
                                                in2[:,0:3],0,1))
        
        return total_loss
        
    return gen_loss, loss_fn_alex

def define_loss_denoiser(args, gpu):
    all_losses = []
    if 'MSE' in args.loss:
        print('using MSE loss')
        all_losses.append(MSELoss().cuda(gpu))
    if 'L1' in args.loss:
        print('using L1 loss')
        all_losses.append(L1Loss().cuda(gpu))
    if 'TV' in args.loss:
        print('using TV loss')
        loss_tv = lambda a,b: 1e-6*tv_loss(a)
        all_losses.append(loss_tv.cuda(gpu))
    if 'Charbonnier' in args.loss:
        print('using Charbonnier loss')
        all_losses.append(CharbonnierLoss().cuda(gpu))
    # if 'LP2D' in args.loss:
    #     print('using LPIPS loss')
    #     import lpips
    #     # loss_lpips1 = lpips.LPIPS(net='alex').cuda(gpu)
    #     # if 'ccm' in args.space:
    #     #     loss_lpips = lambda a,b: torch.sum(1e-1*loss_lpips1(ccm(a),ccm(b)))
    #     # else:
    #     loss_lpips = lambda a,b: torch.sum(1e-1*loss_lpips1(a[:,0:3],b[:,0:3]))
    #     all_losses.append(loss_lpips)
    # if 'LPIPS' in args.loss:
    #     print('using LPIPS loss')
    #     import lpips
    #     loss_lpips1 = lpips.LPIPS(net='alex').cuda(gpu)

    #     def loss_lpips(a,b):
    #         sz = a.shape

    #         a_new = a.reshape(sz[0]*sz[2], sz[1], sz[3], sz[4])
    #         b_new = b.reshape(sz[0]*sz[2], sz[1], sz[3], sz[4])

    #         final_loss = torch.sum(1e-1*loss_lpips1(a_new[:,0:3],
    #                  b_new[:,0:3]))
    #         return final_loss

    #     all_losses.append(loss_lpips)
        
    if 'angular' in args.loss:
        cos_between = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        angular_loss = lambda a,b: torch.mean(1e-1*torch.acos(torch.clamp(cos_between(a,b),-0.99999, 0.99999))*180/np.pi) 
        
        all_losses.append(angular_loss)

    loss_function = lambda a,b: torch.sum(torch.stack([torch.sum(all_losses[i](a,b)) for i in range(0, len(all_losses))]))
    
    return loss_function