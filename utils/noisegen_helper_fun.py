import torch.nn as nn
import torch
import numpy as np
from models.spectral_normalization import SpectralNorm
from torch.autograd import Variable
Tensor = torch.cuda.FloatTensor
import torch.autograd as autograd
import scipy.io
import argparse, json, glob, os, sys
from models.unet import Unet
from models.gan_noisemodel import NoiseGenerator2d_distributed_ablation,\
DiscriminatorS2d_sig,NoiseGenerator2d_v2,NoiseGenerator2d_v3


from pathlib import Path
_script_dir = Path( __file__ ).parent
_root_dir = _script_dir.parent

def t32(x):
    return torch.transpose(x,0, 2).squeeze(2)
def t23(x):
    return torch.transpose(x, 0,1).unsqueeze(0)

def t32_1(x):
    '''(b,c,t,h,w)->(b*t,c,h,w)'''
    # 这个的作用是dataloader加载的数据转换为gan能接受的大小
    x= torch.transpose(x,1,2)
    return x.reshape(x.shape[0]*x.shape[1],x.shape[2],x.shape[3],x.shape[4])

def t23_1(x,t_length):
    '''(b*t,c,h,w)->(b,c,t,h,w)'''
    return x.view(-1,t_length,x.shape[-3],x.shape[-2],x.shape[-1]).transpose(1, 2)


def get_discriminator_models(args,keys):
    discriminators={}
    if args.discriminator_loss == 'mean' or args.discriminator_loss == 'complex' or args.discriminator_loss == 'mixed':
        disc_channels = 8
    else:
        disc_channels = 4
    # 最后一个阶段不需要判别器
    for key in keys[:-1]:
        discriminators[key] = DiscriminatorS2d_sig(channels = disc_channels)
    return discriminators


def get_model_noisegen(args, device, mode, best='best'):
    '''best: best或者latest'''
    folder_name=None
    noise_list_new=args.noiselist
    dataset=args.dataset
    if args.preloaded:
        folder_name=args.preloaded
        parser = argparse.ArgumentParser(description='Process some integers.')
        args = parser.parse_args('')
        with open(folder_name + '/args.txt', 'r') as f:
            args.__dict__ = json.load(f)
            args.fraction_video = 50
            args.resume_from_checkpoint = folder_name

    if args.discriminator_loss == 'mean' or args.discriminator_loss == 'complex' \
        or args.discriminator_loss == 'mixed':
        disc_channels = 8
    else:
        disc_channels = 4

    # old version: 
    # discriminator = gh.DiscriminatorS2d().to(args.device)
    discriminator = DiscriminatorS2d_sig(channels = disc_channels)
    
    # old version: 
    # generator = NoiseGenerator2d_distributed_ablation(net = model, unet_opts = args.network, noise_list = args.noiselist, 
    #                                            device = device)
    if dataset=='CRVD':
        shape=(4,1080,1920)
        noise_levels=5
    elif dataset=='RNVD':
        shape=(4,1536,2048)
        noise_levels=4
    
    print('noise:',noise_list_new)
    if args.stage==1:
        # 第一个阶段无dynamic
        # generator = NoiseGenerator2d_v2(net = None, noise_list = args.noiselist, 
        #                                         device = device,res_learn=True,dynamic=False, noise_levels=5)
        generator = NoiseGenerator2d_v3(net = None, noise_list = noise_list_new, device = device,\
                                        res_learn=True,dynamic=False, noise_levels=noise_levels, fixed_shape=shape)
        if mode=='eval' or folder_name:
            generator=load_from_checkpoint_ab(generator, folder_name, device = device, ep=best)
            print('resuming from ',folder_name)
            # for k, v in generator.named_parameters():
            #     if 'read' or 'uniform' or 'fixed' in k:
            #         v.data[0].zero_()
            #     print(k,v)

    elif args.stage==2:
        # 第二个阶段有dynamic，且加载预训练的模型
        # generator = NoiseGenerator2d_v2(net = None, noise_list = args.noiselist, 
        #                                         device = device,res_learn=True,dynamic=True, noise_levels=5)
        generator = NoiseGenerator2d_v3(net = None, noise_list = args.noiselist, 
                                                device = device,res_learn=True,dynamic=True, noise_levels=5)
        generator=load_from_checkpoint_ab(generator, folder_name, device = device, ep=best)
    else:
        print("Invalid stage!")
    return generator, discriminator

# 早期1种噪声级别的generator
def get_model_noisegen_old(args,device):#得重新写，这部分不是很统一
    generator = NoiseGenerator2d_distributed_ablation(net = None, unet_opts = args.network, noise_list = args.noiselist, 
                                        device = device,res_learn=True,dynamic=True)
    generator = load_from_checkpoint_ab(generator, args.preloaded, device = device)

    return generator#.cuda(gpu)


def load_from_checkpoint_ab(generator, folder_name, device='cuda:0', ep='best'):
    '''
    res_learn: 返回的是加噪声的图像还是每一步生成的噪声
    dynamic: 是否采用动态权重
    '''
    
    if ep == 'best':
        list_of_files = glob.glob(folder_name + '/bestgen*.pt') # * means all if need specific format then *.csv
        kld_best = []
        for i in range(0,len(list_of_files)):
            kld_best.append(float(list_of_files[i].split('KLD')[-1].split('.pt')[0]))
        inds_sorted = np.argsort(kld_best)
        best_files = np.array(list_of_files)[inds_sorted]

        latest_file = best_files[0]        
        print('best kld:' , np.min(kld_best))

    elif ep == 'latest':
        list_of_files = glob.glob(folder_name + '/gen*.pt') # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getctime)
    else:
        list_of_files = glob.glob(folder_name + '/generatorcheckpoint' + str(ep) +'_' + '*.pt')
        #print(list_of_files)
        latest_file = list_of_files[0]
        
    path = latest_file
    
    saved_state_dict = torch.load(path, map_location ='cuda:'+str(device))
    distributed_model = False
    for key in saved_state_dict:
        if 'module' in key:
            distributed_model = True
            print('distributed')
            break
        
    if distributed_model == True:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in saved_state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        saved_state_dict = new_state_dict
        
    generator.load_state_dict(saved_state_dict,strict=False)
    
    if ep == 'best':
        curr_epoch = int(path.split('/')[-1].split('_')[0].split('bestgenerator')[1])
    else:
        curr_epoch = int(path.split('/')[-1].split('_')[0].split('generatorcheckpoint')[1])
    print('resuming from epoch', curr_epoch)

    return generator

    
def split_into_patches(x, patch_size = 64):
    patches = torch.empty([1,4,16,patch_size,patch_size])
    for xx in range(0,x.shape[-2]//patch_size):
        for yy in range(0,x.shape[-1]//patch_size):
            patches = torch.cat([patches, x[...,xx*patch_size:(xx+1)*patch_size, yy*patch_size:(yy+1)*patch_size]], 0)
    patches = patches[1:,...]
    return patches

def split_into_patches2d(x, patch_size = 64):#(b*t,c,h,w)->()
    patches = torch.empty([1,x.shape[1],patch_size,patch_size], device = x.device)#(1,c,h,w)
    for xx in range(0,x.shape[-2]//patch_size):
        for yy in range(0,x.shape[-1]//patch_size):
            patches = torch.cat([patches, x[...,xx*patch_size:(xx+1)*patch_size, yy*patch_size:(yy+1)*patch_size]], 0)
    patches = patches[1:,...]
    return patches

def get_histogram(data, bin_edges=None, left_edge=0.0, right_edge=1.0, n_bins=1000):
    data_range = right_edge - left_edge
    bin_width = data_range / n_bins
    if bin_edges is None:
        bin_edges = np.arange(left_edge, right_edge + bin_width, bin_width)
    bin_centers = bin_edges[:-1] + (bin_width / 2.0)
    n = np.prod(data.shape)
    # 有空的时候改成并行的
    batch_sz=data.shape[0]
    allhist=[]
    for i in range(batch_sz):
        hist, _ = np.histogram(data[i,...], bin_edges)
        allhist.append(hist)
    allhist=np.stack(allhist,axis=0)
    return allhist / n, bin_centers

def cal_kld(p_data, q_data, left_edge=0.0, right_edge=1.0, n_bins=1000):
    #这个kld是针对一种噪声级别写的，几个级别算一块去了，如果要适用于多种级别得重新写
    """Returns forward, inverse, and symmetric KL divergence between two sets of data points p and q"""
    bw = 0.2 / 64
    bin_edges = np.concatenate(([-1000.0], np.arange(-0.1, 0.1 + 1e-9, bw), [1000.0]), axis=0)
    bin_edges = None
    # print(p_data.shape,q_data.shape)
    # (64, 4, 64, 64) (64, 4, 64, 64)
    # (1000,) (1000,)
    p, _ = get_histogram(p_data, bin_edges, left_edge, right_edge, n_bins)
    q, _ = get_histogram(q_data, bin_edges, left_edge, right_edge, n_bins)
    idx = (p > 0) & (q > 0)
    p = p[idx]
    q = q[idx]
    logp = np.log(p)
    logq = np.log(q)
    kl_fwd = np.sum(p * (logp - logq))
    kl_inv = np.sum(q * (logq - logp))
    kl_sym = (kl_fwd + kl_inv) / 2.0
    return kl_sym #kl_fwd #, kl_inv, kl_sym

def compute_gradient_penalty2d(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.tensor(np.random.random((real_samples.size(0), 1, 1, 1)), dtype = real_samples.dtype, device = real_samples.device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)[...,0]
    fake = Variable(Tensor(d_interpolates.shape[0], 1).fill_(1.0), requires_grad=False).view(-1)
    
    #print(d_interpolates.shape, interpolates.shape, fake.shape)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.tensor(np.random.random((real_samples.size(0), 1, 1, 1,1)), dtype = real_samples.dtype, device = real_samples.device)
    # Get random interpolation between real and fake samples
    #print(alpha.shape, fake_samples.shape)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(d_interpolates.shape[0], 1).fill_(1.0), requires_grad=False).view(-1)
    
    #print(d_interpolates.shape, interpolates.shape, fake.shape)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty