import sys, os, glob
os.environ['CUDA_VISIBLE_DEVICES']='4'

sys.path.append("../.")
sys.path.append("../data/")
import numpy as np
import matplotlib.pyplot as plt
import torch

from utils.noisegen_helper_fun import get_model_noisegen_old,get_model_noisegen
from PIL import Image
import argparse, json
import scipy.io

import utils.noisegen_helper_fun as gh
import datasets.dataset_helper as datah
from utils.losses import define_loss


import torch.distributed as dist

import torch.multiprocessing as mp
import torch.nn as nn
# import time


def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])
    
    
def main():
    parser = argparse.ArgumentParser(description='Gan noise model training options.')
    parser.add_argument('--mode', default='train', help = 'Train or eval')
    parser.add_argument('--network', default='noUnet', help = 'Options: Unet, Unet_cat, noUnet, Unet_first')
    parser.add_argument('--stage', default=1, help = '1 for no dynamic, and 2 for dynamic')
    parser.add_argument('--noiselist', default='shot_read_uniform_fixed',#rnvd不太好train，所以去掉fixed噪声，只训练前面的
                        help = 'Specify the type of noise to include. \
                        Options: read, shot, uniform, row1, rowt, fixed1, learnedfixed, periodic')
    parser.add_argument('--crop_size', default=256, type = int)
    parser.add_argument('--t_length', default=16, type = int)
    parser.add_argument('--dataset', default= 'CRVD', help = 'Choose which data to use during training')
    parser.add_argument('--discriminator_loss', default='fourier', 
                        help = 'Choose generator loss. Options: mixed, fourier, real, mean') 
    parser.add_argument('--notes', default= 'yournamehere') 
    parser.add_argument('--preloaded', default = None,#'../saved_models/noisemodel_v3_alllevel',
                        help='Use pretrained model. Specify the file path here to load in the model')
    parser.add_argument('--generator_loss', default='lpips', help = 'Choose generator loss. Default: lpips') 
    parser.add_argument('--split_into_patches', default='patches_after') 
    parser.add_argument('--save_path', default = '../saved_models/', help='Specify where to save checkpoints during training')
    parser.add_argument('--unet_opts', default='residualFalse_conv_tconv_selu')
    parser.add_argument('--num_iter', default= 500000) 
    parser.add_argument('--device', default= 'cuda:0')
    parser.add_argument('--lr', default = 0.00003, type=float)
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument('--batch_size', default = 24, type=int)

    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, 
                        metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--milestone',default=[300])

    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes

    os.environ['MASTER_ADDR'] = '202.38.71.239'
    os.environ['MASTER_PORT'] = find_free_port()
    
    torch.cuda.empty_cache()
    
    folder_name = args.save_path + 'noisemodel' +"_".join([str(i) for i in list(args.__dict__.values())[0:6]])+'/'

   
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    with open(folder_name + 'args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    args.folder_name = folder_name
    
    # DDP
    if args.mode=='train':
        mp.spawn(train, nprocs=args.gpus, args=(args,))
    elif args.mode=='eval':
        mp.spawn(eval,nprocs=args.gpus,args=(args,))


def eval(gpu, args):
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(
        backend='nccl',
        init_method='env://',                                   
        world_size=args.world_size,
        rank=rank)
    print('loading generator')
    generator,_ = get_model_noisegen(args, gpu, mode=args.mode, best='best')
    # generator=get_model_noisegen_old(args,gpu)
    
    print('put on GPU', gpu)
    torch.cuda.set_device(gpu)
    generator.cuda(gpu)

    gen_loss, loss_fn_alex = define_loss(args, gpu)

    # Wrap the model
    generator = nn.parallel.DistributedDataParallel(generator,
                                                device_ids=[gpu], find_unused_parameters=True)
    
    # Set up dataset
    if args.dataset=='CRVD':
        _, dataset_list_test = datah.get_dataset_CRVD(args,dimension=2,test_mode='all')
    elif args.dataset=='RNVD':
        _, dataset_list_test=datah.get_dataset_RNVD(args,dimension=2,test_mode='all')

    print(dataset_list_test.__len__())
    print('shot=',generator.module.shot_noise)
    print('read=',generator.module.read_noise)
    print('uniform=',generator.module.uniform_noise)
    print('fixed=',generator.module.fixed_coeff)
    
    test_loader = torch.utils.data.DataLoader(dataset=dataset_list_test, 
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True)

    if args.split_into_patches == 'patches_before':
        split_patches = True
    else:
        split_patches = False


    # 计算KLD
    tot_kld = 0
    for i, sample in enumerate(test_loader):
        with torch.no_grad():
            # (b*t,c,h,w)
            B,C,H,W=sample['noisy_input'].shape
            noisy_raw = sample['noisy_input'].to(gpu)
            clean_raw = sample['gt_label_nobias'].to(gpu)
            pos=sample['pos']#pos[0]:(b); pos[1]:(b)
            noise_level=sample['noise_level']

            gen_noisy,_ = generator(clean_raw, split_patches,pos=pos,noise_level=noise_level)

            if split_patches == False:
                gen_noisy_ = gh.split_into_patches2d(gen_noisy).to(gpu)

            real_noisy = gh.split_into_patches2d(noisy_raw).to(gpu)
            clean = gh.split_into_patches2d(clean_raw).to(gpu)

            gen1 = (gen_noisy_).detach().cpu().numpy()
            real1 = (real_noisy).detach().cpu().numpy()
            kld_val = gh.cal_kld(gen1, real1)
            print('%dth image, iso=%d, kld=%f'%(i,noise_level[0],kld_val))
            tot_kld += kld_val

    # save_name = f'kld{kld_val:.5f}_noise{noise_level[0]:d}_gen.png'
    # Image.fromarray((np.clip(gen_noisy[0,:3].detach().cpu().numpy().transpose(1,2,0),0,1) * 255).astype(np.uint8)).save(save_name)
    # save_name = f'kld{kld_val:.5f}_noise{noise_level[0]:d}real.png'
    # Image.fromarray((np.clip(noisy_raw[0,:3].detach().cpu().numpy().transpose(1,2,0),0,1) * 255).astype(np.uint8)).save(save_name)

    print('Total KLD value: %.4f, Avg KLD value: %.4f'%(tot_kld,tot_kld/(i+1)))

    
def train(gpu, args):
    print('entering training function')
    
    print(args.nr, args.gpus, gpu, args.world_size)
    rank = args.nr * args.gpus + gpu                         
    dist.init_process_group(                                
        backend='nccl',                                         
        init_method='env://',                                   
        world_size=args.world_size,                              
        rank=rank)    
    
    
    print('loading model')
    generator, discriminator = get_model_noisegen(args, gpu, mode=args.mode)
    
    print('put on GPU', gpu)
    torch.cuda.set_device(gpu)
    generator.cuda(gpu)
    discriminator.cuda(gpu)
    
    folder_name = args.folder_name
    batch_size = args.batch_size
    
    gen_loss, loss_fn_alex = define_loss(args, gpu)
    
    optimizer_G = torch.optim.Adam([{'params':generator.parameters(), 'initial_lr':args.lr}], lr=args.lr, betas=(args.b1, args.b2))
    scheduler_G = torch.optim.lr_scheduler.MultiStepLR(optimizer_G, 
                                           milestones=args.milestone,gamma=0.5,last_epoch=0)#0.3162277
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    # 只训练模型的一部分
    for item,param in generator.named_parameters():
        # param.requires_grad = False if not 'basemodel_shot.tail' or 'basemodel_shot.tail0' in item else True
        if ('shot_noise' in item) or ('read_noise' in item) or ('uniform_noise' in item):
            param.requires_grad = False
    for name,param in generator.named_parameters():
        print(name, param.requires_grad)

    # Wrap the model
    generator = nn.parallel.DistributedDataParallel(generator,
                                                device_ids=[gpu], find_unused_parameters=True)
    
    # Wrap the model
    discriminator = nn.parallel.DistributedDataParallel(discriminator,
                                                device_ids=[gpu], find_unused_parameters=True)
    
    # Set up dataset
    if args.dataset=='CRVD':
        dataset_list, dataset_list_test = datah.get_dataset_CRVD(args,dimension=2)
    elif args.dataset=='RNVD':
        dataset_list, dataset_list_test=datah.get_dataset_RNVD(args,dimension=2)

    print(dataset_list.__len__(),dataset_list_test.__len__())
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_list, 
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    
    train_loader = torch.utils.data.DataLoader(dataset=dataset_list, 
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=train_sampler) 
    
    test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_list_test, 
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    
    test_loader = torch.utils.data.DataLoader(dataset=dataset_list_test, 
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=test_sampler) 
    
    
    
    if args.split_into_patches == 'patches_before':
        split_patches = True
    else:
        split_patches = False
    

    ## WGAN-GP
    n_critic = 5
    lambda_gp = 10
    num_epochs = 500
    G_losses = []
    D_losses = []
    kld_list = []
    real_list = []
    fake_list = []


    best_kld = 1e6
    for epoch in range(0,num_epochs):
        # 在ddp下保证shuffle
        train_sampler.set_epoch(epoch)
        for i, sample in enumerate(train_loader):
            # 训练合成噪声用二维数据，每次加载一张图片
            # 如果噪声级别不同的话在每个c通道再cat一个噪声级别
            B,C,H,W=sample['noisy_input'].shape
            noisy_raw = sample['noisy_input'].to(gpu)
            clean_raw = sample['gt_label_nobias'].to(gpu)
            pos=sample['pos']#pos[0]:(b); pos[1]:(b)
            noise_level=sample['noise_level']
            # print('1',noisy_raw.shape)

            # -----------------
            #  Train Discriminator 
            # -----------------
            ## Train with batch

            optimizer_D.zero_grad()

            # Generator fake noisy images
            gen_noisy,_ = generator(clean_raw, split_patches,pos=pos,noise_level=noise_level)
            # print('1',gen_noisy.shape)
            # gen_noisy:(b,c,h,w)
            if args.discriminator_loss == 'mean':
                gen_mean = torch.mean(gen_noisy,0).unsqueeze(0)#(b,c,h,w)
                real_mean = torch.mean(noisy_raw,0).unsqueeze(0)

                gen_noisy = torch.cat((gen_mean.repeat(args.t_length,1,1,1), gen_noisy),1)
                noisy_raw = torch.cat((real_mean.repeat(args.t_length,1,1,1), noisy_raw),1)


            # 这里的这个split_patch应该是为了判别器，因为判别器最后要输出一个loss，所以必须裁成64的倍数
            if split_patches == False:#patch_before是true，patch_after是false
                # split_patch:(b,c,h,w)->(b*n*n,c,64,64)
                gen_noisy = gh.split_into_patches2d(gen_noisy).to(gpu)
            real_noisy = gh.split_into_patches2d(noisy_raw).to(gpu)
            clean = gh.split_into_patches2d(clean_raw).to(gpu)

            if 'fourier' in args.discriminator_loss:
                #print('using fourier loss for discriminator')
                real_noisy = torch.abs(torch.fft.fftshift(torch.fft.fft2(real_noisy, norm="ortho")))
                gen_noisy = torch.abs(torch.fft.fftshift(torch.fft.fft2(gen_noisy, norm="ortho")))
                # print(real_noisy.shape,gen_noisy.shape)
                # torch.Size([64, 4, 64, 64]) torch.Size([64, 4, 64, 64])

            elif 'mixed' in args.discriminator_loss:
                #print('using fourier + real loss for discriminator')
                real_noisy1 = torch.abs(torch.fft.fftshift(torch.fft.fft2(real_noisy, norm="ortho")))
                gen_noisy1 = torch.abs(torch.fft.fftshift(torch.fft.fft2(gen_noisy, norm="ortho")))

                real_noisy = torch.cat((real_noisy, real_noisy1),1)
                gen_noisy = torch.cat((gen_noisy, gen_noisy1),1)

            elif 'complex' in args.discriminator_loss:
                #print('using fourier complex loss for discriminator')
                real_noisy1 = torch.fft.fftshift(torch.fft.fft2(real_noisy, norm="ortho"))
                gen_noisy1 = torch.fft.fftshift(torch.fft.fft2(gen_noisy, norm="ortho"))

                real_noisy = torch.cat((torch.real(real_noisy1), torch.imag(real_noisy1)),1)
                gen_noisy = torch.cat((torch.real(gen_noisy1), torch.imag(gen_noisy1)),1)

                
            real_validity = discriminator(real_noisy)
            fake_validity = discriminator(gen_noisy)


            # Gradient penalty
            gradient_penalty = gh.compute_gradient_penalty2d(discriminator, real_noisy.data, gen_noisy.data)
            # Adversarial loss
            # d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
            # 下面这个是一个更强一些的loss
            d_loss = torch.mean(fake_validity-real_validity) + lambda_gp * gradient_penalty

            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()


            # Train the generator every n_critic steps
            if i % n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------

                # Generate a batch of images
                fake_imgs,_ = generator(clean_raw, split_patches,pos=pos,noise_level=noise_level)
                if args.discriminator_loss == 'mean':
                    fake_imgs_mean = torch.mean(fake_imgs,0).unsqueeze(0)
                    fake_imgs = torch.cat((fake_imgs_mean.repeat(args.t_length,1,1,1), fake_imgs),1)
                
                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                if split_patches == False:
                    fake_imgs = gh.split_into_patches2d(fake_imgs).to(gpu)
                
                if 'fourier' in args.discriminator_loss:
                    #print('using fourier loss for discriminator')
                    fake_imgs = torch.abs(torch.fft.fftshift(torch.fft.fft2(fake_imgs, norm="ortho")))
                elif 'mixed' in args.discriminator_loss:
                    #print('using mixed loss for discriminator')
                    fake_imgs1 = torch.abs(torch.fft.fftshift(torch.fft.fft2(fake_imgs, norm="ortho")))
                    fake_imgs = torch.cat((fake_imgs, torch.abs(fake_imgs1)), 1)
                elif 'complex' in args.discriminator_loss:
                    #print('using mixed loss for discriminator')
                    fake_imgs1 = torch.fft.fftshift(torch.fft.fft2(fake_imgs, norm="ortho"))
                    fake_imgs = torch.cat((torch.real(fake_imgs1), torch.imag(fake_imgs1)),1)

                fake_validity = discriminator(fake_imgs)
                g_loss = -torch.mean(fake_validity)

                if args.generator_loss == 'lpips':
                    g_loss += gen_loss(fake_imgs, real_noisy)


                g_loss.backward()
                optimizer_G.step()

        scheduler_G.step()

        print(
            "[Epoch %d/%d] [Batch %d] [D loss: %f] [G loss: %f]"
            % (epoch, num_epochs, i, d_loss.item(), g_loss.item())
        )

        gen1 = (gen_noisy).detach().cpu().numpy()
        real1 = (real_noisy).detach().cpu().numpy()
        kld_val = gh.cal_kld(gen1, real1)
        print('KLD', kld_val)

        G_losses.append(g_loss.item())
        D_losses.append(d_loss.item())
        kld_list.append(kld_val)

        real_list.append(torch.mean(real_validity).item())
        fake_list.append(torch.mean(fake_validity).item())


        # Check if Best KLD value
        if epoch % 10 == 0:
            tot_kld = 0
            for i, sample in enumerate(test_loader):
                # 看看能不能在这整一个包括不同噪声级别的kld对比

                with torch.no_grad():
                    # (b,c,h,w)
                    noisy_raw = sample['noisy_input'].to(gpu)
                    clean_raw = sample['gt_label_nobias'].to(gpu)
                    pos=sample['pos']#pos[0]:(b); pos[1]:(b)
                    noise_level=sample['noise_level']

                    gen_noisy,_ = generator(clean_raw, split_patches,pos=pos,noise_level=noise_level)

                    if split_patches == False:
                        gen_noisy = gh.split_into_patches2d(gen_noisy).to(gpu)

                    real_noisy = gh.split_into_patches2d(noisy_raw).to(gpu)
                    clean = gh.split_into_patches2d(clean_raw).to(gpu)

                    gen1 = (gen_noisy).detach().cpu().numpy()
                    real1 = (real_noisy).detach().cpu().numpy()
                    kld_val = gh.cal_kld(gen1, real1)
                    tot_kld += kld_val

            print('Total KLD value:', tot_kld)
            print('shot=',generator.module.shot_noise)
            print('read=',generator.module.read_noise)
            print('uniform=',generator.module.uniform_noise)
            print('fixed=',generator.module.fixed_coeff)

            if tot_kld < best_kld:
                best_kld = tot_kld

                print('saving best')
                checkpoint_name = folder_name + f'bestgenerator{epoch}_KLD{best_kld:.5f}.pt'
                torch.save(generator.state_dict(), checkpoint_name)

                checkpoint_name = folder_name + f'bestdiscriminatort{epoch}_KLD{best_kld:.5f}.pt'
                torch.save(discriminator.state_dict(), checkpoint_name)


            if gpu==0:
                print('saving checkpoint')

                out_plt = gen_noisy.cpu().detach().numpy()[0].transpose(1,2,0)[...,0:3]

                checkpoint_name = folder_name + f'generatorcheckpoint{epoch}_Gloss{G_losses[-1]:.5f}_Dloss{np.round(D_losses[-1], 5)}.pt'
                torch.save(generator.state_dict(), checkpoint_name)

                checkpoint_name = folder_name + f'discriminatorcheckpoint{epoch}_Gloss{G_losses[-1]:.5f}_Dloss{np.round(D_losses[-1], 5)}.pt'
                torch.save(discriminator.state_dict(), checkpoint_name)

                save_name = folder_name + f'testimage{epoch}.jpg'
                Image.fromarray((np.clip(out_plt,0,1) * 255).astype(np.uint8)).save(save_name)

                scipy.io.savemat(folder_name + 'losses.mat',
                                {'G_losses':G_losses, 
                                 'D_losses':D_losses,
                                'kld_list':kld_list,
                                'real_list':real_list,
                                'fake_list':fake_list})
        
if __name__ == '__main__':
    main()