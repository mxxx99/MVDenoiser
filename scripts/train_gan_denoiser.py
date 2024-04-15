import sys, os, glob
sys.path.append('..')
os.environ['CUDA_VISIBLE_DEVICES']='7'

import numpy as np
import torch
from torch.optim import Adam,lr_scheduler
from PIL import Image
import argparse, json
import scipy.io
import re
import datasets.crvd_supervised_dataset as dset
import utils.noisegen_helper_fun as gh
import datasets.dataset_helper as datah
from utils.losses import define_loss_denoiser
from utils.lr_scheduler import WarmupMultiStepLR
from utils.denoiser_helper_fun import load_generator_model,\
get_model_denoiser,resume_from_checkpoint,preload_model,\
save_checkpoint,denoise_shuffle
from utils.losses import define_loss

import torch.distributed as dist

import torch.multiprocessing as mp
import torch.nn as nn
import time

from pathlib import Path
_script_dir = Path( __file__ ).parent
_root_dir = _script_dir.parent
    
def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])
    
def main():
    parser = argparse.ArgumentParser(description='Denoising options')
    parser.add_argument('--mode', default='eval', help = 'train or eval')
    parser.add_argument('--network', default='Unet3D', help='Choose which network to use')
    parser.add_argument('--noise_cat',default=True)
    parser.add_argument('--noise_type', default='unetfourier', help='Choose which noise model to load')

    parser.add_argument('--loss', default= 'Charbonnier',#'MSE',#'L1_LP2D', 
                        help = 'Choose loss or a combination of losses. Options: MSE, L1, TV, LP2D (LPIPS),\
                        angular (Cosine similarity between the 4 color channels)')
    parser.add_argument('--discriminator_loss', default='fourier', 
                        help = 'Choose generator loss. Options: mixed, fourier, real, mean')
    parser.add_argument('--generator_loss', default='lpips', help = 'Choose generator loss. Default: lpips') 

    parser.add_argument('--dataset', default= 'CRVD', help = 'Choose which data to use during training')
    parser.add_argument('--multiply', default= 'None', 
                        help = 'Choose what sort of processing to do on the ground truth images. Options: None, \
                        histeq (does histogram equalization on the ground truth images before taking the loss), \
                        gamma (goes gamma correction on the ground truth images, makes the network learn \
                        denoising + gamma correction')
    parser.add_argument('--space', default= 'linear') # 
    parser.add_argument('--notes', default= 'Test') 
    parser.add_argument('--t_length', default=5, type = int) 
    parser.add_argument('--crop_size', default= 256, 
                        help='Choose the image patch size for denoising. \
                        Options: 512, (512x512), full (1024x512), small (128x128), or 256 (256x256)')  
    parser.add_argument('--unet_opts', default='residualFalse_conv_tconv_selu')
    parser.add_argument('--num_iter', default=120)
    parser.add_argument('--preloaded', default = False,
                        help='Use pretrained model. Specify the file path here to load in the model')
    parser.add_argument('--generator_path', default = '../train_results/with_residual/on_CRVD/noisemodel_v3_alllevel',#'../train_results/with_residual/on_RNVD/noisemodel_rnvd_all3',#
                        help='Preload Generator path for noise generate')
    parser.add_argument('--resume_from_checkpoint', default = '../train_results/with_residual/on_RNVD/denoiser_rnvd',#'../train_results/with_residual/on_CRVD/denoiser_crvd5_2',#False,
                        help='Resume training from a saved checkpoint. Specify the filepath here')
    parser.add_argument('--learning_rate', default = 0.0002, type=float)
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

    parser.add_argument('--batch_size', default = 5, type=int)
    parser.add_argument('--save_path', default = '../saved_models/', help='Specify where to save checkpoints during training')
    parser.add_argument('--MOT_path', default = '../data/MOTfiles_raw/',
                        help='If using unprocessed MOT images during training, specify the filepath \
                        for where your MOT dataset is stored')
    parser.add_argument('--stills_path', default = '../data/paired_data/stillpairs_mat', 
                        help='Specify the filepath for the stills dataset.')
    parser.add_argument('--cleanvideo_path', default = '../data/RGBNIR_videodataset_mat/',
                        help='Specify the filepath for the clean video dataset.')
    parser.add_argument('--show_every', default= 40, help='Choose show frequency. show every N batches') 
    parser.add_argument('--save_every', default = 5, type=int, 
                        help='Choose save frequency. Save every N epochs.')
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, 
                        metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--milestone',default=[35,55,70,90,120])
    parser.add_argument('--co_training', default=False, help='if train with optical flow')
    parser.add_argument('--use_syn', default=False, help='if use synthetic noise for training')
    parser.add_argument('--res_learn', default=False, help='use noise or noisy for supervision')
    parser.add_argument('--save_for_srga',default=False)#保存feat
    parser.add_argument('--save_inter',default=True)#保存每个结果生成的img
    parser.add_argument('--stage',default='tune',help='pretrain_clean,pretrain_noisy or tune')

    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    
    # Load in a pre-trained network before starting training
    if args.preloaded:
        print('loading in model')
        parser = argparse.ArgumentParser(description='Process some integers.')
        args1 = parser.parse_args('')
        with open(args.preloaded + '/args.txt', 'r') as f:
            args1.__dict__ = json.load(f)
        args.network = args1.network
        args.loss = args1.loss
        args.unet_opts = args1.unet_opts

        
    # Make folder 
    base_folder = args.save_path
    folder_name = base_folder +"_".join([str(i) for i in list(args.__dict__.values())[0:7]])+'/'

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    with open(folder_name + 'args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        
    args.folder_name = folder_name

    os.environ['MASTER_ADDR'] = '127.0.0.1'              
    os.environ['MASTER_PORT'] = find_free_port()
    
    torch.cuda.empty_cache()
    
    if args.mode=='train':
        mp.spawn(train, nprocs=args.gpus, args=(args,))
    elif args.mode=='eval':
        mp.spawn(eval,nprocs=args.gpus,args=(args,))


def save_interfeat(args,predicted_noise,ind):
    root_dir='rnvd_train_noblind_on%s/%s/'%(args.dataset,ind)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    # for ch in model.module().feature.children():
    #     feat=ch(net_inp)
    for key,pred_noise in predicted_noise.items():
        out_plt = pred_noise.cpu().detach().numpy()[0,:,0].transpose(1,2,0)[...,0:3]#(b,c,t,h,w)->(h,w,c)
        Image.fromarray((np.clip(out_plt,0,1) * 255).astype(np.uint8)).save(os.path.join(root_dir,'%s.png'%key))




def run_test(gpu, args, test_loader, model, criterion, generator,keys):
    print('running test')

    model.eval()#在conv里有bn/dropout，所以inference的时候要加上.eval
    with torch.no_grad():
        avg_test_loss = 0
        avg_test_psnr = 0
        avg_test_ssim = 0
        
        out_plt_inter={}
        for i, sample in enumerate(test_loader): #Loop through dataset
            gt_label = sample['gt_label_nobias'].cuda(non_blocking=True)#(b,c,t,h,w)
            # net_input=gt_label.clone()
            net_input = sample['noisy_input'].cuda(non_blocking=True)
            pos=sample['pos']   # 裁剪位置
            noise_level=sample['noise_level']

            noise_levels=generator.get_noise_level(noise_ind=noise_level)#每个seq一个noise_level
            denoise_all,denoise_inter,predicted_noise,feat_inter,mask = model(net_input,gt_label,noise_level=noise_levels)
            mask=1.0-mask.unsqueeze(2) if not mask==None else None

            if args.save_inter:
                save_interfeat(args,predicted_noise,'%d_%d'%(noise_level[0],i))
            iso='gt_ablation'
            if args.save_for_srga:
                if not os.path.exists('%s_%s'%(args.dataset,iso)):
                    os.mkdir('%s_%s'%(args.dataset,iso))
                torch.save(feat_inter, '%s_%s/%d.pt'%(args.dataset,iso,i))


            
            # test loss暂时只算最终的
            # test_loss = criterion(denoise_all, gt_label[:,:,2:-2])
            if not args.stage=='tune':
                test_loss = criterion(torch.mul(denoise_all,mask), torch.mul(gt_label[:,:,2:-2],mask))
            else:
                test_loss = criterion(denoise_all, gt_label[:,:,2:-2])

            avg_test_loss += test_loss.item()
            # 第一种psnr算法
            cur_test_psnr=torch.mean(datah.batch_psnr(denoise_all,gt_label[:,:,2:-2])).item()
            cur_test_ssim=datah.raw_ssim(denoise_all,gt_label[:,:,2:-2])
            # 第二种psnr算法
            # cur_test_psnr=datah.batch_psnr(denoise_all,gt_label[:,:,2:-2,...])

            avg_test_psnr += cur_test_psnr
            avg_test_ssim += cur_test_ssim
            print('fig ind=',i,' cur psnr=',cur_test_psnr,' cur ssim=',cur_test_ssim)

            if i==0:
                # out_plt = gt_label.cpu().detach().numpy()[0,:,2].transpose(1,2,0)[...,0:3]
                out_plt = denoise_all.cpu().detach().numpy()[0,:,0].transpose(1,2,0)[...,0:3]#(b,c,t,h,w)->(h,w,c)
                for key in keys:
                    out_plt_inter[key]=denoise_inter[key].cpu().detach().numpy()[0,:,2].transpose(1,2,0)[...,0:3]

        avg_test_loss = avg_test_loss/(i+1)
        avg_test_psnr = avg_test_psnr/(i+1)
        avg_test_ssim = avg_test_ssim/(i+1)
        print('avg test ssim: ',avg_test_ssim)
        
    return avg_test_loss, avg_test_psnr, out_plt#,out_plt1#,out_plt_inter

def write_logs(string,log_dir='log.txt'):
    with open(log_dir,'a') as f:
        f.write(string+'\n')
    
# Main training function 
def train(gpu, args):
    lambda1=1.0
    lambda2=1-lambda1
    lambda3=1.0
    discriminator_epoch=92
    write_logs('entering training function')
    print('entering training function')
    # print(args.nr, args.gpus, gpu, args.world_size)
    # Setup for distributed training on multiple GPUs
    rank = args.nr * args.gpus + gpu               
    dist.init_process_group(                                   
        backend='nccl',                                         
        init_method='env://',                                   
        world_size=args.world_size,
        rank=rank)
    
    # -----------------
    # load models and put on gpu
    # model：去噪模型；generator：合成噪声模型；discriminator：判别器
    write_logs('loading generator %d'%gpu)
    print('loading generator', gpu)
    generator = load_generator_model(args, gpu,args.res_learn)

    # keys=generator.get_keys()#generator中间返回几个结果，就需要几个去噪模型
    # keys=['shot','read', 'uniform']#, 'fixed']
    keys=[]
    write_logs('keys: %s'%keys)
    print('keys:',keys)

    write_logs('loading model')
    print('loading model')
    model = get_model_denoiser(args,gpu,keys,args.res_learn)
    
    # 判别器是2d的，输入是(b,c,h,w)
    # 暂时还是用noisegen的判别器
    discriminators=gh.get_discriminator_models(args,keys)
    
    write_logs('put on GPU %d'%gpu)
    print('put on GPU', gpu)
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    generator.cuda(gpu)

    for key in keys[:-1]:
        discriminators[key].cuda(gpu)
    
    # if args.preloaded:
    #     model = preload_model(args, model, gpu)
    
    if args.resume_from_checkpoint:
        write_logs('resuming model from checkpoint')
        print('resuming model from checkpoint')
        model, discriminators,curr_epoch, G_losses, D_losses, real_list, fake_list, test_psnr_list, test_loss, folder_name \
            = resume_from_checkpoint(args, model, gpu, best=False, discriminators=discriminators,load_discriminators=False,strict=True)
            
        best_test_loss = test_loss
        test_loss_list=[test_loss]
        write_logs('best loss is: %f'%best_test_loss)
        print('best loss is: ', best_test_loss)
    else:
        curr_epoch = 0; train_loss = []; test_loss_list = []; best_test_loss = 1e9
        folder_name = args.folder_name
        
    
    batch_size = args.batch_size
    # -----------------
    # losses and optimizers
    gen_loss, loss_fn_alex = define_loss(args, gpu)
    denoiser_loss = define_loss_denoiser(args, gpu)


    # g是一个optimizer，每个key的d都有各自的optimizer
    if hasattr(model,'trainable_parameters'):
        print('using trainable parameters')
        params=[]

        for item in model.trainable_parameters():
            params.append({'params':item,'initial_lr':args.learning_rate})

        for item,param in model.named_parameters():
            # param.requires_grad = False if not 'basemodel_shot.tail' or 'basemodel_shot.tail0' in item else True
            if 'pwcnet' in item:
                param.requires_grad = False
            # if re.match('basemodel_[a-z]*.norm',item)!= None:
            #     param.requires_grad = False

        if args.co_training:
            params.append({'params':model.pwcnet.parameters(),'initial_lr':1e-5})
            print('co-training with optical flow with lr %f'%(1e-5))
            write_logs('co-training with optical flow with lr %f'%(1e-5))

        optimizer_G = Adam(params, lr=args.learning_rate)
    else:
        params=model.parameters()        
        optimizer_G = Adam([{'params':params, 'initial_lr':args.learning_rate}], lr=args.learning_rate)
    scheduler_G = lr_scheduler.MultiStepLR(optimizer_G, 
                                           milestones=args.milestone,gamma=0.5,last_epoch=curr_epoch-1)#0.3162277
    # scheduler_G = WarmupMultiStepLR(optimizer_G, warmup_steps=3,milestones=args.milestone,gamma=0.5,last_epoch=curr_epoch-1)

    for name,param in model.named_parameters():
        print(name, param.requires_grad)

    optimizer_D={}
    for key in keys[:-1]:
        optimizer_D[key] =Adam(discriminators[key].parameters(), lr=args.learning_rate, betas=(args.b1, args.b2))
    

    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[gpu], find_unused_parameters=True)
    
    for key in keys[:-1]:
        discriminators[key] = nn.parallel.DistributedDataParallel(discriminators[key],
                                                    device_ids=[gpu], find_unused_parameters=True)

    # Set up dataset
    # dataset_list, dataset_list_test, i0 = get_dataset(args)
    if args.dataset=='CRVD':
        dataset_list, dataset_list_test=datah.get_dataset_CRVD(args,dimension=3)
    elif args.dataset=='SRVD':
        dataset_list, dataset_list_test=datah.get_dataset_SRVD(args,dimension=3)
    elif args.dataset=='DAVISraw':
        if not args.use_syn:
            dataset_list, dataset_list_test=datah.get_dataset_DAVISraw(args,dimension=3)
        else:
            dataset_list, dataset_list_test=datah.get_dataset_DAVISraw_syn(args,dimension=3)
    elif args.dataset=='RNVD':
        dataset_list, dataset_list_test=datah.get_dataset_RNVD(args,dimension=3)
    print('using %s'%args.dataset)

    write_logs('train length: %d  test length: %d'%(dataset_list.__len__(),dataset_list_test.__len__()))
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

    split_patches = False   # 在generator中按照正常尺寸处理，裁剪后再送入discriminator
    
    ## WGAN-GP
    n_critic = 1#可以改成判别器1步，生成器1步
    lambda_gp = 10
    G_losses = []
    D_losses = {}
    for key in keys[:-1]:
        D_losses[key]=[]
    real_list = []
    fake_list = []
    test_psnr_list=[]

    write_logs('Enter training loop')
    print('Enter training loop')
    for epoch in range(curr_epoch, args.num_iter):
        # noisy_all、noisy_inter分别表示合成噪声模型的最终合成结果和中间阶段
        # denoise_all、denoise_inter分别表示去噪模型的最终结果和中间阶段
        # 两个阶段的key对应的是相同的阶段
        train_sampler.set_epoch(epoch)
        for i, sample in enumerate(train_loader):
            # 真实的带噪图片
            B,C,T,H,W=sample['noisy_input'].shape
            noisy_raw = sample['noisy_input'].cuda(non_blocking=True)
            clean_raw = sample['gt_label_nobias'].cuda(non_blocking=True)#(b,c,t,h,w)
            noise_level=sample['noise_level']
            pos=sample['pos']

            # 从generator处获得中间加噪声的结果
            with torch.no_grad():
                # gh.t32_1(gt_label)：(b,c,t,h,w)->(b*t,c,h,w)
                # loader得到的noise_level是(b,)tensor，对每个sequence，所以用到合成模型里时要加上t维度
                # noise_inter:每个阶段和之前的噪声总和
                '''noisy_all,noise_inter=generator(gh.t32_1(clean_raw),pos=pos,noise_level=noise_level.unsqueeze(1).repeat(1,T))'''
                noisy_all,noise_inter=generator(gh.t32_1(clean_raw[:,:,2:-2]),pos=pos,noise_level=noise_level.unsqueeze(1).repeat(1,T-4))
                
                noise_levels=generator.get_noise_level(noise_ind=noise_level)#每个seq一个noise_level
                # noisy_all=gh.t23_1(noisy_all,T)#(b*t,c,h,w)->(b,c,t,h,w)
                # for key in keys:
                #     noisy_inter[key] = gh.t23_1(noisy_inter[key],T)

            # -----------------
            #  Train Discriminator 
            # -----------------
            ## Train with batch
            d_loss=0.0
            fake_validity=None;real_validity=None
            if len(keys)>0:
                if epoch<discriminator_epoch:
                    # 真实噪声图训练
                    with torch.no_grad():
                        # 训练判别器时不需要生成器的梯度
                        gen_denoise_all,gen_noise_inter,gen_predicted_noise,_,_ = model(noisy_raw,clean_raw,noise_level=noise_levels)

                    # 在训练gan前可以先对generator用mse loss算一次梯度
                    # 呃...需要考虑下咋样能把gan loss和mse loss搞到一起
                    # g_loss=denoiser_loss(gen_denoise_all,clean_raw)
                    # g_loss.backward()
                    # optimizer_G.step()

                    for key in keys[:-1]:
                        optimizer_D[key].zero_grad()

                    # 倒序和合成噪声的每个加噪结果形成监督
                    for key in keys[:-1]:
                        # 在每个epoch中，每个key对应的判别器分别优化
                        if split_patches == False:#patch_before是true，patch_after是false
                            # split_patch:(b*t,c,h,w)->(b*t*n*n,c,64,64)
                            # 把noise generator产生的当作real，和model产生的fake训练discriminator
                            '''gen_noise_inter_cur = gh.split_into_patches2d(gh.t32_1(gen_noise_inter[key])).to(gpu)'''
                            gen_noise_inter_cur = gh.split_into_patches2d(gh.t32_1(gen_noise_inter[key][:,:,2:-2])).to(gpu)
                            real_noise_inter_cur = gh.split_into_patches2d(noise_inter[key]).to(gpu)
                        # clean = gh.split_into_patches2d(clean_raw).to(gpu)

                        if 'fourier' in args.discriminator_loss:
                            #print('using fourier loss for discriminator')
                            real_noise_inter_cur = torch.abs(torch.fft.fftshift(torch.fft.fft2(real_noise_inter_cur, norm="ortho")))
                            gen_noise_inter_cur = torch.abs(torch.fft.fftshift(torch.fft.fft2(gen_noise_inter_cur, norm="ortho")))
                        
                        # 在送入判别器之前最好能再对3d->2d的images做一次shuffle，保证随机性
                        real_noise_inter_cur,gen_noise_inter_cur=\
                            denoise_shuffle(real_noise_inter_cur,gen_noise_inter_cur)
                        real_validity = discriminators[key](real_noise_inter_cur)
                        fake_validity = discriminators[key](gen_noise_inter_cur)

                        # Gradient penalty
                        gradient_penalty = gh.compute_gradient_penalty2d(discriminators[key], real_noise_inter_cur.data, gen_noise_inter_cur.data)
                        # Adversarial loss
                        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
                        
                        D_losses[key].append(d_loss.item())
                        d_loss.backward()
                        optimizer_D[key].step()

            optimizer_G.zero_grad()

            # Train the generator every n_critic steps
            # 几个生成器一起优化，loss是几个gan loss和一个mse loss的和
            # 如果每n_critic步优化一次generator，那实际上相当于generator只利用了一部分信息
            if i % n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------

                # Generate a batch of images
                fake_denoise_all,denoised_inter,predicted_noise,_,mask = model(noisy_raw,clean_raw,noise_level=noise_levels)
                mask=1.0-mask.unsqueeze(2)

                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                g_loss=0.0

                for idx, key in enumerate(keys[:-1]):
                    if split_patches == False:
                        '''denoise_inter_cur=gh.split_into_patches2d(gh.t32_1(denoised_inter[key])).to(gpu)'''
                        denoise_inter_cur=gh.split_into_patches2d(gh.t32_1(denoised_inter[key][:,:,2:-2])).to(gpu)
                        real_noisy_inter_cur=gh.split_into_patches2d(noise_inter[key]).to(gpu)

                    if 'fourier' in args.discriminator_loss:
                        denoise_inter_cur=torch.abs(torch.fft.fftshift(torch.fft.fft2(denoise_inter_cur, norm="ortho")))
                        real_noisy_inter_cur=torch.abs(torch.fft.fftshift(torch.fft.fft2(real_noisy_inter_cur, norm="ortho")))

                    if epoch>discriminator_epoch:
                        with torch.no_grad():
                            fake_validity = discriminators[key](denoise_inter_cur)
                    else:
                        fake_validity = discriminators[key](denoise_inter_cur)
                    g_loss -= torch.mean(fake_validity)

                    # 现在这里是对加噪声后结果频谱图的一个lpips loss
                    # g_loss += lambda1*gen_loss(fake_noise_inter_cur, real_noise_inter[key])
                    g_loss += lambda1*gen_loss(denoise_inter_cur, real_noisy_inter_cur)
                    
                    # 一个额外的约束项
                    # 这部分还没有改
                    if not lambda2==0:
                        denoise_after=denoised_inter[keys[idx-1]] if idx>=1 else clean_raw
                        noise_cur=gh.t23_1(noise_inter[key],T) if args.res_learn else \
                            (gh.t23_1(noise_inter[key],T)-
                            (gh.t23_1(noise_inter[keys[idx-1]],T) if idx>=1 else clean_raw))
                        denoised_cur,_=model.module.denoise_stage(
                                    [denoise_after+noise_cur,noise_levels[key]] if args.noise_cat
                                    else denoise_after+noise_cur
                                    ,key)
                        
                        g_loss+=lambda2*denoiser_loss(denoised_cur,denoise_after)
                
                # g_loss+=lambda3*denoiser_loss(fake_denoise_all,clean_raw[:,:,2:-2])
                g_loss+=lambda3*denoiser_loss(torch.mul(fake_denoise_all,mask),torch.mul(clean_raw[:,:,2:-2],mask))
                G_losses.append(g_loss.item())
                g_loss.backward()

                # gradient clip
                # 有trainable_parameters的是rnn模型，不需要clip梯度也不会爆
                # 用不用trainable_parameters算norm都不影响，光流的部分都不包括梯度，不会参与计算
                if hasattr(model,'trainable_parameters'):
                    for item in model.trainable_parameters():
                        nn.utils.clip_grad_norm_(item, max_norm=1, norm_type=2)
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=2, norm_type=2)

                optimizer_G.step()

                train_psnr=torch.mean(datah.batch_psnr(fake_denoise_all,clean_raw[:,:,2:-2])).item()

            if i % args.show_every==0:
                write_logs(
                    "[Epoch %d/%d] [Batch %d] [D loss: %f] [G loss: %f] [PSNR: %f] [LR: %f]"
                    % (epoch, args.num_iter, i, d_loss if isinstance(d_loss,float) else d_loss.item(), \
                    g_loss.item(), train_psnr, optimizer_G.param_groups[0]['lr'])
                )
                print(
                    "[Epoch %d/%d] [Batch %d] [D loss: %f] [G loss: %f] [PSNR: %f] [LR: %f]"
                    % (epoch, args.num_iter, i, d_loss if isinstance(d_loss,float) else d_loss.item(), \
                    g_loss.item(), train_psnr, optimizer_G.param_groups[0]['lr'])
                )

        scheduler_G.step()

        if 'real_validity' in dir():
            if (not real_validity==None) and (not fake_validity==None):
                real_list.append(torch.mean(real_validity).item())
                fake_list.append(torch.mean(fake_validity).item())
        
        if gpu == 0:
            avg_test_loss, test_psnr, out_plt = run_test(gpu, args, test_loader, model, denoiser_loss, generator, keys)
            write_logs('test of epoch %d, avg loss: %.6f, psnr: %.4f'%(epoch,avg_test_loss,test_psnr))
            print('test of epoch %d, avg loss: %.6f, psnr: %.4f'%(epoch,avg_test_loss,test_psnr))
            test_loss_list.append(avg_test_loss)
            test_psnr_list.append(test_psnr)
        
        if epoch%args.save_every == 0 and gpu==0:
            scipy.io.savemat(folder_name + 'losses.mat',
                            {'G_losses':G_losses,
                            'D_losses':D_losses,
                            'real_list':real_list,
                            'fake_list':fake_list,
                            'test_psnr':test_psnr_list})

            best_test_loss=save_checkpoint(folder_name, epoch, test_loss_list, best_test_loss, 
                                           test_psnr, model, out_plt,discriminators=None if epoch<discriminator_epoch else discriminators)
            # 每20epoch清一次缓存
            torch.cuda.empty_cache()
        

def eval(gpu, args):

    # 以后可以再考虑下在去噪模型中用更多噪声合成模型的知识，所以这个接口先保留
    print('loading generator', gpu)
    generator = load_generator_model(args, gpu)
    generator.cuda(gpu)

    # keys=generator.get_keys()#generator中间返回几个结果，就需要几个去噪模型
    keys=['shot','read', 'uniform']#, 'fixed']
    print('keys:',keys)

    print('entering training function')
    # print(args.nr, args.gpus, gpu, args.world_size)
    # Setup for distributed training on multiple GPUs
    rank = args.nr * args.gpus + gpu                         
    dist.init_process_group(                                   
        backend='nccl',                                         
        init_method='env://',                                   
        world_size=args.world_size,                              
        rank=rank)
    
    
    print('loading model')
    # model = dh.load_from_checkpoint(base_folder + chkp_path, best=True, keys=keys, args=args)
    model = get_model_denoiser(args,gpu,keys,res_learn=True)

    torch.cuda.set_device(gpu)
    model=model.cuda()

    model, discriminators, curr_epoch, G_losses, D_losses, real_list, fake_list, test_psnr_list, test_loss, folder_name \
            = resume_from_checkpoint(args, model, gpu,best=False)
    print('resuming from ep ',curr_epoch)

    # for name,param in model.named_parameters():
    #     print(name, param.requires_grad)
    
    criterion = define_loss_denoiser(args, gpu)
    
    # # Wrap the model
    # model = nn.parallel.DistributedDataParallel(model,
    #                                             device_ids=[gpu], find_unused_parameters=True)
    
    
    # Set up dataset
    # dataset_list, dataset_list_test, i0 = get_dataset(args)
    if args.dataset=='CRVD':
        dataset_list, dataset_list_test=datah.get_dataset_CRVD(args,dimension=3,test_mode='all')
    elif args.dataset=='SRVD':
        dataset_list, dataset_list_test=datah.get_dataset_SRVD(args,dimension=3,test_mode='all')
    elif args.dataset=='RNVD':
        dataset_list, dataset_list_test=datah.get_dataset_RNVD(args,dimension=3,test_mode='all')
    elif args.dataset=='DAVISraw':
        if not args.use_syn:
            dataset_list, dataset_list_test=datah.get_dataset_DAVISraw(args,dimension=3)
        else:
            dataset_list, dataset_list_test=datah.get_dataset_DAVISraw_syn(args,dimension=3)
    print('using %s'%args.dataset)

    print('test_length:', dataset_list_test.__len__())
    
    test_loader = torch.utils.data.DataLoader(dataset=dataset_list_test,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True)

    print('Testing')
    avg_test_loss, test_psnr, out_plt = run_test(gpu, args, test_loader, model, criterion, generator, keys)
    print('Avg loss: %.6f, psnr: %.4f'%(avg_test_loss,test_psnr))
    save_name = f'test_loss{avg_test_loss:.5f}_psnr{test_psnr:.5f}.png'
    Image.fromarray((np.clip(out_plt,0,1) * 255).astype(np.uint8)).save(save_name)

    # save_name = f'test_loss{avg_test_loss:.5f}_psnr{test_psnr:.5f}_gt.png'
    # Image.fromarray((np.clip(out_plt1,0,1) * 255).astype(np.uint8)).save(save_name)

    # for key in keys:
    #     save_name = f'test_loss{avg_test_loss:.5f}_psnr{test_psnr:.5f}_{key}_gt.png'
    #     Image.fromarray((np.clip(out_inter[key],0,1) * 255).astype(np.uint8)).save(save_name)

if __name__ == '__main__':
    main()