import sys, os, glob
os.environ['CUDA_VISIBLE_DEVICES']='5,6,7'

sys.path.append("..")
import numpy as np
import torch
from torch.optim import Adam,lr_scheduler
from PIL import Image
import argparse, json
import scipy.io
import datasets.crvd_supervised_dataset as dset
import utils.noisegen_helper_fun as gh
import datasets.dataset_helper as datah
from utils.losses import define_loss_denoiser
from utils.denoiser_helper_fun import load_generator_model,\
get_model_denoiser,resume_from_checkpoint,preload_model,\
save_checkpoint

import torch.distributed as dist

import torch.multiprocessing as mp
import torch.nn as nn
import time

    
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
    parser.add_argument('--mode', default='train', help = 'train or eval')
    parser.add_argument('--network', default='Unet3D', help='Choose which network to use')
    parser.add_argument('--noise_cat',default=False)
    parser.add_argument('--noise_type', default='unetfourier', help='Choose which noise model to load') 
    
    parser.add_argument('--loss', default= 'MSE',#'L1_LP2D', 
                        help = 'Choose loss or a combination of losses. Options: MSE, L1, TV, LP2D (LPIPS),\
                        angular (Cosine similarity between the 4 color channels)')

    parser.add_argument('--dataset', default='CRVD', help = 'Choose which dataset to use. Options: gray, color')
    parser.add_argument('--multiply', default= 'None', 
                        help = 'Choose what sort of processing to do on the ground truth images. Options: None, \
                        histeq (does histogram equalization on the ground truth images before taking the loss), \
                        gamma (goes gamma correction on the ground truth images, makes the network learn \
                        denoising + gamma correction')
    parser.add_argument('--space', default= 'linear') # 
    parser.add_argument('--notes', default= 'Test') 
    parser.add_argument('--t_length', default=8, type = int) 
    parser.add_argument('--crop_size', default= 256, 
                        help='Choose the image patch size for denoising. \
                        Options: 512, (512x512), full (1024x512), small (128x128), or 256 (256x256)')  
    parser.add_argument('--unet_opts', default='residualFalse_conv_tconv_selu')
    parser.add_argument('--num_iter', default=200)
    parser.add_argument('--preloaded', default = False, 
                        help='Use pretrained model. Specify the file path here to load in the model')
    parser.add_argument('--generator_path', default = '../train_results/on_CRVD/noisemodel_crvd_v7',#
                        help='Preload Generator path for noise generate')
    parser.add_argument('--resume_from_checkpoint', default = False,#'../saved_models/crvd_syn_blind',
                        help='Resume training from a saved checkpoint. Specify the filepath here')
    parser.add_argument('--learning_rate', default = 0.0002, type=float)
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

    parser.add_argument('--batch_size', default = 8, type=int)
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
                        help='Choose save frequency. Save every N epochs. ')
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=3, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, 
                        metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--milestone',default=[120,140,160,180])
    parser.add_argument('--co_training', default=False, help='if train with optical flow')
    parser.add_argument('--use_syn', default=False, help='if use synthetic noise for training')
    parser.add_argument('--res_learn', default=False, help='use noise or noisy for supervision')
    parser.add_argument('--save_for_srga',default=False)#保存feat
    parser.add_argument('--save_inter',default=True)#保存每个结果生成的img

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

    # # Resume training from a saved checkpoint. 
    # if args.resume_from_checkpoint:
    #     print('resuming from checkpoint')
    #     print('checkpoint filepath:', args.resume_from_checkpoint)
    #     folder_name = args.resume_from_checkpoint
    #     parser = argparse.ArgumentParser(description='Process some integers.')
    #     args1 = parser.parse_args('')
    #     with open(args.resume_from_checkpoint + '/args.txt', 'r') as f:
    #         args1.__dict__ = json.load(f)
    #         args1.resume_from_checkpoint = folder_name
    #     if 'n' not in args1:
    #         args1.nodes = args.nodes
            # args1.gpus = args.gpus
            # args1.nr = args.nr
            # args1.world_size = args.gpus * args.nodes
        #     args1.batch_size = args.batch_size
        # if args1.data == 'video_combined_new':
        #     args1.data = 'stills_realvideo_MOTvideo'
        # elif args1.data == 'video_real':
        #     args1.data = 'stills_realvideo'
            
        # args1.preloaded = False
        
        # if 'crop_size' not in args1:
        #     args1.crop_size = 'full'
        
        # args = args1
        
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
            net_input = sample['noisy_input'].cuda(non_blocking=True)
            pos=sample['pos']   # 裁剪位置
            noise_level=sample['noise_level']

            # noisy_all,noise_inter=generator(gh.t32_1(gt_label),pos=pos)
            # noisy_all=gh.t23_1(noisy_all,args.t_length)#(b*t,c,h,w)->(b,c,t,h,w)
            noise_levels=generator.get_noise_level(noise_ind=noise_level)#每个seq一个noise_level

            # 用真实的噪声测试
            # denoise_all,denoise_inter = model(net_input,pos=pos)
            denoise_all,denoise_inter,predicted_noise,feat_inter = model(net_input,gt_label,noise_level=noise_levels)
            # 用合成的噪声测试
            # denoise_all,denoise_inter = model(noisy_all,pos=pos)
            
            # test loss暂时只算最终的
            test_loss = criterion(denoise_all, gt_label[:,:,2:-2])
            # # loss = criterion(net_output, gt_label)
            # test_loss=0
            # for key in keys:
            #     # 这里之后可能要换成gan loss
            #     if key == keys[-1]:
            #         # keys是合成噪声阶段合成的最后一个图，即加入所有噪声的图，和noisy_all对应
            #         # 也就是denoiser的输入，这里不需要施加约束
            #         pass
            #     # 所有中间去噪结果和中间噪声生成的loss
            #     loss+=criterion(denoise_inter[key],noisy_inter[key])
            # # 去噪最终结果和原始干净图的loss
            # loss+=criterion(denoise_all,gt_label)
            avg_test_loss += test_loss.item()

            # 第一种psnr算法
            cur_test_psnr=torch.mean(datah.batch_psnr(denoise_all,gt_label[:,:,2:-2])).item()
            cur_test_ssim=datah.raw_ssim(denoise_all,gt_label[:,:,2:-2])
            avg_test_psnr += cur_test_psnr
            avg_test_ssim += cur_test_ssim

            
        avg_test_loss = avg_test_loss/(i+1)
        avg_test_psnr = avg_test_psnr/(i+1)
        avg_test_ssim = avg_test_ssim/(i+1)
        print('avg test ssim: ',avg_test_ssim)
        
        out_plt = denoise_all.cpu().detach().numpy()[0,:,0].transpose(1,2,0)[...,0:3]

        
    return avg_test_loss, avg_test_psnr, out_plt


def write_logs(string,log_dir='log.txt'):
    with open(log_dir,'a') as f:
        f.write(string+'\n')


# Main training function 
def train(gpu, args):

    # print(args.nr, args.gpus, gpu, args.world_size)
    # Setup for distributed training on multiple GPUs
    write_logs('entering training function')
    print('entering training function')
    rank = args.nr * args.gpus + gpu                         
    dist.init_process_group(                                   
        backend='nccl',                                         
        init_method='env://',                                   
        world_size=args.world_size,                              
        rank=rank)

    write_logs('loading model')
    print('loading model')
    generator = load_generator_model(args, gpu)

    # keys=generator.get_keys()#generator中间返回几个结果，就需要几个去噪模型
    keys=['shot','read', 'uniform']#, 'fixed']
    write_logs('keys: %s'%keys)
    print('keys:',keys)    
    
    print('loading model')
    model = get_model_denoiser(args,gpu,keys,args.res_learn)
    
    write_logs('put on GPU %d'%gpu)
    print('put on GPU', gpu)
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    generator.cuda(gpu)
    
    
    if args.preloaded:
        model = preload_model(args, model, gpu)
    
    if args.resume_from_checkpoint:
        write_logs('resuming model from checkpoint')
        print('resuming model from checkpoint')
        model, _,curr_epoch, G_losses, D_losses, real_list, fake_list, test_psnr_list, test_loss, folder_name \
            = resume_from_checkpoint(args, model, gpu, best=False, discriminators=None,load_discriminators=False,strict=True)
        
        train_loss=[]
        best_test_loss = test_loss
        test_loss_list=[test_loss]
        write_logs('best loss is: %f'%best_test_loss)
        print('best loss is: ', best_test_loss)
    else:
        curr_epoch = 0; train_loss = []; test_loss_list = []; best_test_loss = 1e9;  test_psnr_list=[]; G_losses=[]
        fake_list=[]; real_list=[]
        folder_name = args.folder_name
        
    
    batch_size = args.batch_size
    
    # -----------------
    # losses and optimizers
    criterion = define_loss_denoiser(args, gpu)
    
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

    for name,param in model.named_parameters():
        print(name, param.requires_grad)
    
    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model,
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
    
    D_losses = {}
    for key in keys[:-1]:
        D_losses[key]=[]

    total_step = len(train_loader)
    print('Enter training loop')
    for epoch in range(curr_epoch, args.num_iter):
        # noisy_all、noisy_inter分别表示合成噪声模型的最终合成结果和中间阶段
        # denoise_all、denoise_inter分别表示去噪模型的最终结果和中间阶段
        # 两个阶段的key对应的是相同的阶段
        avg_loss = 0
        for i, sample in enumerate(train_loader):
            start = time.time()

            # 真实的带噪图片
            B,C,T,H,W=sample['noisy_input'].shape
            noisy_raw = sample['noisy_input'].cuda(non_blocking=True)
            clean_raw = sample['gt_label_nobias'].cuda(non_blocking=True)#(b,c,t,h,w)
            noise_level=sample['noise_level']
            pos=sample['pos']
            # 从generator处获得中间加噪声的结果
            with torch.no_grad():
                # gh.t32_1(gt_label)：(b,c,t,h,w)->(b*t,c,h,w)
                noisy_all,noise_inter=generator(gh.t32_1(clean_raw),pos=pos,noise_level=noise_level.unsqueeze(1).repeat(1,T))
                noisy_all=gh.t23_1(noisy_all,args.t_length)#(b*t,c,h,w)->(b,c,t,h,w)

                noise_levels=generator.get_noise_level(noise_ind=noise_level)#每个seq一个noise_level
                for key in keys:
                    noise_inter[key] = gh.t23_1(noise_inter[key],args.t_length)


            # 真实的带噪图片
            noisy_all_label = sample['noisy_input'].cuda(non_blocking=True)

            # 倒序和合成噪声的每个加噪结果形成监督
            
            # denoise_all,denoise_inter = model(noisy_all_label)
            # 直接用合成的噪声来训练，先mse（如果用真实噪声训练要改成gan loss）
            gen_denoise_all,gen_denoise_inter,_,_ = model(noisy_all,clean_raw,noise_level=noise_levels)

            # loss = criterion(net_output, gt_label)
            g_loss=0
            for key in keys[:-1]:
                # 所有中间去噪结果和中间噪声生成的loss
                loss_key=criterion(gen_denoise_inter[key][:,:,2:-2],noise_inter[key][:,:,2:-2])
                D_losses[key].append(loss_key.item())
                g_loss+=loss_key
            # 去噪最终结果和原始干净图的loss
            g_loss+=criterion(gen_denoise_all,clean_raw[:,:,2:-2])
            
            # Backward and optimize
            start = time.time()
            optimizer_G.zero_grad()
            G_losses.append(g_loss.item())

            g_loss.backward()
            optimizer_G.step()
            
            avg_loss+=g_loss.item()
            if i % args.show_every==0:
                train_psnr=torch.mean(datah.batch_psnr(gen_denoise_all,clean_raw[:,:,2:-2])).item()
                write_logs(
                    "[Epoch %d/%d] [Batch %d] [G loss: %f] [PSNR: %f] [LR: %f]"
                    % (epoch, args.num_iter, i, g_loss.item(), train_psnr, optimizer_G.param_groups[0]['lr'])
                )
                print(
                    "[Epoch %d/%d] [Batch %d] [G loss: %f] [PSNR: %f] [LR: %f]"
                    % (epoch, args.num_iter, i, g_loss.item(), train_psnr, optimizer_G.param_groups[0]['lr'])
                )

        train_loss.append(avg_loss/i)
        scheduler_G.step()

        if gpu==0:
            avg_test_loss, test_psnr, out_plt = run_test(gpu, args, test_loader, model, criterion, generator, keys)
            write_logs('test of epoch %d, avg loss: %.6f, psnr: %.4f'%(epoch,avg_test_loss,test_psnr))
            print('test of epoch %d, avg loss: %.6f, psnr: %.4f'%(epoch,avg_test_loss,test_psnr))
            test_loss_list.append(avg_test_loss)
            test_psnr_list.append(test_psnr)

        if epoch%args.save_every == 0 and gpu == 0:
            scipy.io.savemat(folder_name + 'losses.mat',
                            {'G_losses':G_losses,
                            'D_losses':D_losses,
                            'real_list':real_list,
                            'fake_list':fake_list,
                            'test_psnr':test_psnr_list})
            
            best_test_loss=save_checkpoint(folder_name, epoch, test_loss_list, best_test_loss, 
                                           test_psnr, model, out_plt,discriminators=None)
            # 每20epoch清一次缓存
            torch.cuda.empty_cache()
        

def eval(gpu, args):

    print('loading generator', gpu)
    generator = load_generator_model(args, gpu)
    generator.cuda(gpu)

    # keys=generator.get_keys()#generator中间返回几个结果，就需要几个去噪模型
    keys=['shot_read', 'uniform', 'row', 'fixed']
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
    import Starlight_ours.utils.denoiser_helper_fun as dh
    base_folder='../saved_models/'
    # chkp_path ='denoiser_1stage'
    chkp_path='/Unet3D_unetfourier_MSE_stills_realvideo_color_None_linear'
    args, model = dh.load_from_checkpoint(base_folder + chkp_path, keys=keys)
    torch.cuda.set_device(gpu)
    model=model.cuda()        
    
    batch_size = args.batch_size
    
    criterion = define_loss_denoiser(args, gpu)
    
    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[gpu], find_unused_parameters=True)
    
    
    # Set up dataset
    # dataset_list, dataset_list_test, i0 = get_dataset(args)
    dataset_list, dataset_list_test=datah.get_dataset_CRVD(args)
    
    test_loader = torch.utils.data.DataLoader(dataset=dataset_list_test, 
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True)

    print('Testing')
    avg_test_loss, test_psnr, out_plt = run_test(gpu, args, test_loader, model, criterion, generator, keys)
    print('Avg loss: %.6f, psnr: %.4f'%(avg_test_loss,test_psnr))

if __name__ == '__main__':
    main()