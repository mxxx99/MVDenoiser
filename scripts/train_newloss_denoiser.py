import sys, os, glob
os.environ['CUDA_VISIBLE_DEVICES']='6,7'

sys.path.append("..")
import numpy as np
import torch
from torch.optim import Adam
from PIL import Image
import argparse, json
import scipy.io
import datasets.crvd_supervised_dataset as dset
import utils.noisegen_helper_fun as gh
import datasets.dataset_helper as datah
from utils.losses import define_loss_denoiser
from utils.denoiser_helper_fun import load_generator_model,\
get_model_denoiser,resume_from_checkpoint,preload_model,\
save_checkpoint,denoise_shuffle
from utils.losses import define_loss

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
    parser.add_argument('--noise_type', default='unetfourier', help='Choose which noise model to load')

    parser.add_argument('--loss', default= 'MSE',#'L1_LP2D', 
                        help = 'Choose loss or a combination of losses. Options: MSE, L1, TV, LP2D (LPIPS),\
                        angular (Cosine similarity between the 4 color channels)')
    parser.add_argument('--discriminator_loss', default='fourier', 
                        help = 'Choose generator loss. Options: mixed, fourier, real, mean')
    parser.add_argument('--generator_loss', default='lpips', help = 'Choose generator loss. Default: lpips') 

    parser.add_argument('--data', default= 'stills_realvideo', help = 'Choose which data to use during training. \
                         Options: stills, realvideo, MOTvideo')
    parser.add_argument('--dataset', default='color', help = 'Choose which dataset to use. Options: gray, color')
    parser.add_argument('--multiply', default= 'None', 
                        help = 'Choose what sort of processing to do on the ground truth images. Options: None, \
                        histeq (does histogram equalization on the ground truth images before taking the loss), \
                        gamma (goes gamma correction on the ground truth images, makes the network learn \
                        denoising + gamma correction')
    parser.add_argument('--space', default= 'linear') # 
    parser.add_argument('--notes', default= 'Test') 
    parser.add_argument('--t_length', default=16, type = int) 
    parser.add_argument('--crop_size', default= 256, 
                        help='Choose the image patch size for denoising. \
                        Options: 512, (512x512), full (1024x512), small (128x128), or 256 (256x256)')  
    parser.add_argument('--unet_opts', default='residualFalse_conv_tconv_selu')
    parser.add_argument('--num_iter', default= 540)
    parser.add_argument('--preloaded', default = False, 
                        help='Use pretrained model. Specify the file path here to load in the model')
    parser.add_argument('--generator_path', default = '../saved_models/noisemodel_v3_alllevel', 
                        help='Preload Generator path for noise generate')
    parser.add_argument('--resume_from_checkpoint', default = '../saved_models/denoiser_multistage_retrain',#False,
                        help='Resume training from a saved checkpoint. Specify the filepath here')
    parser.add_argument('--learning_rate', default = 0.0002, type=float)
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

    parser.add_argument('--batch_size', default = 4, type=int)
    parser.add_argument('--save_path', default = '../saved_models/', help='Specify where to save checkpoints during training')
    parser.add_argument('--MOT_path', default = '../data/MOTfiles_raw/',
                        help='If using unprocessed MOT images during training, specify the filepath \
                        for where your MOT dataset is stored')
    parser.add_argument('--stills_path', default = '../data/paired_data/stillpairs_mat', 
                        help='Specify the filepath for the stills dataset.')
    parser.add_argument('--cleanvideo_path', default = '../data/RGBNIR_videodataset_mat/',
                        help='Specify the filepath for the clean video dataset.')
    parser.add_argument('--show_every', default= 40, help='Choose show frequency. show every N batches') 
    parser.add_argument('--save_every', default = 10, type=int, 
                        help='Choose save frequency. Save every N epochs.')
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=2, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, 
                        metavar='N',
                        help='number of total epochs to run')

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

    # Resume training from a saved checkpoint. 
    if args.resume_from_checkpoint:
        print('resuming from checkpoint')
        print('checkpoint filepath:', args.resume_from_checkpoint)
        folder_name = args.resume_from_checkpoint
        parser = argparse.ArgumentParser(description='Process some integers.')
        args1 = parser.parse_args('')
        with open(args.resume_from_checkpoint + '/args.txt', 'r') as f:
            args1.__dict__ = json.load(f)
            args1.resume_from_checkpoint = folder_name
        if 'n' not in args1:
            args1.nodes = args.nodes
            args1.gpus = args.gpus
            args1.nr = args.nr
            args1.world_size = args.gpus * args.nodes
            args1.batch_size = args.batch_size
        if args1.data == 'video_combined_new':
            args1.data = 'stills_realvideo_MOTvideo'
        elif args1.data == 'video_real':
            args1.data = 'stills_realvideo'
            
        args1.preloaded = False
        
        if 'crop_size' not in args1:
            args1.crop_size = 'full'
        
        args = args1
        
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
        for i, sample in enumerate(test_loader): #Loop through dataset
            gt_label = sample['gt_label_nobias'].cuda(non_blocking=True)#(b,c,t,h,w)
            net_input = sample['noisy_input'].cuda(non_blocking=True)
            pos=sample['pos']   # 裁剪位置

            denoise_all,denoise_inter = model(net_input,pos=pos)
            
            # test loss暂时只算最终的
            test_loss = criterion(denoise_all, gt_label)

            avg_test_loss += test_loss.item()
            # 第一种psnr算法
            avg_test_psnr += torch.mean(datah.batch_psnr(denoise_all,gt_label)).item()
            # 第二种psnr算法
            # avg_test_psnr+=datah.batch_psnr(denoise_all,gt_label)

        avg_test_loss = avg_test_loss/(i+1)
        avg_test_psnr = avg_test_psnr/(i+1)
        
        # out_plt1 = gt_label.cpu().detach().numpy()[0,:,8].transpose(1,2,0)[...,0:3]
        out_plt = denoise_all.cpu().detach().numpy()[0,:,8].transpose(1,2,0)[...,0:3]

        
    return avg_test_loss, avg_test_psnr, out_plt#,out_plt1

def write_logs(string,log_dir='log.txt'):
    with open(log_dir,'a') as f:
        f.write(string+'\n')
    
# Main training function 
def train(gpu, args):
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
    # model：去噪模型；generator：合成噪声模型
    write_logs('loading generator %d'%gpu)
    print('loading generator', gpu)
    generator = load_generator_model(args, gpu)

    # keys=generator.get_keys()#generator中间返回几个结果，就需要几个去噪模型
    keys=['shot','read', 'uniform']#, 'fixed']
    write_logs('keys: %s'%keys)
    print('keys:',keys)

    write_logs('loading model')
    print('loading model')
    model = get_model_denoiser(args,keys)
    
    write_logs('put on GPU %d'%gpu)
    print('put on GPU', gpu)
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    generator.cuda(gpu)
    
    # if args.preloaded:
    #     model = preload_model(args, model, gpu)
    
    if args.resume_from_checkpoint:
        write_logs('resuming model from checkpoint')
        print('resuming model from checkpoint')
        model, curr_epoch, G_losses, D_losses, real_list, fake_list, test_psnr_list, test_loss, folder_name \
            = resume_from_checkpoint(args, model, gpu)
            
        best_test_loss = test_loss
        test_loss_list=[test_loss]
        write_logs('best loss is: %f'%best_test_loss)
        print('best loss is: ', best_test_loss)
    else:
        curr_epoch = 0; test_loss_list = []; best_test_loss = 1e9
        folder_name = args.folder_name
        
    
    batch_size = args.batch_size

    denoiser_loss = define_loss_denoiser(args, gpu)
    optimizer_G = Adam(model.parameters(), lr=args.learning_rate)

    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[gpu], find_unused_parameters=True)

    # Set up dataset
    # dataset_list, dataset_list_test, i0 = get_dataset(args)
    dataset_list, dataset_list_test=datah.get_dataset_CRVD(args,dimension=3,test_mode='all')
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
                                               batch_size=1,#batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=test_sampler)
    
    G_losses = []
    test_psnr_list=[]
    train_loss = []

    write_logs('Enter training loop')
    print('Enter training loop')
    for epoch in range(curr_epoch, args.num_iter):
        # noisy_all、noisy_inter分别表示合成噪声模型的最终合成结果和中间阶段
        # denoise_all、denoise_inter分别表示去噪模型的最终结果和中间阶段
        # 两个阶段的key对应的是相同的阶段
        train_sampler.set_epoch(epoch)
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
                noisy_all,noisy_inter=generator(gh.t32_1(clean_raw),pos=pos,noise_level=noise_level.unsqueeze(1).repeat(1,T))
                #(b*t,c,h,w)->(b,c,t,h,w)
                for key in keys:
                    noisy_inter[key] = gh.t23_1(noisy_inter[key],t_length=T)


            # 真实噪声图训练
            optimizer_G.zero_grad()

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            fake_denoise_all,fake_denoise_inter = model(noisy_raw,pos=pos)

            # 在这里加loss
            g_loss=0.0
            for idx, key in enumerate(keys[:-1]):
                g_loss+=denoiser_loss(fake_denoise_inter[key],noisy_inter[key])
                # 下面是根据师兄的意见加的一个额外的约束项
                denoise_after=fake_denoise_inter[keys[idx-1]] if idx>=1 else clean_raw
                noisy_cur=noisy_inter[key]-(noisy_inter[keys[idx-1]] if idx>=1 else clean_raw)
                denoised_cur,_,_=model.module.denoise_stage(denoise_after+noisy_cur,key)
                g_loss+=denoiser_loss(denoise_after,denoised_cur)
                
            
            g_loss+=denoiser_loss(fake_denoise_all,clean_raw)
            G_losses.append(g_loss.item())

            g_loss.backward()
            optimizer_G.step()

            train_psnr=torch.mean(datah.batch_psnr(fake_denoise_all,clean_raw)).item()

        avg_loss+=g_loss.item()
        train_loss.append(avg_loss/i)

        if i % args.show_every:
            write_logs(
                "[Epoch %d/%d] [Batch %d] [G loss: %f] [Avg loss: %f] [PSNR: %f]"
                % (epoch, args.num_iter, i, g_loss.item(), train_loss[-1], train_psnr)
            )
            print(
                "[Epoch %d/%d] [Batch %d] [G loss: %f] [Avg loss: %f] [PSNR: %f]"
                % (epoch, args.num_iter, i, g_loss.item(), train_loss[-1], train_psnr)
            )

        
        if epoch%args.save_every == 0 and gpu == 0:
            avg_test_loss, test_psnr, out_plt = run_test(gpu, args, test_loader, model, denoiser_loss, generator, keys)
            write_logs('test of epoch %d, avg loss: %.6f, psnr: %.4f'%(epoch,avg_test_loss,test_psnr))
            print('test of epoch %d, avg loss: %.6f, psnr: %.4f'%(epoch,avg_test_loss,test_psnr))
            test_loss_list.append(avg_test_loss)
            test_psnr_list.append(test_psnr)

            scipy.io.savemat(folder_name + 'losses.mat',
                            {'G_losses':G_losses,
                            'train_losses':train_loss,
                            'test_losses':test_loss_list,
                            'test_psnr':test_psnr_list})

            best_test_loss=save_checkpoint(folder_name, epoch, test_loss_list, best_test_loss, test_psnr, model, out_plt)
        

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
    import utils.denoiser_helper_fun as dh
    base_folder='../saved_models/'
    chkp_path='denoiser_gan_v2_retrain'
    # chkp_path='denoiser_multistage_compare'
    args, model = dh.load_from_checkpoint(base_folder + chkp_path, best=False, keys=keys)
    torch.cuda.set_device(gpu)
    model=model.cuda()
    
    criterion = define_loss_denoiser(args, gpu)
    
    # # Wrap the model
    # model = nn.parallel.DistributedDataParallel(model,
    #                                             device_ids=[gpu], find_unused_parameters=True)
    
    
    # Set up dataset
    # dataset_list, dataset_list_test, i0 = get_dataset(args)
    dataset_list, dataset_list_test=datah.get_dataset_CRVD(args,dimension=3,test_mode='all')
    
    test_loader = torch.utils.data.DataLoader(dataset=dataset_list_test,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True)

    print('Testing')
    avg_test_loss, test_psnr, out_plt = run_test(gpu, args, test_loader, model, criterion, generator, keys)
    print('Avg loss: %.6f, psnr: %.4f'%(avg_test_loss,test_psnr))
    # save_name = f'test_loss{avg_test_loss:.5f}_psnr{test_psnr:.5f}.jpg'
    # Image.fromarray((np.clip(out_plt,0,1) * 255).astype(np.uint8)).save(save_name)

    # save_name = f'test_loss{avg_test_loss:.5f}_psnr{test_psnr:.5f}_gt.jpg'
    # Image.fromarray((np.clip(out_plt1,0,1) * 255).astype(np.uint8)).save(save_name)

if __name__ == '__main__':
    main()