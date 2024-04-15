import argparse, json, glob, os
import torch
import scipy
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, widgets, fixed
import utils.noisegen_helper_fun as gh
from models.gan_noisemodel import NoiseGenerator2d_distributed_ablation,NoiseGenerator2d_v3
from PIL import Image
import cv2
from utils.isp import ISP


def denoise_shuffle(a,b=None):
    # a,b:(b,c,h,w)
    B,C,H,W=a.shape
    idx=torch.randperm(B)
    a=a[idx,...]
    if b==None:
        return a
    if b!=None:
        b=b[idx,...]
        return a,b
    

def raw2rgb_crvd(data):
    # data=(data*(2**12-1-240)+240)/2**12
    # print(data.max(),data.min())
    data=torch.stack([data[:,0],data[:,1],data[:,3],data[:,2]],dim=1)
    isp = ISP().cuda()
    state_dict=torch.load('/data3/mxx/compare_models/Baseline_new/tools/isp.pth')['state_dict']
    isp.load_state_dict(state_dict)
    # isp=torch.load('../utils/isp/ISP_CNN.pth')
    data_rgb=isp(data)
    data_rgb=np.clip(data_rgb.cpu().detach().numpy()[0].transpose(1,2,0),0,1)*255
    return cv2.cvtColor(data_rgb,cv2.COLOR_BGR2RGB)


def unpack_raw(im):
    h, w, _ = im.shape
    H, W = h * 2, w * 2
    img2 = np.zeros((H, W))
    img2[0:H:2, 0:W:2] = im[:, :, 0]
    img2[0:H:2, 1:W:2] = im[:, :, 1]
    img2[1:H:2, 0:W:2] = im[:, :, 2]
    img2[1:H:2, 1:W:2] = im[:, :, 3]
    return img2


def raw2rgb_rnvd(data):
    data_raw=data.cpu().detach().numpy()[0].transpose(1,2,0)
    data_raw2=(np.clip(unpack_raw(data_raw),0,1)*255).astype(np.uint8)
    bgr = cv2.cvtColor(data_raw2, cv2.COLOR_BayerRG2RGB_EA)
    return cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)


############下面是denoiser的模型加载和保存
def load_generator_model(args,gpu,res_learn=True):
    folder_name=args.generator_path
    dataset=args.dataset
    print('loading generator from checkpoint')
    parser = argparse.ArgumentParser(description='Process some integers.')
    args = parser.parse_args('')
    with open(folder_name + '/args.txt', 'r') as f:
        args.__dict__ = json.load(f)
        args.fraction_video = 50
        args.resume_from_checkpoint = folder_name

    if dataset=='CRVD':
        shape=(4,1080,1920)#2,073,600
        noise_levels=5
    elif dataset=='RNVD':
        shape=(4,1536,2048)#3,145,728
        noise_levels=4
    elif dataset=='SRVD' or dataset=='DAVISraw':
        shape=(4,1080,1920)#2,073,600
        noise_levels=5

        # 第二个阶段有dynamic，且加载预训练的模型
    # generator = NoiseGenerator2d_distributed_ablation(net = None, noise_list = args.noiselist, 
    #                                         device = gpu,res_learn=False, dynamic=False)
    generator = NoiseGenerator2d_v3(net = None, noise_list = args.noiselist, 
                                            device = gpu,res_learn=res_learn,dynamic=False, noise_levels=noise_levels)
    generator = gh.load_from_checkpoint_ab(generator, folder_name, gpu)
    return generator#.cuda(gpu)


def load_from_checkpoint(folder_name, best = True, keys=[], args=[]):
    device = 'cuda:0'
    print('loading from checkpoint')

    from models.stage_denoiser import Stage_denoise3
    model = Stage_denoise3(n_channel_in=4,res_learn=True,
                keys=keys[::-1],t_length=args.t_length)
        
    if best == True:
        list_of_files = glob.glob(folder_name + '/best*.pt') # * means all if need specific format then *.csv
    else:
        list_of_files = glob.glob(folder_name + '/checkpoint*.pt') # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    path = latest_file
    
    saved_state_dict = torch.load(path, map_location = device)
    
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
        
    model.load_state_dict(saved_state_dict)#,strict=False)
    
    if best == True:
        curr_epoch = int(path.split('/')[-1].split('_')[0].split('best')[1])
    else:
        curr_epoch = int(path.split('/')[-1].split('_')[0].split('checkpoint')[1])
    # loaded = scipy.io.loadmat(args.resume_from_checkpoint + '/losses.mat')
    # folder_name = args.resume_from_checkpoint + '/'
    print('resuming from checkoint, epoch:', curr_epoch)
    
    return model


def get_model_denoiser(args,gpu,keys,res_learn):
# Define the denoising model
    # print('inv keys',keys[::-1])
    if args.network == 'Unet3D':
        # 先用unet_3d做实验
        from models.stage_denoiser import Stage_denoise3
        # from models.stage_denoiser_recurr import Stage_denoise32
        keys=['shot','read', 'uniform']
        res_opt = bool(args.unet_opts.split('_')[0].split('residual')[-1])
        input_mask=args.input_mask if args.stage in ['pretrain_clean','pretrain_noisy','mask_noisy2clean'] else None
        model = Stage_denoise3(n_channel_in=4, device=gpu,
                    residual=res_opt, 
                    down=args.unet_opts.split('_')[1], 
                    up=args.unet_opts.split('_')[2], 
                    activation=args.unet_opts.split('_')[3],
                    keys=keys[::-1],res_learn=res_learn,
                    noise_cat=args.noise_cat,t_length=args.t_length,
                    input_mask=input_mask,mask_ratio=args.mask_ratio)

        # from models.stage_denoiser import Stage_denoise4
        # # from models.stage_denoiser_recurr import Stage_denoise32
        # res_opt = bool(args.unet_opts.split('_')[0].split('residual')[-1])
        # model = Stage_denoise4(n_channel_in=4, device=gpu,
        #             residual=res_opt, 
        #             down=args.unet_opts.split('_')[1], 
        #             up=args.unet_opts.split('_')[2], 
        #             activation=args.unet_opts.split('_')[3],
        #             keys=keys[::-1],res_learn=res_learn,
        #             noise_cat=args.noise_cat,t_length=args.t_length)
    else:
        print('Error, invalid network')
        
    return model

# Load in a pretrained model from a checkpoint
def preload_model(args, model, device):
    print(args.preloaded)
    list_of_files = glob.glob(args.preloaded + '/checkpoint*.pt')
    latest_file = max(list_of_files, key=os.path.getctime)
    path = latest_file
   
    saved_state_dict = torch.load(path, map_location = 'cuda:'+str(device))
    
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
        
    model.load_state_dict(saved_state_dict)

    curr_epoch = int(path.split('/')[-1].split('_')[0].split('checkpoint')[1])
    print('resuming from preloaded, epoch:', curr_epoch)
    return model

# Resume training from a checkpoint
def resume_from_checkpoint(args, model, device, best=True, discriminators=None,load_discriminators=False,strict=True):
    if best == True:
        list_of_files = glob.glob(args.resume_from_checkpoint + '/best*.pt') # * means all if need specific format then *.csv
    else:
        list_of_files = glob.glob(args.resume_from_checkpoint + '/checkpoint*.pt') # * means all if need specific format then *.csv
    # list_of_files = glob.glob(args.resume_from_checkpoint + '/checkpoint*.pt')
    latest_file = max(list_of_files, key=os.path.getctime)
    path = latest_file
    test_loss=float(path.split('/')[-1].split('_')[-2].replace('loss',''))    
    split_str='best' if best==True else 'checkpoint'
    curr_epoch = int(path.split('/')[-1].split('_')[0].split(split_str)[1])
    model=load_state_dict_ddp(model,path,device,strict=strict)
    if load_discriminators:
        for key in discriminators:
            basedir=os.path.join(args.resume_from_checkpoint,f'./ckpt{curr_epoch}_discriminators')
            discriminators[key]=load_state_dict_ddp(discriminators[key],os.path.join(basedir,f'ep{curr_epoch}_{key}.pt'),device,strict=strict)
            print('loaded discriminators %s'%key)

    loaded = scipy.io.loadmat(args.resume_from_checkpoint + '/losses.mat')
    G_losses = list(loaded['G_losses'][0])
    
    D_losses = loaded['D_losses']
    # real_list = list(loaded['real_list'][0])
    # fake_list = list(loaded['fake_list'][0])
    D_losses=[];real_list=[];fake_list=[]

    test_psnr_list = list(loaded['test_psnr'][0])

    folder_name = args.resume_from_checkpoint + '/'
    return model, discriminators,curr_epoch, G_losses, D_losses, real_list, fake_list, test_psnr_list, test_loss, folder_name


def load_state_dict_ddp(model,path,device,strict=True):
    saved_state_dict = torch.load(path, map_location = 'cuda:'+str(device))
    
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
            # if not 'head' in name:
            new_state_dict[name] = v
        saved_state_dict = new_state_dict

        # # import re
        # new_state_dict = OrderedDict()
        # for k, v in saved_state_dict.items():
        #     name=k.replace('basemodel_shot.layer_1','basemodel_shot.layer_2')
        #     new_state_dict[name] = v
        #     if k.find('layer_1'):
        #         new_state_dict[k]=torch.zeros_like(v)
        # print('changing shot.layer1 to shot.layer2')
        # saved_state_dict = new_state_dict

    model.load_state_dict(saved_state_dict,strict=strict)
    return model


def save_checkpoint(folder_name, ep, test_loss_list, best_test_loss, test_psnr, 
                    model, out_plt,discriminators=None):
    
    print('saving checkpoint')
    checkpoint_name = folder_name + f'checkpoint{ep}_test_loss{test_loss_list[-1]:.5f}_psnr{test_psnr:.5f}.pt'
    torch.save(model.state_dict(), checkpoint_name)

    if not discriminators==None:
        save_basedir=folder_name + f'ckpt{ep}_discriminators'
        if not os.path.exists(save_basedir):
            os.mkdir(save_basedir)
        for key in discriminators:
            torch.save(discriminators[key].state_dict(),os.path.join(save_basedir,f'ep{ep}_{key}.pt'))

    save_name = folder_name + f'testimage{ep}_test_loss{test_loss_list[-1]:.5f}_psnr{test_psnr:.5f}.jpg'
    Image.fromarray((np.clip(out_plt,0,1) * 255).astype(np.uint8)).save(save_name)

    if best_test_loss > test_loss_list[-1]:
        print('best loss', test_loss_list[-1])
        best_test_loss = test_loss_list[-1]
        checkpoint_name = folder_name + f'best{ep}_test_loss{test_loss_list[-1]:.5f}_psnr{test_psnr:.5f}.pt'
        torch.save(model.state_dict(), checkpoint_name)
        save_name = folder_name + f'best{ep}_test_loss{test_loss_list[-1]:.5f}_psnr{test_psnr:.5f}.jpg'
        Image.fromarray((np.clip(out_plt,0,1) * 255).astype(np.uint8)).save(save_name)

    return best_test_loss

    