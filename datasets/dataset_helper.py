from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import MSELoss, L1Loss
from torch.optim import Adam
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity
from datasets.dataset_utils import NoisySyn,NoisySyn_gaussian,NoisySyn_locvar,NoisySyn_eld
import re
import random

import argparse, json, torchvision
import scipy.io
import os
import glob
import math

from pathlib import Path
_script_dir = Path( __file__ ).parent
_root_dir = _script_dir.parent


def raw_ssim(pack1, pack2):
    '''b,c,t,h,w'''
    B,C,T,H,W=pack1.shape
    pack1=pack1.cpu().numpy()
    pack2=pack2.cpu().numpy()
    test_raw_ssim = 0
    for b_ind in range(B):
        for t_ind in range(T):
            for i in range(C):
                test_raw_ssim += structural_similarity(pack1[b_ind,i,t_ind], pack2[b_ind,i,t_ind], data_range=1.0)
    return test_raw_ssim / (C*T*B)


# the same as skimage.metrics.peak_signal_noise_ratio
def batch_psnr(a, b):
    #第一种算法，pytorch psnr
    a = torch.clamp(a, 0, 1)
    b = torch.clamp(b, 0, 1)
    x = torch.mean((a - b) ** 2, dim=[-4, -2, -1])#(b,c,t,h,w)->(b,t)
    return 20 * torch.log(1 / torch.sqrt(x)) / math.log(10)

    # # 第二种算法，用skimage的库，验证两种psnr算出来差不多
    # # calculate psnr the same as RViDeNet
    # B,C,T,H,W=a.shape
    # seq_raw_psnr=0
    # for i in range(T):
    #     seqdn=a[0,:,i].cpu().numpy()
    #     gt=b[0,:,i].cpu().numpy()
    #     seq_raw_psnr += compare_psnr(gt,
    #                                 (np.uint16(seqdn * (2 ** 12 - 1 - 240) + 240).astype(np.float32) - 240) / (2 ** 12 - 1 - 240),
    #                                 data_range=1.0)
    # return seq_raw_psnr/T


def get_dataset_noise_visualization(dataset_arg, filepath_data,iso):
    crop_size = 512

    dataset_list_test = []
        
    if 'crvd' in dataset_arg:
        import datasets.crvd_supervised_dataset as dset
        all_files_test=glob.glob(filepath_data+'/*.tiff')
        # print(all_files_test)
        dataset_test_real = dset.Get_sample_batch_2d(all_files_test,patch_size=crop_size, iso=iso)
        dataset_list_test.append(dataset_test_real)
        
    if len(dataset_list_test)>1:
        dataset_list_test = torch.utils.data.ConcatDataset(tuple(dataset_list_test))
    else:
        dataset_list_test = dataset_list_test[0]
        
    return dataset_list_test


def get_dataset_CRVD(args,dimension=2,test_mode='crop'):
    '''dimension: =2时加载(b,c,h,w)单张图片，=3时加载(b,c,t,h,w)patch
    test_mode: ='crop'时测试集也裁剪，='all'时测试集不裁剪'''
    import datasets.crvd_supervised_dataset as dset
    
    if args.mode=='train':
        train_scene_id_list=[1, 2, 3, 4, 5, 6]
        test_scene_id_list=[7]#,8,9,10,11]
        noise_level=[0,1,2,3,4,5,6,7,8,9]
    elif args.mode=='eval':
        train_scene_id_list=[1,2]
        test_scene_id_list=[7,8,9,10,11]
        noise_level=[0]
    iso_list = [1600, 3200, 6400, 12800, 25600]

    dataset_list = []
    dataset_list_test = []
    test_cropsz=args.crop_size if test_mode=='crop' else None
        
    CRVD_path='/data3/mxx/denoise_dataset/CRVD/CRVD_dataset/'
    for noise in noise_level:
        for iso in iso_list:
            for scene_id in train_scene_id_list:
                filepath_data_train=os.path.join(CRVD_path, 'indoor_raw_noisy/indoor_raw_noisy_scene%d/scene%d/ISO%d' % (scene_id, scene_id, iso))
                # "/data3/mxx/denoise_dataset/CRVD/CRVD_dataset/indoor_raw_noisy/indoor_raw_noisy_scene1/scene1/ISO12800/frame1_noisy0.tiff"
                all_files=glob.glob(filepath_data_train+('/frame?_noisy%d*.tiff'%noise))
                if dimension==3:#读视频patch->(b,c,t,h,w)
                    dataset_train_real = dset.Get_sample_batch(all_files, t_length=args.t_length,
                                                            patch_size=args.crop_size, iso=iso)
                else:
                    dataset_train_real = dset.Get_sample_batch_2d(all_files,
                                                            patch_size=args.crop_size, iso=iso)
                dataset_list.append(dataset_train_real)

            for scene_id in test_scene_id_list:
                filepath_data_test=os.path.join(CRVD_path, 'indoor_raw_noisy/indoor_raw_noisy_scene%d/scene%d/ISO%d' % (scene_id, scene_id, iso))
                all_files_test=glob.glob(filepath_data_test+('/frame?_noisy%d*.tiff'%noise))
                if dimension==3:#读视频patch
                    dataset_test_real = dset.Get_sample_batch(all_files_test, t_length=args.t_length, 
                                                            patch_size=test_cropsz,iso=iso)
                else:
                    dataset_test_real = dset.Get_sample_batch_2d(all_files_test,
                                                            patch_size=test_cropsz, iso=iso)
                dataset_list_test.append(dataset_test_real)
            
    random.shuffle(dataset_list)

    if len(dataset_list)>1:
        dataset_list = torch.utils.data.ConcatDataset(tuple(dataset_list))
        dataset_list_test = torch.utils.data.ConcatDataset(tuple(dataset_list_test))
    else:
        dataset_list= dataset_list[0]
        dataset_list_test = dataset_list_test[0]
        
    return dataset_list, dataset_list_test


def get_dataset_RNVD(args,dimension=2,test_mode='crop'):
    import datasets.rnvd_supervised_dataset as dset

    if args.mode=='train':
        train_scene_id_list=[14,10,11,12,13,14,15,16,18,19,20,3,4,5,9]
        test_scene_id_list=[17]#,2,6,7,8]
        frame_rate=[20]#,60,120,200]#读frame rate
    elif args.mode=='eval':
        train_scene_id_list=[14,10]
        test_scene_id_list=[17,2,6,7,8]
        frame_rate=[20]#,60,120,200]#读frame rate

    noise_level=[25,30,35,40]
    seq_len={20:30,60:90,120:180,200:300}

    dataset_list = []
    dataset_list_test = []
    test_cropsz=args.crop_size if test_mode=='crop' else None
        
    RNVD_path='/data1/wjf/datasets/ruidataset/raw/'
    for noise_lv in noise_level:
        for frame_r in frame_rate:
            for scene_id in train_scene_id_list:
                # "/data1/wjf/datasets/ruidataset/raw/noisy/train/frame_rate/scene9/20f/35dnoise/30.Raw"
                filepath_data_train=os.path.join(RNVD_path, 'noisy/train/frame_rate/scene%d/%df/%ddnoise' % (scene_id, frame_r, noise_lv))
                train_lists=glob.glob(filepath_data_train+('/*.Raw'))
                all_files=sorted(train_lists,key=lambda x:int(re.findall(r"\d+",x)[-1]))[:seq_len[frame_r]]
                if dimension==3:
                    dataset_train_real = dset.Get_sample_batch_RNVD(all_files, t_length=args.t_length, \
                                                        patch_size=args.crop_size, seq_len=seq_len[frame_r])
                else:
                    dataset_train_real = dset.Get_sample_batch_RNVD_2d(all_files, \
                                                        patch_size=args.crop_size, seq_len=seq_len[frame_r])
                dataset_list.append(dataset_train_real)

            for scene_id in test_scene_id_list:
                # /data1/wjf/datasets/ruidataset/raw/noisy/test/frame_rate/scene8/20f/
                filepath_data_test=os.path.join(RNVD_path, 'noisy/test/frame_rate/scene%d/%df/%ddnoise' % (scene_id, frame_r, noise_lv))
                test_lists=glob.glob(filepath_data_test+('/*.Raw'))
                all_files_test=sorted(test_lists,key=lambda x:int(re.findall(r"\d+",x)[-1]))[:seq_len[frame_r]]
                if dimension==3:
                    dataset_test_real = dset.Get_sample_batch_RNVD(all_files_test, t_length=args.t_length, \
                                                        patch_size=test_cropsz, seq_len=seq_len[frame_r])
                else:
                    dataset_test_real = dset.Get_sample_batch_RNVD_2d(all_files_test, \
                                                        patch_size=test_cropsz, seq_len=seq_len[frame_r])
                dataset_list_test.append(dataset_test_real)
    
    random.shuffle(dataset_list)
        
    if len(dataset_list)>1:
        dataset_list = torch.utils.data.ConcatDataset(tuple(dataset_list))
        dataset_list_test = torch.utils.data.ConcatDataset(tuple(dataset_list_test))
    else:
        dataset_list= dataset_list[0]
        dataset_list_test = dataset_list_test[0]
        
    return dataset_list, dataset_list_test


def get_dataset_SRVD(args,dimension=3,test_mode='crop'):
    # dimension：=2时加载(b,c,h,w)单张图片，=3时加载(b,c,t,h,w)patch
    # test_mode：='crop'时测试集也裁剪，='all'时测试集不裁剪
    import datasets.SRVD_supervised_dataset as dset
    
    train_scene_id_list=[2,9,10,11]
    test_scene_id_list=[2]
    noise_level=[0,1]
    iso_list = [1600, 3200, 6400, 12800, 25600]
    MAX_LEN=500
    MAX_LEN_TEST=20

    dataset_list = []
    dataset_list_test = []
        
    SRVD_path='/data3/mxx/denoise_dataset/MOT_challenge_17/SRVD/'
    for noise in noise_level:
        for iso in iso_list:
            for scene_id in train_scene_id_list:
                filepath_data_train=os.path.join(SRVD_path, 'RAW_noisy/MOT17-%02d_raw/' % (scene_id))
                # ""/data3/mxx/denoise_dataset/MOT_challenge_17/SRVD/RAW_noisy/MOT17-11_raw/000236_raw_iso6400_noisy1.tiff""
                # ""/data3/mxx/denoise_dataset/MOT_challenge_17/SRVD/RAW_clean/MOT17-11_raw/000246_raw.tiff""
                all_files=sorted(glob.glob(filepath_data_train+('/*_raw_iso%d_noisy%d.tiff'%(iso,noise))))
                if dimension==3:#读视频patch->(b,c,t,h,w)
                    dataset_train_real = dset.Get_sample_batch_srvd(all_files, t_length=args.t_length,
                                                            patch_size=args.crop_size, iso=iso,max_len=MAX_LEN)
                else:
                    pass
                    # dataset_train_real = dset.Get_sample_batch_srvd_2d(all_files,
                    #                                         patch_size=args.crop_size, iso=iso,max_len=MAX_LEN)
                dataset_list.append(dataset_train_real)

            for scene_id in test_scene_id_list:
                filepath_data_test=os.path.join(SRVD_path, 'RAW_noisy/MOT17-%02d_raw/' % (scene_id))
                all_files_test=sorted(glob.glob(filepath_data_test+('/*_raw_iso%d_noisy%d.tiff'%(iso,noise))))

                test_cropsz=args.crop_size if test_mode=='crop' else None
                if dimension==3:#读视频patch
                    dataset_test_real = dset.Get_sample_batch_srvd(all_files_test, t_length=args.t_length, 
                                                            patch_size=test_cropsz,iso=iso,max_len=MAX_LEN_TEST)
                else:
                    pass
                    # dataset_test_real = dset.Get_sample_batch_srvd_2d(all_files_test, 
                    #                                         patch_size=test_cropsz,iso=iso,max_len=MAX_LEN_TEST)
                dataset_list_test.append(dataset_test_real)
            
    random.shuffle(dataset_list)
        
    if len(dataset_list)>1:
        dataset_list = torch.utils.data.ConcatDataset(tuple(dataset_list))
        dataset_list_test = torch.utils.data.ConcatDataset(tuple(dataset_list_test))
    else:
        dataset_list= dataset_list[0]
        dataset_list_test = dataset_list_test[0]
        
    return dataset_list, dataset_list_test


def get_dataset_DAVISraw(args,dimension=3,test_mode='crop'):
    # dimension：=2时加载(b,c,h,w)单张图片，=3时加载(b,c,t,h,w)patch
    # test_mode：='crop'时测试集也裁剪，='all'时测试集不裁剪
    import datasets.SRVD_supervised_dataset as dset
    DAVIS_path='/data3/mxx/denoise_dataset/DAVIS/DAVIS_my/DAVIS_raw'
    
    train_scene_list=dset.loadpath(os.path.join(DAVIS_path,'train.txt'))
    test_scene_list=dset.loadpath(os.path.join(DAVIS_path,'test_tmp.txt'))
    noise_level=[0,1]
    iso_list = [1600, 3200, 6400, 12800, 25600]
    MAX_LEN=50
    MAX_LEN_TEST=10

    dataset_list = []
    dataset_list_test = []
    for noise in noise_level:
        for iso in iso_list:
            for scene in train_scene_list:
                filepath_data_train=os.path.join(DAVIS_path, 'RAW_noisy_train/%s/' % scene)
                # ""/data3/mxx/denoise_dataset/MOT_challenge_17/SRVD/RAW_noisy/MOT17-11_raw/000236_raw_iso6400_noisy1.tiff""
                # ""/data3/mxx/denoise_dataset/MOT_challenge_17/SRVD/RAW_clean/MOT17-11_raw/000246_raw.tiff""
                all_files=sorted(glob.glob(filepath_data_train+('/*_raw_iso%d_noisy%d.tiff'%(iso,noise))))
                if dimension==3:#读视频patch->(b,c,t,h,w)
                    if args.use_syn:
                        dataset_train_real = dset.Get_sample_batch_syn(all_files, t_length=args.t_length,
                                                                patch_size=args.crop_size, iso=iso,max_len=len(all_files),dataset='davis')
                    else:
                        dataset_train_real = dset.Get_sample_batch_srvd(all_files, t_length=args.t_length,
                                                                patch_size=args.crop_size, iso=iso,max_len=len(all_files),dataset='davis')
                else:
                    pass
                    # dataset_train_real = dset.Get_sample_batch_srvd_2d(all_files, 
                    #                                         patch_size=args.crop_size, iso=iso,max_len=len(all_files),dataset='davis')
                dataset_list.append(dataset_train_real)

            for scene in test_scene_list:
                filepath_data_test=os.path.join(DAVIS_path, 'RAW_noisy_test/%s/' % scene)
                all_files_test=sorted(glob.glob(filepath_data_test+('/*_raw_iso%d_noisy%d.tiff'%(iso,noise))))

                test_cropsz=args.crop_size if test_mode=='crop' else None
                if dimension==3:#读视频patch
                    if args.use_syn:
                        dataset_test_real = dset.Get_sample_batch_syn(all_files_test, t_length=args.t_length, 
                                                                patch_size=test_cropsz,iso=iso,max_len=MAX_LEN_TEST,dataset='davis')
                    else:
                        dataset_test_real = dset.Get_sample_batch_srvd(all_files_test, t_length=args.t_length, 
                                                                patch_size=test_cropsz,iso=iso,max_len=MAX_LEN_TEST,dataset='davis')
                else:
                    pass
                    # dataset_test_real = dset.Get_sample_batch_srvd_2d(all_files_test,
                    #                                         patch_size=test_cropsz,iso=iso,max_len=MAX_LEN_TEST,dataset='davis')
                dataset_list_test.append(dataset_test_real)
            
    random.shuffle(dataset_list)
        
    if len(dataset_list)>1:
        dataset_list = torch.utils.data.ConcatDataset(tuple(dataset_list))
        dataset_list_test = torch.utils.data.ConcatDataset(tuple(dataset_list_test))
    else:
        dataset_list= dataset_list[0]
        dataset_list_test = dataset_list_test[0]
        
    return dataset_list, dataset_list_test


def load_network(net, load_path, strict=False, param_key='params'):
    """Load network.

    Args:
        load_path (str): The path of networks to be loaded.
        net (nn.Module): Network.
        strict (bool): Whether strictly loaded.
        param_key (str): The parameter key of loaded network. If set to
            None, use the root 'path'.
            Default: 'params'.
    """
    load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
    if param_key is not None:
        if param_key not in load_net and 'params' in load_net:
            param_key = 'params'
        load_net = load_net#[param_key]
    # remove unnecessary 'module.'
    for k, v in deepcopy(load_net).items():
        if k.startswith('module.'):
            load_net[k[7:]] = v
            load_net.pop(k)
    net.load_state_dict(load_net, strict=strict)
    return net


def get_dataset_DAVISraw_syn(args,dimension=3,test_mode='crop'):
    # dimension：=2时加载(b,c,h,w)单张图片，=3时加载(b,c,t,h,w)patch
    # test_mode：='crop'时测试集也裁剪，='all'时测试集不裁剪
    import datasets.SRVD_supervised_dataset as dset
    DAVIS_path='/data3/mxx/denoise_dataset/DAVIS/DAVIS_my/DAVIS_raw'
    
    train_scene_list=dset.loadpath(os.path.join(DAVIS_path,'train.txt'))
    test_scene_list=dset.loadpath(os.path.join(DAVIS_path,'test_tmp.txt'))

    # for real noise
    # datasyn=NoisySyn(res_learn=args.res_learn)
    # iso_list = [1600, 3200, 6400, 12800, 25600]
    # iso_list = [3200, 12800, 25, 35]

    # # for gaussian noise
    # datasyn=NoisySyn_gaussian(res_learn=args.res_learn)
    # iso_list=[15,25,50]

    # for locvar
    datasyn=NoisySyn_locvar(res_learn=args.res_learn)

    # # # for eld
    # datasyn=NoisySyn_eld()
    # datasyn=load_network(datasyn,load_path='/data3/mxx/Noise_generate/Starlight_ours_older/datasets/opts/VirtualNoisyPairGenerator_ELD_ptrqc_5VirtualCameras.pth')
    iso_list=[15]#,25,50]
    
    MAX_LEN=50
    MAX_LEN_TEST=10

    dataset_list = []
    dataset_list_test = []
    for scene in train_scene_list:
        filepath_data_train=os.path.join(DAVIS_path, 'RAW_clean_train/%s/' % scene)
        # ""/data3/mxx/denoise_dataset/MOT_challenge_17/SRVD/RAW_noisy/MOT17-11_raw/000236_raw_iso6400_noisy1.tiff""
        # ""/data3/mxx/denoise_dataset/MOT_challenge_17/SRVD/RAW_clean/MOT17-11_raw/000246_raw.tiff""
        all_files=sorted(glob.glob(filepath_data_train+('/*_raw.tiff')))
        if dimension==3:#读视频patch->(b,c,t,h,w)
            dataset_train_real = dset.Get_sample_batch_syn(all_files, t_length=args.t_length,patch_size=args.crop_size, 
                                                        noise_param=datasyn,iso_list=iso_list, max_len=len(all_files),dataset='davis')
        else:
            pass
        dataset_list.append(dataset_train_real)

    for scene in test_scene_list:
        filepath_data_test=os.path.join(DAVIS_path, 'RAW_clean_test/%s/' % scene)
        all_files_test=sorted(glob.glob(filepath_data_test+('/*_raw.tiff')))

        test_cropsz=args.crop_size if test_mode=='crop' else None
        if dimension==3:#读视频patch
            dataset_test_real = dset.Get_sample_batch_syn(all_files_test, t_length=args.t_length,patch_size=test_cropsz,
                                                        noise_param=datasyn,iso_list=iso_list, max_len=MAX_LEN_TEST,dataset='davis')
        else:
            pass
        dataset_list_test.append(dataset_test_real)
            
    random.shuffle(dataset_list)
        
    if len(dataset_list)>1:
        dataset_list = torch.utils.data.ConcatDataset(tuple(dataset_list))
        dataset_list_test = torch.utils.data.ConcatDataset(tuple(dataset_list_test))
    else:
        dataset_list= dataset_list[0]
        dataset_list_test = dataset_list_test[0]
        
    return dataset_list, dataset_list_test


def get_dataset_VIMEOraw_syn(args,dimension=3,test_mode='crop'):
    # dimension：=2时加载(b,c,h,w)单张图片，=3时加载(b,c,t,h,w)patch
    # test_mode：='crop'时测试集也裁剪，='all'时测试集不裁剪
    import datasets.SRVD_supervised_dataset as dset
    DAVIS_path='/data3/mxx/denoise_dataset/DAVIS/DAVIS_my/DAVIS_raw'
    
    train_scene_list=dset.loadpath(os.path.join(DAVIS_path,'train.txt'))
    test_scene_list=dset.loadpath(os.path.join(DAVIS_path,'test_tmp.txt'))
    datasyn=NoisySyn(res_learn=args.res_learn)
    # iso_list = [1600, 3200, 6400, 12800, 25600]
    # iso_list = [3200, 12800, 25, 35]
    iso_list = [1600,3200,6400,12800,25600,25,30,35,40]
    MAX_LEN=50
    MAX_LEN_TEST=10

    dataset_list = []
    dataset_list_test = []
    for scene in train_scene_list:
        filepath_data_train=os.path.join(DAVIS_path, 'RAW_clean_train/%s/' % scene)
        # ""/data3/mxx/denoise_dataset/MOT_challenge_17/SRVD/RAW_noisy/MOT17-11_raw/000236_raw_iso6400_noisy1.tiff""
        # ""/data3/mxx/denoise_dataset/MOT_challenge_17/SRVD/RAW_clean/MOT17-11_raw/000246_raw.tiff""
        all_files=sorted(glob.glob(filepath_data_train+('/*_raw.tiff')))
        if dimension==3:#读视频patch->(b,c,t,h,w)
            dataset_train_real = dset.Get_sample_batch_syn(all_files, t_length=args.t_length,patch_size=args.crop_size, 
                                                        noise_param=datasyn,iso_list=iso_list, max_len=len(all_files),dataset='davis')
        else:
            pass
        dataset_list.append(dataset_train_real)

    for scene in test_scene_list:
        filepath_data_test=os.path.join(DAVIS_path, 'RAW_clean_test/%s/' % scene)
        all_files_test=sorted(glob.glob(filepath_data_test+('/*_raw.tiff')))

        test_cropsz=args.crop_size if test_mode=='crop' else None
        if dimension==3:#读视频patch
            dataset_test_real = dset.Get_sample_batch_syn(all_files_test, t_length=args.t_length,patch_size=test_cropsz,
                                                        noise_param=datasyn,iso_list=iso_list, max_len=MAX_LEN_TEST,dataset='davis')
        else:
            pass
        dataset_list_test.append(dataset_test_real)
            
    random.shuffle(dataset_list)
        
    if len(dataset_list)>1:
        dataset_list = torch.utils.data.ConcatDataset(tuple(dataset_list))
        dataset_list_test = torch.utils.data.ConcatDataset(tuple(dataset_list_test))
    else:
        dataset_list= dataset_list[0]
        dataset_list_test = dataset_list_test[0]
        
    return dataset_list, dataset_list_test