import os,sys
os.environ['CUDA_VISIBLE_DEVICES']='7'

sys.path.append("../.")
sys.path.append("../data/")

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import MSELoss, L1Loss
from torch.optim import Adam
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import cv2
from datasets.dataset_utils import add_noisy_fromgt,pack_gbrg_raw_torch2,normalize_raw_torch
import re

import argparse, json, torchvision
import scipy.io
import glob
import math

class Get_sample_batch_2d_forkld(object):
    """Loads in real still clean/noisy pairs dataset"""
    
    def __init__(self, input_dir, pin_memory=True,iso=25600):
        """
        dataset: real_noise 或是 synthetic
        """
        self.input_dir = input_dir
        self.pin_memory=pin_memory
        self.iso=iso    #对应的是噪声的级别吧，主要是要返回方差

        if pin_memory:#提前加载所有数据
            self.noisy_frames={}
            self.gt_frames={}
            self.noisy_frames_gen={}
            for frame in self.input_dir:#self.input_dir存储的是带噪声图片的地址
                gt_frame=re.sub(r'noisy\d.tiff',r'clean_and_slightly_denoised.tiff',frame).replace('noisy','gt')
                self.gt_frames[gt_frame]=np.asarray(cv2.imread(gt_frame, -1))
                self.noisy_frames[frame]=np.asarray(cv2.imread(frame, -1))


    def __len__(self):
        return len(self.input_dir)

    def crop16(self, img, h, w, sz=16):
        patch = img[0:(h//sz)*sz, 0:(w//sz)*sz]
        return patch

    def __getitem__(self, idx):
        cur_path=self.input_dir[idx]
        noisy_cur=self.noisy_frames[cur_path]

        gt_cur_path=re.sub(r'noisy\d.tiff',r'clean_and_slightly_denoised.tiff',cur_path).replace('noisy','gt')
        gt_cur=self.gt_frames[gt_cur_path]
        noisy_cur_syn=add_noisy_fromgt(gt_cur,self.iso)

        noisy_cur=torch.from_numpy(noisy_cur.astype(np.float32))
        gt_cur=torch.from_numpy(gt_cur.astype(np.float32))
        noisy_cur_syn=torch.from_numpy(noisy_cur_syn.astype(np.float32))

        H,W=noisy_cur.shape
        noisy_cur = self.crop16(noisy_cur,H,W)#(t,h,w)
        noisy_cur_syn = self.crop16(noisy_cur_syn,H,W)
        gt_cur = self.crop16(gt_cur,H,W)

        sample = {'noisy_input': noisy_cur,
                  'noisy_input_syn':noisy_cur_syn,
                  'noise_level':self.iso,
                 'gt_label_nobias': gt_cur}
        
        sample['noisy_input'] = normalize_raw_torch(#normalize的作用是归一化，这里又校正了下暗电平
            pack_gbrg_raw_torch2(sample['noisy_input']))#(4,h,w)
        sample['gt_label_nobias'] = normalize_raw_torch(
            pack_gbrg_raw_torch2(sample['gt_label_nobias']))#(4,h,w)
        sample['noisy_input_syn'] = normalize_raw_torch(#normalize的作用是归一化，这里又校正了下暗电平
            pack_gbrg_raw_torch2(sample['noisy_input_syn']))
        return sample


def get_dataset_CRVD_forkld():    
    test_scene_id_list=[7,8,9,10,11]
    noise_level=[0]#,1,2,3,4,5,6,7,8,9]
    iso_list = [25600]#, 3200, 6400, 12800, 25600]
    dataset_list_test = []
    CRVD_path='/data3/mxx/denoise_dataset/CRVD/CRVD_dataset/'
    for noise in noise_level:
        for iso in iso_list:
            for scene_id in test_scene_id_list:
                filepath_data_test=os.path.join(CRVD_path, 'indoor_raw_noisy/indoor_raw_noisy_scene%d/scene%d/ISO%d' % (scene_id, scene_id, iso))
                all_files_test=glob.glob(filepath_data_test+('/frame?_noisy%d*.tiff'%noise))
                dataset_test_real = Get_sample_batch_2d_forkld(all_files_test,iso=iso)
                dataset_list_test.append(dataset_test_real)

    dataset_list_test = torch.utils.data.ConcatDataset(tuple(dataset_list_test))
    return dataset_list_test


if __name__ == '__main__':
    import utils.noisegen_helper_fun as gh
    device = torch.device('cuda:0')

    dataset_list_test = get_dataset_CRVD_forkld()
    print(dataset_list_test.__len__())
    test_loader = torch.utils.data.DataLoader(dataset=dataset_list_test, 
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=0,
                                            pin_memory=True)

    # 计算KLD
    tot_kld = 0
    for i, sample in enumerate(test_loader):
        with torch.no_grad():
            # (b*t,c,h,w)
            B,C,H,W=sample['noisy_input'].shape
            noisy_raw = sample['noisy_input'].to(device)
            clean_raw = sample['gt_label_nobias'].to(device)
            noisy_raw_gen = sample['noisy_input_syn'].to(device)
            noise_level=sample['noise_level']

            gen_noisy_ = gh.split_into_patches2d(noisy_raw_gen).to(device)
            real_noisy = gh.split_into_patches2d(noisy_raw).to(device)
            gen1 = (gen_noisy_).detach().cpu().numpy()
            real1 = (real_noisy).detach().cpu().numpy()
            kld_val = gh.cal_kld(gen1, real1)
            print('%dth image, iso=%d, kld=%f'%(i,noise_level[0],kld_val))
            tot_kld += kld_val

    print('Total KLD value: %.4f, Avg KLD value: %.4f'%(tot_kld,tot_kld/(i+1)))