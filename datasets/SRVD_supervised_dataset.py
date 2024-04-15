from datasets.dataset_utils import read_16bit_raw, raw_to_4,\
    pack_gbrg_raw_torch,normalize_raw_torch,pack_gbrg_raw_torch2
# from dataset_utils import read_16bit_raw, raw_to_4,\
#     pack_gbrg_raw_torch,normalize_raw_torch,pack_gbrg_raw_torch2
import random
import torch
import sys, os, glob
import numpy as np
import scipy.io
from PIL import Image
import time
import cv2
from skimage import exposure
import torchvision
from pathlib import Path
import re
_script_dir = Path( __file__ ).parent
_root_dir = _script_dir.parent

iso_list = [1600, 3200, 6400, 12800, 25600]

class Get_sample_batch_srvd(object):
    """srvd"""
    def __init__(self, all_files, t_length=5,pin_memory=True,patch_size=256,iso=12800,max_len=20,dataset='srvd',return_mode='ind'):
        """
        Args:
            filenames: List of filenames
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.max_len=max_len
        self.input_dir = all_files[0:max_len]
        self.t_length=t_length
        self.patch_size=patch_size
        self.pin_memory=pin_memory
        self.dataset=dataset
        self.iso=iso    #对应的是噪声的级别吧，主要是要返回方差
        self.return_mode=return_mode

        if pin_memory:#提前加载所有数据
            self.noisy_frames={}
            self.gt_frames={}
            # ""/data3/mxx/denoise_dataset/MOT_challenge_17/SRVD/RAW_noisy/MOT17-11_raw/000236_raw_iso6400_noisy1.tiff""
            # ""/data3/mxx/denoise_dataset/MOT_challenge_17/SRVD/RAW_clean/MOT17-11_raw/000246_raw.tiff""
            for frame in self.input_dir:#self.input_dir存储的是带噪声图片的地址
                self.noisy_frames[frame]=np.asarray(cv2.imread(frame, -1))
                gt_frame=re.sub(r'raw_iso\d+_noisy\d.tiff',r'raw.tiff',frame).replace('noisy','clean')
                self.gt_frames[gt_frame]=np.asarray(cv2.imread(gt_frame, -1))


    def __len__(self):
        return len(self.input_dir)
    
    def cur_padding(self,cur_ind):#现在只支持0~max_len-1了
    # 根据center frame的index选定当前cur的index
    # 镜像padding
        if cur_ind<0:
            cur_ind=-cur_ind
        elif cur_ind>=self.max_len:
            cur_ind=(self.max_len-1)-(cur_ind-self.max_len+1)
        return cur_ind
    
    def crop_position(self, patch_size, H, W):
        position_h = np.random.randint(0, (H - patch_size)//2 - 1) * 2
        position_w = np.random.randint(0, (W - patch_size)//2 - 1) * 2
        aug = np.random.randint(0, 8)
        return position_h, position_w
    
    def crop(self, img, patch_size, position_h, position_w):
        patch = img[:, position_h:position_h + patch_size, position_w:position_w + patch_size]
        return patch

    def crop16(self, img, h, w, sz=16):
        patch = img[...,0:(h//sz)*sz, 0:(w//sz)*sz]
        return patch

    def __getitem__(self, idx):
        # ""/data3/mxx/denoise_dataset/MOT_challenge_17/SRVD/RAW_noisy/MOT17-11_raw/000236_raw_iso6400_noisy1.tiff""
        # ""/data3/mxx/denoise_dataset/MOT_challenge_17/SRVD/RAW_clean/MOT17-11_raw/000246_raw.tiff""
        if self.dataset=='srvd':
            start_ind=int((self.input_dir[idx].split('/')[-1])[:6])
        else:
            start_ind=int((self.input_dir[idx].split('/')[-1])[:5])

        noisy_seq=[]
        gt_seq=[]
        # print(start_ind)
        for f in range(start_ind-self.t_length//2, start_ind+self.t_length//2+1):
            cur_ind=f
            if self.dataset=='srvd':
                while cur_ind<1 or cur_ind>self.max_len:#当前索引cur_ind限制在1~max_len
                    cur_ind=self.cur_padding(cur_ind)
            else:
                while cur_ind<0 or cur_ind>=self.max_len:#当前索引cur_ind限制在0~max_len-1
                    cur_ind=self.cur_padding(cur_ind)
            # print('cur',cur_ind)
            if self.dataset=='srvd':#srvd和davis名字中含有数字的位数不同
                cur_path=self.input_dir[idx].replace('%06d_raw'%start_ind, \
                    '%06d_raw'%cur_ind)
            else:
                cur_path=self.input_dir[idx].replace('%05d_raw'%start_ind, \
                    '%05d_raw'%cur_ind)
                
            if self.pin_memory:
                noisy_cur=self.noisy_frames[cur_path]
            else:
                noisy_cur=np.asarray(cv2.imread(cur_path, -1))
            noisy_seq.append(noisy_cur)

            gt_cur_path=re.sub(r'raw_iso\d+_noisy\d.tiff',r'raw.tiff',cur_path).replace('noisy','clean')

            if self.pin_memory:
                gt_cur=self.gt_frames[gt_cur_path]
            else:
                gt_cur=np.asarray(cv2.imread(gt_cur_path, -1))
            gt_seq.append(gt_cur)

        noisy_seq=np.stack(noisy_seq,axis=0)
        gt_seq=np.stack(gt_seq,axis=0)
        # print('noisy_seq:',noisy_seq.shape,'gt_seq:',gt_seq.shape)
        noisy_seq=torch.from_numpy(noisy_seq.astype(np.float32))
        gt_seq=torch.from_numpy(gt_seq.astype(np.float32))

        # random crop
        T,H,W=noisy_seq.shape
        if self.patch_size!=None:
            position_h, position_w = self.crop_position(self.patch_size, H, W)
            noisy_seq = self.crop(noisy_seq, self.patch_size, position_h, position_w)#(t,h,w)
            gt_seq = self.crop(gt_seq, self.patch_size, position_h, position_w)
        else:
            position_h=0
            position_w=0
            noisy_seq = self.crop16(noisy_seq,H,W)#(t,h,w)
            gt_seq = self.crop16(gt_seq,H,W)

        position_h=(torch.ones(T)*position_h).int()#(t,)
        position_w=(torch.ones(T)*position_w).int()#(t,)

        noise_level=iso_list.index(self.iso)
            
        sample = {'noisy_input': noisy_seq,
                  'pos':[position_h,position_w],
                  'noise_level':noise_level,
                 'gt_label_nobias': gt_seq}

        # 这段代码后续得改一下
        sample['noisy_input'] = normalize_raw_torch(#normalize的作用是归一化，这里又校正了下暗电平
            pack_gbrg_raw_torch(sample['noisy_input']))
        sample['noisy_input']=sample['noisy_input'].permute(1,0,2,3)#(t,4,h,w)->(4,t,h,w)

        sample['gt_label_nobias'] = normalize_raw_torch(
            pack_gbrg_raw_torch(sample['gt_label_nobias']))
        sample['gt_label_nobias']=sample['gt_label_nobias'].permute(1,0,2,3)
        
        return sample
    


class Get_sample_batch_syn(object):
    """srvd"""
    # dset.Get_sample_batch_syn(all_files_test, t_length=args.t_length, 
    #                                                 patch_size=test_cropsz,max_len=MAX_LEN_TEST,dataset='davis')
    def __init__(self, all_files, t_length=5,pin_memory=True,patch_size=256,\
                 noise_param=None,iso_list=[12800],max_len=20,dataset='srvd',return_mode='ind'):
        """
        Args:
            filenames: List of filenames
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.max_len=max_len
        self.input_dir = all_files[0:max_len]
        self.t_length=t_length
        self.patch_size=patch_size
        self.pin_memory=pin_memory
        self.dataset=dataset
        self.noise_param=noise_param
        self.iso_list=iso_list    #对应的是噪声的级别吧，主要是要返回方差
        self.return_mode=return_mode

        if pin_memory:#提前加载所有数据
            self.gt_frames={}
            # ""/data3/mxx/denoise_dataset/MOT_challenge_17/SRVD/RAW_noisy/MOT17-11_raw/000236_raw_iso6400_noisy1.tiff""
            # ""/data3/mxx/denoise_dataset/MOT_challenge_17/SRVD/RAW_clean/MOT17-11_raw/000246_raw.tiff""
            for frame in self.input_dir:#self.input_dir存储的是带噪声图片的地址
                gt_frame=re.sub(r'raw_iso\d+_noisy\d.tiff',r'raw.tiff',frame).replace('noisy','clean')
                self.gt_frames[gt_frame]=np.asarray(cv2.imread(gt_frame, -1))


    def __len__(self):
        return len(self.input_dir)
    
    def cur_padding(self,cur_ind):#现在只支持0~max_len-1了
    # 根据center frame的index选定当前cur的index
    # 镜像padding
        if cur_ind<0:
            cur_ind=-cur_ind
        elif cur_ind>=self.max_len:
            cur_ind=(self.max_len-1)-(cur_ind-self.max_len+1)
        return cur_ind
    
    def crop_position(self, patch_size, H, W):
        position_h = np.random.randint(0, (H - patch_size)//2 - 1) * 2
        position_w = np.random.randint(0, (W - patch_size)//2 - 1) * 2
        aug = np.random.randint(0, 8)
        return position_h, position_w
    
    def crop(self, img, patch_size, position_h, position_w):
        patch = img[:, position_h:position_h + patch_size, position_w:position_w + patch_size]
        return patch

    def crop16(self, img, h, w, sz=16):
        patch = img[...,0:(h//sz)*sz, 0:(w//sz)*sz]
        return patch

    def __getitem__(self, idx):
        # gauss
        # noise_level=self.iso_list[random.randrange(len(self.iso_list))]

        # # locvar
        # noise_level=np.random.randint(50,120)

        # eld
        noise_level=1

        # ""/data3/mxx/denoise_dataset/MOT_challenge_17/SRVD/RAW_noisy/MOT17-11_raw/000236_raw_iso6400_noisy1.tiff""
        # ""/data3/mxx/denoise_dataset/MOT_challenge_17/SRVD/RAW_clean/MOT17-11_raw/000246_raw.tiff""
        if self.dataset=='srvd':
            start_ind=int((self.input_dir[idx].split('/')[-1])[:6])
        else:
            start_ind=int((self.input_dir[idx].split('/')[-1])[:5])

        gt_seq=[]
        # print(start_ind)
        for f in range(start_ind-self.t_length//2, start_ind+self.t_length//2+1):
            cur_ind=f
            if self.dataset=='srvd':
                while cur_ind<1 or cur_ind>self.max_len:#当前索引cur_ind限制在1~max_len
                    cur_ind=self.cur_padding(cur_ind)
            else:
                while cur_ind<0 or cur_ind>=self.max_len:#当前索引cur_ind限制在0~max_len-1
                    cur_ind=self.cur_padding(cur_ind)
            # print('cur',cur_ind)
            if self.dataset=='srvd':#srvd和davis名字中含有数字的位数不同
                cur_path=self.input_dir[idx].replace('%06d_raw'%start_ind, \
                    '%06d_raw'%cur_ind)
            else:
                cur_path=self.input_dir[idx].replace('%05d_raw'%start_ind, \
                    '%05d_raw'%cur_ind)

            if self.pin_memory:
                gt_cur=self.gt_frames[cur_path]
            else:
                gt_cur=np.asarray(cv2.imread(cur_path, -1))
            gt_seq.append(gt_cur)

        gt_seq=np.stack(gt_seq,axis=0)
        # print('noisy_seq:',noisy_seq.shape,'gt_seq:',gt_seq.shape)
        gt_seq=torch.from_numpy(gt_seq.astype(np.float32))

        # random crop
        T,H,W=gt_seq.shape
        if self.patch_size!=None:
            position_h, position_w = self.crop_position(self.patch_size, H, W)
            gt_seq = self.crop(gt_seq, self.patch_size, position_h, position_w)
        else:
            position_h=0
            position_w=0
            gt_seq = self.crop16(gt_seq,H,W)

        position_h=(torch.ones(T)*position_h).int()#(t,)
        position_w=(torch.ones(T)*position_w).int()#(t,)
        gt_seq = normalize_raw_torch(pack_gbrg_raw_torch(gt_seq)).permute(1,0,2,3)
        noise_seq,noisy_inter,noise_levels=self.noise_param.add_noisy_fromgt_syn(gt_seq,noise_level)

        sample = {'noisy_input': noise_seq,
            'noisy_inter':noisy_inter,
            'pos':[position_h,position_w],
            'noise_levels':noise_levels,
            'noise_level':noise_level,
            'gt_label_nobias': gt_seq}
        
        return sample



def loadpath(pathlistfile):
    fp = open(pathlistfile)
    pathlist = fp.read().splitlines()
    fp.close()
    return pathlist


if __name__ == '__main__':
    pass