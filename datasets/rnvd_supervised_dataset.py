from datasets.dataset_utils import read_16bit_raw, raw_to_4,\
    pack_gbrg_raw_torch,normalize_raw_torch,normalize_raw_rnvd,\
    pack_bggr_raw_torch
import torch
import sys, os, glob
import numpy as np
import scipy.io
from PIL import Image
import datasets.post_processing as pp
import time
import cv2
from skimage import exposure
import torchvision
from pathlib import Path
import re
_script_dir = Path( __file__ ).parent
_root_dir = _script_dir.parent

noise_level_list=[25,30,35,40]

class Get_sample_batch_RNVD(object):
    """Loads in real still clean/noisy pairs dataset"""
    
    def __init__(self, input_dir, t_length=16,pin_memory=True,patch_size=256,seq_len=30):
        """
        Args:
            filenames: List of filenames
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input_dir = input_dir
        self.t_length=t_length
        self.patch_size=patch_size
        self.pin_memory=pin_memory
        self.seq_len=seq_len
        self.dtype=np.uint8    #看师兄代码里写的是8位
        self.shape=[1536,2048]

        if pin_memory:#提前加载所有数据
            self.noisy_frames={}
            self.gt_frames={}
            # "/data1/wjf/datasets/ruidataset/raw/noisy/train/frame_rate/scene9/20f/35dnoise/30.Raw"
            # "/data1/wjf/datasets/ruidataset/raw/clean/train/frame_rate/scene9/20f/35dgt/30.raw"
            for frame in self.input_dir:#self.input_dir存储的是带噪声图片的地址
                self.noisy_frames[frame]=np.fromfile(frame,dtype=self.dtype).reshape(self.shape[0],self.shape[1])#np.asarray(cv2.imread(frame, -1))
                # 在noisy里是Raw，gt里是raw
                gt_frame=frame.replace('noisy','clean').replace('noise','gt').replace('Raw','raw')
                self.gt_frames[gt_frame]=np.fromfile(gt_frame,dtype=self.dtype).reshape(self.shape[0],self.shape[1])#np.asarray(cv2.imread(gt_frame, -1))
            # print('loaded %d frame pairs'%len(self.input_dir))


    def __len__(self):
        return len(self.input_dir)
    
    def cur_padding(self,cur_ind):
    # 根据center frame的index选定当前cur的index
    # 镜像padding
        if cur_ind<1:
            cur_ind=1+(1-cur_ind)
        elif cur_ind>self.seq_len:
            cur_ind=self.seq_len-(cur_ind-self.seq_len)
        return cur_ind
    
    def crop_position(self, patch_size, H, W):
        position_h = np.random.randint(0, (H - patch_size)//2 - 1) * 2
        position_w = np.random.randint(0, (W - patch_size)//2 - 1) * 2
        aug = np.random.randint(0, 8)
        return position_h, position_w
    
    def crop(self, img, patch_size, position_h, position_w):
        patch = img[:, position_h:position_h + patch_size, position_w:position_w + patch_size]
        return patch

    def crop16(self, img,sz=16):
        h=img.shape[-2]
        w=img.shape[-1]
        patch = img[...,0:(h//sz)*sz, 0:(w//sz)*sz]
        return patch

    def __getitem__(self, idx):
        start_frame=int(re.findall(r"\d+",self.input_dir[idx])[-1])
        noisy_seq=[]
        gt_seq=[]
        for f in range(start_frame-self.t_length//2, start_frame+self.t_length//2+1):
            cur_ind=f
            while cur_ind<1 or cur_ind>self.seq_len:#当前索引cur_ind限制
                cur_ind=self.cur_padding(cur_ind)
            # print('cur',cur_ind)
            cur_frame=self.input_dir[idx].replace('%d.Raw'%start_frame, \
                '%d.Raw'%cur_ind)
            if self.pin_memory:
                noisy_cur=self.noisy_frames[cur_frame]
            else:
                noisy_cur=np.fromfile(cur_frame,dtype=self.dtype).reshape(self.shape[0],self.shape[1])
                # print('noisy_cur:',noisy_cur.shape,noisy_cur.dtype,noisy_cur[0])
            noisy_seq.append(noisy_cur)

            gt_cur_frame=cur_frame.replace('noisy','clean').replace('noise','gt').replace('Raw','raw')

            if self.pin_memory:
                gt_cur=self.gt_frames[gt_cur_frame]
            else:
                gt_cur=np.fromfile(gt_cur_frame,dtype=self.dtype).reshape(self.shape[0],self.shape[1])#np.asarray(cv2.imread(gt_cur_frame, -1))
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
            noisy_seq = self.crop16(noisy_seq)#(t,h,w)
            gt_seq = self.crop16(gt_seq)

        # 计算噪声级别
        noise_level=int(re.findall(r"\d+",re.findall(r"\d+dnoise",self.input_dir[idx])[0])[0])
        noise_level=noise_level_list.index(noise_level)

        sample = {'noisy_input': noisy_seq,
                  'pos':[position_h,position_w],
                  'noise_level':noise_level,#这个noise_level现在是增益，要改成噪声的级别
                 'gt_label_nobias': gt_seq,
                'frame_name':self.input_dir[idx].split('/')[-4]+'_'+
                 self.input_dir[idx].split('/')[-2]+'_'+self.input_dir[idx].split('/')[-1]
                 }
        


        # 这段代码后续得改一下
        # 我们的数据集在归一化的同时没法校正，因为不知道暗电平的值，起到的就只是一个0~255->0~1的作用
        # 但是noise和gt的min是有差别的...
        # 这里的pack好像有点问题...不知道是bggr还是啥模式，我直接用的crvd里的pack
        sample['noisy_input'] = normalize_raw_rnvd(pack_bggr_raw_torch(sample['noisy_input']))
        sample['noisy_input']=sample['noisy_input'].permute(1,0,2,3)#(t,4,h,w)->(4,t,h,w)

        sample['gt_label_nobias'] = normalize_raw_rnvd(pack_bggr_raw_torch(sample['gt_label_nobias']))
        sample['gt_label_nobias']=sample['gt_label_nobias'].permute(1,0,2,3)
        
        # print('max',torch.max(sample['noisy_input']),torch.max(sample['gt_label_nobias']),
        #       'min',torch.min(sample['noisy_input']),torch.min(sample['gt_label_nobias']))
        # max tensor(1.) tensor(1.) min tensor(0.0118) tensor(0.1333)
        
        return sample
    

class Get_sample_batch_RNVD_2d(object):
    """2d RNVD"""
    
    def __init__(self, input_dir, pin_memory=True,patch_size=256,seq_len=30):
        """
        Args:
            filenames: List of filenames
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input_dir = input_dir
        self.patch_size=patch_size
        self.pin_memory=False
        self.seq_len=seq_len
        self.dtype=np.uint8    #看师兄代码里写的是8位
        self.return_mode='ind'
        self.shape=[1536,2048]

        if pin_memory:#提前加载所有数据
            self.noisy_frames={}
            self.gt_frames={}
            # "/data1/wjf/datasets/ruidataset/raw/noisy/train/frame_rate/scene9/20f/35dnoise/30.Raw"
            # "/data1/wjf/datasets/ruidataset/raw/clean/train/frame_rate/scene9/20f/35dgt/30.raw"
            for frame in self.input_dir:#self.input_dir存储的是带噪声图片的地址
                self.noisy_frames[frame]=np.fromfile(frame,dtype=self.dtype).reshape(self.shape[0],self.shape[1])#np.asarray(cv2.imread(frame, -1))
                # 在noisy里是Raw，gt里是raw
                gt_frame=frame.replace('noisy','clean').replace('noise','gt').replace('Raw','raw')
                self.gt_frames[gt_frame]=np.fromfile(gt_frame,dtype=self.dtype).reshape(self.shape[0],self.shape[1])#np.asarray(cv2.imread(gt_frame, -1))
            # print('loaded %d frame pairs'%len(self.input_dir))


    def __len__(self):
        return len(self.input_dir)
    
    def crop_position(self, patch_size, H, W):
        position_h = np.random.randint(0, (H - patch_size)//2 - 1) * 2
        position_w = np.random.randint(0, (W - patch_size)//2 - 1) * 2
        aug = np.random.randint(0, 8)
        return position_h, position_w
    
    def crop(self, img, patch_size, position_h, position_w):
        patch = img[position_h:position_h + patch_size, position_w:position_w + patch_size]
        return patch

    def crop16(self, img,sz=16):
        h=img.shape[-2]
        w=img.shape[-1]
        patch = img[0:(h//sz)*sz, 0:(w//sz)*sz]
        return patch

    def __getitem__(self, idx):
        cur_path=self.input_dir[idx]
        if self.pin_memory:
            noisy_cur=self.noisy_frames[cur_path]
        else:
            noisy_cur=np.fromfile(cur_path,dtype=self.dtype).reshape(self.shape[0],self.shape[1])

        gt_cur_path=cur_path.replace('noisy','clean').replace('noise','gt').replace('Raw','raw')

        if self.pin_memory:
            gt_cur=self.gt_frames[gt_cur_path]
        else:
            gt_cur=np.fromfile(gt_cur_path,dtype=self.dtype).reshape(self.shape[0],self.shape[1])

        noisy_cur=torch.from_numpy(noisy_cur.astype(np.float32))
        gt_cur=torch.from_numpy(gt_cur.astype(np.float32))

        # random crop
        H,W=noisy_cur.shape
        if self.patch_size!=None:
            position_h, position_w = self.crop_position(self.patch_size, H, W)
            noisy_cur = self.crop(noisy_cur, self.patch_size, position_h, position_w)#(t,h,w)
            gt_cur = self.crop(gt_cur, self.patch_size, position_h, position_w)
        else:
            position_h=0
            position_w=0
            noisy_cur = self.crop16(noisy_cur)#(t,h,w)
            gt_cur = self.crop16(gt_cur)

        # 计算噪声级别
        if self.return_mode=='ind':
            noise_level=int(re.findall(r"\d+",re.findall(r"\d+dnoise",self.input_dir[idx])[0])[0])
            noise_level=noise_level_list.index(noise_level)

        sample = {'noisy_input': noisy_cur,
                  'pos':[position_h,position_w],
                  'noise_level':noise_level,#这个noise_level现在是增益，要改成噪声的级别
                 'gt_label_nobias': gt_cur}
        
        # 这段代码后续得改一下
        # 我们的数据集在归一化的同时没法校正，因为不知道暗电平的值，起到的就只是一个0~255->0~1的作用
        # 但是noise和gt的min是有差别的...
        # 这里的pack好像有点问题...不知道是bggr还是啥模式，我直接用的crvd里的pack
        sample['noisy_input'] = normalize_raw_rnvd(pack_bggr_raw_torch(sample['noisy_input']))
        sample['gt_label_nobias'] = normalize_raw_rnvd(pack_bggr_raw_torch(sample['gt_label_nobias']))

        return sample