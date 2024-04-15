from datasets.dataset_utils import read_16bit_raw, raw_to_4,\
    pack_gbrg_raw_torch,normalize_raw_torch,pack_gbrg_raw_torch2,\
    add_noisy_fromgt
# from dataset_utils import read_16bit_raw, raw_to_4,\
#     pack_gbrg_raw_torch,normalize_raw_torch,pack_gbrg_raw_torch2
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

a_list = [3.513262, 6.955588, 13.486051, 26.585953, 52.032536]
g_noise_var_list = [11.917691, 38.117816, 130.818508, 484.539790, 1819.818657]
iso_list = [1600, 3200, 6400, 12800, 25600]
 
    
class Get_sample_batch(object):
    """Loads in real still clean/noisy pairs dataset"""
    def __init__(self, input_dir, t_length=5,pin_memory=True,patch_size=256,iso=12800,return_mode='ind'):
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
        self.iso=iso    #对应的是噪声的级别吧，主要是要返回方差
        self.return_mode=return_mode

        if pin_memory:#提前加载所有数据
            self.noisy_frames={}
            self.gt_frames={}
            for frame in self.input_dir:#self.input_dir存储的是带噪声图片的地址
                self.noisy_frames[frame]=np.asarray(cv2.imread(frame, -1))
                gt_frame=re.sub(r'noisy\d.tiff',r'clean_and_slightly_denoised.tiff',frame).replace('noisy','gt')
                self.gt_frames[gt_frame]=np.asarray(cv2.imread(gt_frame, -1))
            # print('loaded %d frame pairs'%len(self.input_dir))


    def __len__(self):
        return len(self.input_dir)
    
    def cur_padding(self,cur_ind):
    # 根据center frame的index选定当前cur的index
    # 镜像padding
        if cur_ind<1:
            # index超出范围
            # 0->2,-1->3,-2->4,-3->5,-4->6,-5->7,-6->8
            # 8->6,9->5,10->4,11->3,12->2,13->1,14->0
            cur_ind=1+(1-cur_ind)
        elif cur_ind>7:
            cur_ind=7-(cur_ind-7)
        # 第一次处理完会有越界的，再对特殊情况处理下
        if cur_ind==0:
            cur_ind=2
        if cur_ind==8:
            cur_ind=6
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
        start_ind=int((self.input_dir[idx].split('/')[-1])[5])
        noisy_seq=[]
        gt_seq=[]
        for f in range(start_ind-self.t_length//2, start_ind+(self.t_length+1)//2):
        # for f in range(start_ind, start_ind+self.t_length):
            cur_ind=f
            while cur_ind<1 or cur_ind>7:#当前索引cur_ind限制在1~7内（写得跟shi一样有空改改）
                cur_ind=self.cur_padding(cur_ind)
            # print('cur',cur_ind)
            cur_path=self.input_dir[idx].replace('frame%01d'%start_ind, \
                'frame%01d'%cur_ind)
            if self.pin_memory:
                noisy_cur=self.noisy_frames[cur_path]
            else:
                noisy_cur=np.asarray(cv2.imread(cur_path, -1))
            noisy_seq.append(noisy_cur)

            gt_cur_path=re.sub(r'noisy\d.tiff',r'clean_and_slightly_denoised.tiff',cur_path).replace('noisy','gt')

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
        
        # print(gt_dir)
        # "/data3/mxx/denoise_dataset/CRVD/CRVD_dataset/indoor_raw_gt/indoor_raw_gt_scene1/scene1/ISO12800/frame1_clean_and_slightly_denoised.tiff"
        # "/data3/mxx/denoise_dataset/CRVD/CRVD_dataset/indoor_raw_noisy/indoor_raw_noisy_scene1/scene1/ISO12800/frame1_noisy0.tiff"

        # print('gt_seq:',gt_seq.shape,'noisy_seq:',noisy_seq.shape)#(t,h,w)
        # (16, 640, 1080, 4) (16, 640, 1080, 4)

        if self.return_mode=='ind':
            noise_level=iso_list.index(self.iso)
        else:
            a = torch.tensor(a_list[iso_list.index(self.iso)], dtype=torch.float32
                            ).view((1, 1, 1)) / max(a_list)#(2 ** 12 - 1 - 240)
            b = torch.tensor(g_noise_var_list[iso_list.index(self.iso)], dtype=torch.float32
                            ).view((1, 1, 1))  / max(g_noise_var_list)#((2 ** 12 - 1 - 240) ** 2)
            noise_level=[a,b]
            
        sample = {'noisy_input': noisy_seq,
                  'pos':[position_h,position_w],
                  'noise_level':noise_level,
                 'gt_label_nobias': gt_seq,
                 'frame_name':self.input_dir[idx].split('/')[-3]+'_'+
                 self.input_dir[idx].split('/')[-2]+'_'+self.input_dir[idx].split('/')[-1]
                 }
        
        # del noisy_im,gt_im

        # 这段代码后续得改一下
        sample['noisy_input'] = normalize_raw_torch(#normalize的作用是归一化，这里又校正了下暗电平
            pack_gbrg_raw_torch(sample['noisy_input']))
        sample['noisy_input']=sample['noisy_input'].permute(1,0,2,3)#(t,4,h,w)->(4,t,h,w)

        sample['gt_label_nobias'] = normalize_raw_torch(
            pack_gbrg_raw_torch(sample['gt_label_nobias']))
        sample['gt_label_nobias']=sample['gt_label_nobias'].permute(1,0,2,3)
        
        # print('max',torch.max(sample['noisy_input']),torch.max(sample['gt_label_nobias']),
        # 'min',torch.min(sample['noisy_input']),torch.min(sample['gt_label_nobias']))
        # print(sample['noisy_input'].shape,sample['gt_label_nobias'].shape)
        # sample: torch.Size([4, 16, 256, 256]) torch.Size([4, 16, 256, 256])
        # max tensor(0.9997) tensor(1.) min tensor(0.) tensor(0.0021)
        
        return sample
    

class Get_sample_batch_2d(object):
    """Loads in real still clean/noisy pairs dataset"""
    
    def __init__(self, input_dir, pin_memory=True,patch_size=256,iso=25600,return_mode='ind',dataset='real_noise'):
        """
        dataset: real_noise 或是 synthetic
        """
        self.input_dir = input_dir
        self.patch_size=patch_size
        self.pin_memory=pin_memory
        self.iso=iso    #对应的是噪声的级别吧，主要是要返回方差
        self.return_mode=return_mode    #新一版的合成模型返回的是ind（不过ind也作为输入送到了动态权重学习里，可能要归一化一下）
        self.dataset=dataset

        if pin_memory:#提前加载所有数据
            self.noisy_frames={}
            self.gt_frames={}
            for frame in self.input_dir:#self.input_dir存储的是带噪声图片的地址
                gt_frame=re.sub(r'noisy\d.tiff',r'clean_and_slightly_denoised.tiff',frame).replace('noisy','gt')
                self.gt_frames[gt_frame]=np.asarray(cv2.imread(gt_frame, -1))

                if self.dataset=='real_noise':
                    self.noisy_frames[frame]=np.asarray(cv2.imread(frame, -1))
                else:
                    self.noisy_frames[frame]=add_noisy_fromgt(self.gt_frames[gt_frame],iso)
                
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

    def crop16(self, img, h, w, sz=16):
        patch = img[0:(h//sz)*sz, 0:(w//sz)*sz]
        return patch

    def __getitem__(self, idx):
        cur_path=self.input_dir[idx]
        if self.pin_memory:
            noisy_cur=self.noisy_frames[cur_path]
        else:
            noisy_cur=np.asarray(cv2.imread(cur_path, -1))

        gt_cur_path=re.sub(r'noisy\d.tiff',r'clean_and_slightly_denoised.tiff',cur_path).replace('noisy','gt')

        if self.pin_memory:
            gt_cur=self.gt_frames[gt_cur_path]
        else:
            gt_cur=np.asarray(cv2.imread(gt_cur_path, -1))

        # print('noisy_seq:',noisy_seq.shape,'gt_seq:',gt_seq.shape)
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
            noisy_cur = self.crop16(noisy_cur,H,W)#(t,h,w)
            gt_cur = self.crop16(gt_cur,H,W)


        # print('gt_seq:',gt_seq.shape,'noisy_seq:',noisy_seq.shape)#(t,h,w)
        # (16, 640, 1080, 4) (16, 640, 1080, 4)
        if self.return_mode=='ind':
            noise_level=iso_list.index(self.iso)
        else:
            a = torch.tensor(a_list[iso_list.index(self.iso)], dtype=torch.float32
                            ).view((1, 1, 1)) / max(a_list)#(2 ** 12 - 1 - 240)
            b = torch.tensor(g_noise_var_list[iso_list.index(self.iso)], dtype=torch.float32
                            ).view((1, 1, 1))  / max(g_noise_var_list)#((2 ** 12 - 1 - 240) ** 2)
            noise_level=[a,b]

        sample = {'noisy_input': noisy_cur,
                  'pos':[position_h,position_w],
                  'noise_level':noise_level,
                 'gt_label_nobias': gt_cur}
        
        # del noisy_im,gt_im

        # 这段代码后续得改一下
        sample['noisy_input'] = normalize_raw_torch(#normalize的作用是归一化，这里又校正了下暗电平
            pack_gbrg_raw_torch2(sample['noisy_input']))#(4,h,w)
        sample['gt_label_nobias'] = normalize_raw_torch(
            pack_gbrg_raw_torch2(sample['gt_label_nobias']))#(4,h,w)
        
        return sample
    

def Get_sample_seq(seq_dir):#按照sequence依次加载
    noisy_seq=[]
    gt_seq=[]
    for ind in range(7):
        # print('cur',cur_ind)
        cur_path=seq_dir[ind]
        noisy_cur=np.asarray(cv2.imread(cur_path, -1))
        noisy_seq.append(noisy_cur)

        gt_cur_path=re.sub(r'noisy\d.tiff',r'clean_and_slightly_denoised.tiff',cur_path).replace('noisy','gt')
        gt_cur=np.asarray(cv2.imread(gt_cur_path, -1))
        gt_seq.append(gt_cur)

    noisy_seq=np.stack(noisy_seq,axis=0)
    gt_seq=np.stack(gt_seq,axis=0)
    # print('noisy_seq:',noisy_seq.shape,'gt_seq:',gt_seq.shape)
    noisy_seq=torch.from_numpy(noisy_seq.astype(np.float32))
    gt_seq=torch.from_numpy(gt_seq.astype(np.float32))
        
    sample = {'noisy_input': noisy_seq,
                'gt_label_nobias': gt_seq}
    
    # del noisy_im,gt_im

    # 这段代码后续得改一下
    sample['noisy_input'] = normalize_raw_torch(#normalize的作用是归一化，这里又校正了下暗电平
        pack_gbrg_raw_torch(sample['noisy_input']))
    sample['noisy_input']=sample['noisy_input'].permute(1,0,2,3)#(t,4,h,w)->(4,t,h,w)

    sample['gt_label_nobias'] = normalize_raw_torch(
        pack_gbrg_raw_torch(sample['gt_label_nobias']))
    sample['gt_label_nobias']=sample['gt_label_nobias'].permute(1,0,2,3)
    
    return sample
    
if __name__ == '__main__':
    import csv
    scene_id_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    noise_level=[0,1,2,3,4,5,6,7,8,9]
    iso_list = [1600, 3200, 6400, 12800, 25600]
    CRVD_path='/data3/mxx/denoise_dataset/CRVD/CRVD_dataset/'
    with open('avg_CRVD.csv','a') as file:
        writer=csv.writer(file)
        writer.writerow(['scene_id','iso','noise','mean_noisy0','mean_noisy1','mean_noisy2','mean_noisy3',
                         'mean_clean0','mean_clean1','mean_clean2','mean_clean3'])

    mean_gbrg_noisy=[]
    mean_gbrg_clean=[]
    for noise in noise_level:
        for iso in iso_list:
            for scene_id in scene_id_list:
                filepath_data=os.path.join(CRVD_path, 'indoor_raw_noisy/indoor_raw_noisy_scene%d/scene%d/ISO%d' % (scene_id, scene_id, iso))
                all_files_test=glob.glob(filepath_data+('/frame?_noisy%d*.tiff'%noise))
                dataset_real = Get_sample_seq(all_files_test)
                mean_noisy=torch.mean(dataset_real['noisy_input'],dim=[1,2,3])
                mean_clean=torch.mean(dataset_real['gt_label_nobias'],dim=[1,2,3])
                mean_gbrg_noisy.append(mean_noisy)
                mean_gbrg_clean.append(mean_clean)
                print('Avg gbrg of noise:%d, iso:%d, scene:%d is (noisy,clean)'%(noise,iso,scene_id),mean_noisy,mean_clean)
                with open('avg_CRVD.csv','a') as file:
                    writer=csv.writer(file)
                    writer.writerow([scene_id,iso,noise,mean_noisy[0].item(),mean_noisy[1].item(),mean_noisy[2].item(),mean_noisy[3].item(),
                                     mean_clean[0].item(),mean_clean[1].item(),mean_clean[2].item(),mean_clean[3].item()])
    # mean_gbrg_noisy=np.array(mean_gbrg_noisy)
    # mean_gbrg_clean=np.array(mean_gbrg_clean)