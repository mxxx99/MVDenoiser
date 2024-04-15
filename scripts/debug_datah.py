import sys, os, glob
os.environ['CUDA_VISIBLE_DEVICES']='4,5'

sys.path.append("../.")
sys.path.append("../data/")
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import MSELoss, L1Loss
from torch.optim import Adam
from PIL import Image
import argparse, json, torchvision
import datasets.dataset_helper as datah
import cv2


# concat dataset，每个seq一个dataset类
parser = argparse.ArgumentParser(description='Gan noise model training options.')
parser.add_argument('--crop_size', default=256, type = int)
parser.add_argument('--t_length', default=5, type = int) 
parser.add_argument('--dataset', default='color', help = 'Choose which dataset to use. Options: gray, color')

args = parser.parse_args()
dataset_list, dataset_list_test = datah.get_dataset_SRVD(args)
# dataset_list, dataset_list_test = datah.get_dataset(args)
print(dataset_list.__len__(),dataset_list_test.__len__())
# train_loader = torch.utils.data.DataLoader(dataset=dataset_list, 
#                                             batch_size=3,
#                                             shuffle=False,
#                                             num_workers=0,
#                                             pin_memory=True)
sample = dataset_list.__getitem__(1)
print(sample['noisy_input'].shape,sample['pos'],\
      sample['noise_level'],sample['gt_label_nobias'].shape)
# torch.Size([4, 5, 128, 128])
# [tensor([446, 446, 446, 446, 446], dtype=torch.int32), tensor([740, 740, 740, 740, 740], dtype=torch.int32)]
# 0
# torch.Size([4, 5, 128, 128])

# (c,t,h,w),(t,),(1),(c,t,h,w)