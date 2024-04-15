import sys, os, glob
os.environ['CUDA_VISIBLE_DEVICES']='4'
sys.path.append('..')

import torch
import numpy as np
from  CKA.cka import CKACalculator_ours
from utils.denoiser_helper_fun import load_state_dict_ddp
import datasets.dataset_helper as datah
import matplotlib.pyplot as plt

def get_model(path,device):
    from models.stage_denoiser import Stage_denoise3
    # from models.stage_denoiser_recurr import Stage_denoise32
    keys=['shot','read', 'uniform']
    model = Stage_denoise3(n_channel_in=4, device=device,
                keys=keys[::-1],res_learn=False,
                noise_cat=False,t_length=5,input_mask=False)
    model=load_state_dict_ddp(model,path,0)
    model=model.to(device)
    return model

def get_dataset(patchsz=512):
    import datasets.crvd_supervised_dataset as dset
    scene_id_list=[7,8,9,10,11]
    iso_list = [25600]#[1600, 3200, 6400, 12800, 25600]
    dataset_list = []
    CRVD_path='/data3/mxx/denoise_dataset/CRVD/CRVD_dataset/'
    for iso in iso_list:
        for scene_id in scene_id_list:
            filepath_data_train=os.path.join(CRVD_path, \
                                             'indoor_raw_noisy/indoor_raw_noisy_scene%d/scene%d/ISO%d' % (scene_id, scene_id, iso))
            all_files=glob.glob(filepath_data_train+('/frame?_noisy0.tiff'))
            dataset_train_real = dset.Get_sample_batch(all_files, t_length=5,patch_size=patchsz, iso=iso)
            dataset_list.append(dataset_train_real)
    dataset_list = torch.utils.data.ConcatDataset(tuple(dataset_list))
    return dataset_list



if __name__=='__main__':
    device="cuda:0"
    x_model_path='/data3/mxx/Noise_generate/Starlight_ours/train_results/pretrained/model_ours/crvd_tune.pt'
    y_model_path='/data3/mxx/Noise_generate/Starlight_ours/train_results/pretrained/model_ours/davis_clean_pretrain.pt'

    x_model=get_model(x_model_path,device)
    y_model=get_model(y_model_path,device)
    x_model.eval();y_model.eval()

    dataset_list=get_dataset()
    test_loader = torch.utils.data.DataLoader(dataset=dataset_list,
                                            batch_size=7)
    calculator=CKACalculator_ours(x_model,y_model,test_loader)
    cka_output = calculator.calculate_cka_matrix()
    print(f"CKA output size: {cka_output.size()}")
    plt.imshow(cka_output.cpu().numpy(), cmap='inferno',vmin=0,vmax=1.0)
    plt.colorbar()
    plt.xlabel(x_model_path.split('/')[-1])
    plt.ylabel(y_model_path.split('/')[-1])
    plt.savefig('/data3/mxx/Noise_generate/Starlight_ours/scripts/cka.png')

    for i, name in enumerate(calculator.module_names_X):
        print(f"Layer {i}: \t{name}")