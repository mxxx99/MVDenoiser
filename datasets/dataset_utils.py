import numpy as np
import torch
from scipy.stats import poisson
from numbers import Number
from copy import deepcopy
import torch.nn as nn
from datasets.opts.read_cfg import DictToClass
import yaml
import random

iso_list = [1600,3200,6400,12800,25600]
a_list = [3.513262,6.955588,13.486051,26.585953,52.032536]
b_list = [11.917691,38.117816,130.818508,484.539790,1819.818657]

def pack_gbrg_raw_torch2(raw):  
    '''T H W 或者 H W 均可'''
    H = raw.shape[-2]
    W = raw.shape[-1]
    im = raw.unsqueeze(-3)
    out = torch.cat((im[..., 1:H:2, 0:W:2],# bottom left (IR)
                     im[..., 1:H:2, 1:W:2],# botom right (B)
                     im[..., 0:H:2, 0:W:2],# top left (R)
                     im[..., 0:H:2, 1:W:2]),# top right (G)
                     dim=-3)
    return out

def pack_gbrg_raw_torch(raw):  
    '''T H W'''
    T, H, W = raw.shape
    im = raw.unsqueeze(1)
    out = torch.cat((im[:, :, 1:H:2, 0:W:2],# bottom left (IR)
                     im[:, :, 1:H:2, 1:W:2],# botom right (B)
                     im[:, :, 0:H:2, 0:W:2],# top left (R)
                     im[:, :, 0:H:2, 1:W:2]),# top right (G)
                     dim=1)
    return out


def add_noisy_fromgt(gt_raw,iso):

    '''img:(h w); iso:1600/3200....'''
    gaussian_noise_var = b_list[iso_list.index(iso)]
    a=a_list[iso_list.index(iso)]

    poisson_noisy_img = poisson(np.maximum(gt_raw-240,0)/a).rvs()*a
    gaussian_noise = np.sqrt(gaussian_noise_var)*np.random.randn(gt_raw.shape[0], gt_raw.shape[1])
    noisy_img = poisson_noisy_img + gaussian_noise + 240
    noisy_img = np.minimum(np.maximum(noisy_img,0), 2**12-1)
    #不用考虑noise level
    return noisy_img


class NoisySyn():
    def __init__(self,noise_list='shot_read_uniform_fixed',res_learn=False,device = 'cpu'):
        self.noise_list=noise_list
        self.shot={1600:0.001,3200:0.0016,6400:0.0037,12800:0.0069,25600:0.0132,\
                   25:0.0056,30:0.0146,35:0.0399,40:0.0654}
        self.read={1600:0.0009,3200:0.0019,6400:0.0022,12800:0.0074,25600:0.0129,\
                   25:1.10e-6,30:1.11e-5,35:6.93e-5,40:3.50e-4}
        self.uniform={1600:0.0002,3200:0.0002,6400:0.0016,12800:0.0016,25600:0.0008,\
                      25:0.0002,30:0.0004,35:0.0004,40:0.0003}
        self.fixed={1600:1.99e-4,3200:-8.29e-5,6400:-6.26e-4,12800:-8.84e-4,25600:-2.11e-4,\
                    25:0,30:0,35:0,40:0}
        self.device=device
        self.all_noise={}
        self.res_learn=res_learn
    
    def add_noisy_fromgt_syn(self,x,iso):
        noise=torch.zeros_like(x)
        self.all_noise={}

        if 'shot' in self.noise_list:
            shot_variance=torch.tensor(self.shot[iso],device=self.device)
            shot_noise=torch.poisson(x/torch.abs(shot_variance))*torch.abs(shot_variance)-x
            noise += shot_noise
            self.all_noise['shot'] = noise if self.res_learn else noise+x

        if 'read' in self.noise_list:
            read_variance=torch.tensor(self.read[iso],device=self.device)
            read_noise=torch.randn(x.shape, requires_grad= False, device = self.device)*read_variance
            noise += read_noise
            self.all_noise['read'] = noise if self.res_learn else noise+x

        if 'uniform' in self.noise_list:
            uniform_variance=torch.tensor(self.uniform[iso],device=self.device)
            uniform_noise=torch.rand(x.shape, requires_grad= False, device = self.device)*uniform_variance
            noise += uniform_noise
            self.all_noise['uniform'] = noise if self.res_learn else noise+x

        if 'fixed' in self.noise_list:
            fixed_noise=torch.tensor(self.fixed[iso],device=self.device)
            noise += fixed_noise
            self.all_noise['fixed'] = noise if self.res_learn else noise+x
        
        noisy=torch.clip(noise+x,0,1)
        noise_levels={'shot':shot_variance,'read':read_variance,'uniform':uniform_variance,'fixed':fixed_noise}
        return noisy,self.all_noise,noise_levels


class NoisySyn_gaussian():
    def __init__(self,noise_list='shot_read_uniform_fixed',res_learn=False,device = 'cpu'):
        self.noise_list=noise_list
        self.sigma={15:15/255.0,25:25/255.0,50:50/255.0,70:70/255.0}
        self.device=device
        self.all_noise={}
        self.res_learn=res_learn
    
    def add_noisy_fromgt_syn(self,x,iso):
        noise=torch.zeros_like(x)
        self.all_noise={}

        sigma=torch.tensor(self.sigma[iso],device=self.device)
        noise=torch.randn(x.shape, requires_grad= False, device = self.device)*sigma
        self.all_noise = noise if self.res_learn else noise+x
        
        noisy=torch.clip(noise+x,0,1)
        noise_levels={'gaussian':sigma}
        return noisy,self.all_noise,noise_levels
    

class NoisySyn_locvar():
    def __init__(self,noise_list='shot_read_uniform_fixed',res_learn=False,device = 'cpu'):
        self.noise_list=noise_list
        self.sigma={15:15/255.0,25:25/255.0,50:50/255.0,70:70/255.0}
        self.device=device
        self.all_noise={}
        self.res_learn=res_learn
    
    '''
    def add_noisy_fromgt_syn(self,x,sigma):
        log_shot=np.log(sigma/10000)#sigma=[50,120]
        sigma_shot=np.exp(log_shot)
        line = lambda x: 2.18 * x + 1.20
        log_read=line(log_shot)
        sigma_read=np.exp(log_read)

        loc_var=x*sigma_shot+sigma_read
        # print('loc_var:',sigma_shot,sigma_read)
        noise=torch.randn(x.shape, requires_grad= False, device = x.device)*(np.sqrt(loc_var))
        noisy = torch.clip(x + noise,0,1)
        noise_levels={'locvar':sigma}

        # noise=torch.zeros_like(x)
        # self.all_noise={}

        # sigma=torch.tensor(self.sigma[iso],device=self.device)
        # noise=torch.randn(x.shape, requires_grad= False, device = self.device)*sigma
        # self.all_noise = noise if self.res_learn else noise+x
        
        # noisy=torch.clip(noise+x,0,1)
        # noise_levels={'gaussian':sigma}
        return noisy,self.all_noise,noise_levels
    '''
    def add_noisy_fromgt_syn(self,x,sigma):
        sigma_gauss=random.randint(15,25)/255.0
        sigma_poiss=random.randint(50,100)
        noise_gauss=torch.randn(x.shape, requires_grad= False, device = self.device)*sigma_gauss
        noisy_poiss = torch.poisson(x*sigma_poiss)/sigma_poiss
        noisy=torch.clip(noise_gauss+noisy_poiss,0,1)
        noise_levels={'gaussian':sigma_gauss,'poiss':sigma_poiss}
        return noisy,self.all_noise,noise_levels


def _uniform_batch(min_, max_, shape=(1,), device='cpu'):
    return torch.rand(shape, device=device) * (max_ - min_) + min_

def _normal_batch(scale=1.0, loc=0.0, shape=(1,), device='cpu'):
    return torch.randn(shape, device=device) * scale + loc

def _randint_batch(min_, max_, shape=(1,), device='cpu'):
    return torch.randint(min_, max_, shape, device=device)

def shot_noise(x, k):
    return torch.poisson(torch.clamp(x,min==0) / torch.abs(k)) * torch.abs(k) - x

def gaussian_noise(x, scale, loc=0):
    return torch.randn_like(x) * scale + loc

def tukey_lambda_noise(x, scale, t_lambda=1.4):
    def tukey_lambda_ppf(p, t_lambda):
        assert not torch.any(t_lambda == 0.0)
        return 1 / t_lambda * (p ** t_lambda - (1 - p) ** t_lambda)

    epsilon = 1e-10
    U = torch.rand_like(x) * (1 - 2 * epsilon) + epsilon
    Y = tukey_lambda_ppf(U, t_lambda) * scale

    return Y

def quant_noise(x, q):
    return (torch.rand_like(x) - 0.5) * q

def row_noise(x, scale, loc=0):
    if x.dim() == 4:
        B, _, H, W = x.shape
        noise = (torch.randn((B, H * 2, 1), device=x.device) * scale + loc).repeat((1, 1, W * 2))
        return noise
    elif x.dim() == 5:
        B, T, C, H, W = x.shape
        noise = (torch.randn((B, T, 1, H, 1), device=x.device) * scale + loc).repeat((1, 1, C, 1, W))
        return noise
    else:
        raise NotImplementedError()


class NoisySyn_eld(nn.Module):
    def __init__(self, device='cpu') -> None:
        super().__init__()
        self.opt = yaml.load(open('/data3/mxx/Noise_generate/Starlight_ours_older/datasets/opts/noise_g_virtual.yaml',\
                                  'r',encoding='utf-8').read(),Loader=yaml.FullLoader)
        # self.load_network(self.noise_g, noise_g_path, self.opt['path'].get('strict_load_g', True), None)
        self.device = device
        self.sample_virtual_cameras()
        # print('Current Using Cameras: ', self.cameras)

        self.noise_type = self.opt['noise_type']
        self.read_type = 'TukeyLambda' if 't' in self.noise_type else \
            ('Gaussian' if 'g' in self.noise_type else None)

        self.black_level=2**10;self.white_level=2**14-1
        self.scale = self.white_level - self.black_level


    def sample_virtual_cameras(self):
        self.noise_type = self.opt['noise_type']
        self.param_ranges = self.opt['param_ranges']
        self.virtual_camera_count = self.opt['virtual_camera_count']
        self.sample_strategy = self.opt['sample_strategy']
        self.shuffle=False

        # sampling strategy
        sample = self.split_range if self.sample_strategy == 'coverage' else self.uniform_range

        # overall system gain
        self.k_range = torch.tensor(self.param_ranges['K'], device=self.device)

        # read noise
        if 'g' in self.noise_type:
            read_slope_range = self.param_ranges['Gaussian']['slope']
            read_bias_range = self.param_ranges['Gaussian']['bias']
            read_sigma_range = self.param_ranges['Gaussian']['sigma']
        elif 't' in self.noise_type:
            read_slope_range = self.param_ranges['TukeyLambda']['slope']
            read_bias_range = self.param_ranges['TukeyLambda']['bias']
            read_sigma_range = self.param_ranges['TukeyLambda']['sigma']
            read_lambda_range = self.param_ranges['TukeyLambda']['lambda']
            self.tukey_lambdas = sample(self.virtual_camera_count, read_lambda_range, self.shuffle, self.device)
            self.tukey_lambdas = nn.Parameter(self.tukey_lambdas, False)
        if 'g' in self.noise_type or 't' in self.noise_type:
            self.read_slopes = sample(self.virtual_camera_count, read_slope_range, self.shuffle, self.device)
            self.read_biases = sample(self.virtual_camera_count, read_bias_range, self.shuffle, self.device)
            self.read_sigmas = sample(self.virtual_camera_count, read_sigma_range, self.shuffle, self.device)
            self.read_slopes = nn.Parameter(self.read_slopes, False)
            self.read_biases = nn.Parameter(self.read_biases, False)
            self.read_sigmas = nn.Parameter(self.read_sigmas, False)

        # row noise
        if 'r' in self.noise_type:
            row_slope_range = self.param_ranges['Row']['slope']
            row_bias_range = self.param_ranges['Row']['bias']
            row_sigma_range = self.param_ranges['Row']['sigma']
            self.row_slopes = sample(self.virtual_camera_count, row_slope_range, self.shuffle, self.device)
            self.row_biases = sample(self.virtual_camera_count, row_bias_range, self.shuffle, self.device)
            self.row_sigmas = sample(self.virtual_camera_count, row_sigma_range, self.shuffle, self.device)
            self.row_slopes = nn.Parameter(self.row_slopes, False)
            self.row_biases = nn.Parameter(self.row_biases, False)
            self.row_sigmas = nn.Parameter(self.row_sigmas, False)

        # color bias
        if 'c' in self.noise_type:
            self.color_bias_count = self.param_ranges['ColorBias']['count']
            ## ascend sigma
            color_bias_sigmas = self.split_range_overlap(self.color_bias_count,
                                                         self.param_ranges['ColorBias']['sigma'],
                                                         overlap=0.1)
            self.color_biases = torch.tensor(np.array([
                [
                    random.uniform(*self.param_ranges['ColorBias']['bias']) + \
                        torch.randn(4).numpy() * random.uniform(*color_bias_sigmas[i]).cpu().numpy()
                    for _ in range(self.color_bias_count)
                ] for i in range(self.virtual_camera_count)
            ]), device=self.device)
            self.color_biases = nn.Parameter(self.color_biases, False)

    @staticmethod
    def uniform_range(splits, range_, shuffle=True, device='cuda'):
        results = [random.uniform(*range_) for _ in range(splits)]
        if shuffle:
            random.shuffle(results)
        return torch.tensor(results, device=device)

    @staticmethod
    def split_range(splits, range_, shuffle=True, device='cuda'):
        length = range_[1] - range_[0]
        i_length = length / (splits - 1)
        results = [range_[0] + i_length * i for i in range(splits)]
        if shuffle:
            random.shuffle(results)
        return torch.tensor(results, device=device)

    @staticmethod
    def split_range_overlap(splits, range_, overlap=0.5, device='cuda'):
        length = range_[1] - range_[0]
        i_length = length / (splits * (1 - overlap) + overlap)
        results = []
        for i in range(splits):
            start = i_length * (1 - overlap) * i
            results.append([start, start + i_length])
        return torch.tensor(results, device=device)

    def sample_overall_system_gain(self, batch_size, for_video):
        if self.current_camera is None:
            index = _randint_batch(0, self.virtual_camera_count, (batch_size,), self.device)
            self.current_camera = index
        log_K_max = torch.log(self.k_range[1])
        log_K_min = torch.log(self.k_range[0])
        log_K = _uniform_batch(log_K_min, log_K_max, (batch_size, 1, 1, 1), self.device)
        if for_video:
            log_K = log_K.unsqueeze(-1)
        self.log_K = log_K
        self.cur_batch_size = batch_size
        return torch.exp(log_K)

    def sample_read_sigma(self):
        slope = self.read_slopes[self.current_camera]
        bias = self.read_biases[self.current_camera]
        sigma = self.read_sigmas[self.current_camera]
        mu = self.log_K.squeeze() * slope + bias
        sample = _normal_batch(sigma, mu, (self.cur_batch_size,), self.device)
        return torch.exp(sample).reshape(self.log_K.shape)

    def sample_tukey_lambda(self, batch_size, for_video):
        tukey_lambda = self.tukey_lambdas[self.current_camera].reshape(batch_size, 1, 1, 1)
        if for_video:
            tukey_lambda = tukey_lambda.unsqueeze(-1)
        return tukey_lambda

    def sample_row_sigma(self):
        slope = self.row_slopes[self.current_camera]
        bias = self.row_biases[self.current_camera]
        sigma = self.row_sigmas[self.current_camera]
        mu = self.log_K.squeeze() * slope + bias
        sample = _normal_batch(sigma, mu, (self.cur_batch_size,), self.device)
        return torch.exp(sample).reshape(self.log_K.squeeze(-3).shape)

    def sample_color_bias(self, batch_size, for_video):
        i_range = (self.k_range[1] - self.k_range[0]) / self.color_bias_count
        index = ((torch.exp(self.log_K.squeeze()) - self.k_range[0]) // i_range).long()
        color_bias = self.color_biases[self.current_camera, index]
        color_bias = color_bias.reshape(batch_size, 4, 1, 1)
        if for_video:
            color_bias = color_bias.unsqueeze(1)
        return color_bias

    @staticmethod
    def add_noise(img, noise, noise_params):
        img_=img.detach().clone()
        tail = [1 for _ in range(img_.dim() - 1)]
        ratio = noise_params['isp_dgain']
        scale = noise_params['scale']
        for n in noise.values():
            img_ += n
        img_/=scale
        img_ = img_ * ratio
        return torch.clamp(img_, max=1.0)

    @torch.no_grad()
    def add_noisy_fromgt_syn(self,img,ratio=1,vcam_id=None):
        ratio=random.randint(100,300);vcam_id=random.randint(0,4)
        img = torch.clamp(img,0,1) * self.scale / ratio
        img=img.unsqueeze(0).transpose(1,2)
        b = img.size(0)
        for_video = True
        self.current_camera = vcam_id * torch.ones((b,), dtype=torch.long, device=self.device) \
                              if vcam_id is not None else None

        K = self.sample_overall_system_gain(b, for_video)
        noise = {};noise_params = {'isp_dgain':ratio, 'scale': self.scale};self.all_noise={}
        # shot noise
        if 'p' in self.noise_type:
            _shot_noise = shot_noise(img, K)
            noise['shot'] = _shot_noise
            noise_params['shot'] = K.squeeze()
        # read noise
        if 'g' in self.noise_type:
            read_param = self.sample_read_sigma()
            _read_noise = gaussian_noise(img, read_param)
            noise['read'] = _read_noise
            noise_params['read'] = read_param.squeeze()
        elif 't' in self.noise_type:
            tukey_lambda = self.sample_tukey_lambda(b, for_video)
            read_param = self.sample_read_sigma()
            _read_noise = tukey_lambda_noise(img, read_param, tukey_lambda)
            noise['read'] = _read_noise
            noise_params['read'] = {
                'sigma': read_param,
                'tukey_lambda': tukey_lambda
            }
        # row noise
        if 'r' in self.noise_type:
            row_param = self.sample_row_sigma()
            _row_noise = row_noise(img, row_param)
            noise['row'] = _row_noise
            noise_params['row'] = row_param.squeeze()
        # quant noise
        if 'q' in self.noise_type:
            _quant_noise = quant_noise(img, 1)
            noise['quant'] = _quant_noise
        # color bias
        if 'c' in self.noise_type:
            color_bias = self.sample_color_bias(b, for_video)
            noise['color_bias'] = color_bias

        img_lq = self.add_noise(img, noise, noise_params)
        img_lq=img_lq.squeeze(0).transpose(0,1)

        return img_lq,self.all_noise,noise_params

    def cpu(self):
        super().cpu()
        self.device = 'cpu'
        return self

    def cuda(self, device=None):
        super().cuda(device)
        self.device = 'cuda'
        return self


def pack_bggr_raw_torch(raw):
    '''T H W 或者 H W 均可'''
    H = raw.shape[-2]
    W = raw.shape[-1]
    im = raw.unsqueeze(-3)
    out = torch.cat((im[..., 0:H:2, 0:W:2],# top left (R)
                     im[..., 0:H:2, 1:W:2],# top right (G)
                     im[..., 1:H:2, 0:W:2],# bottom left (IR)
                     im[..., 1:H:2, 1:W:2]),# botom right (R)
                     dim=-3)
    return out

def normalize_raw_torch(raw):
    black_level = 240
    white_level = 2 ** 12 - 1
    raw = torch.clamp(raw.type(torch.float32) - black_level, 0) / (white_level - black_level)
    return raw

def normalize_raw_rnvd(raw):
    black_level = 0
    white_level = 2 ** 8 - 1    #255,uint8精度
    raw = torch.clamp(raw.type(torch.float32) - black_level, 0,255) / (white_level - black_level)
    return raw

def read_16bit_raw(filename, height = 1280, width = 2160):
    pgmf = open(filename, 'rb')

    raster = []
    for y in range(height):
        row = []
        for yy in range(width):
            low_bits = ord(pgmf.read(1))
            row.append(low_bits+255*ord(pgmf.read(1)))
        raster.append(row)

    pgmf.close()
    return np.asarray(raster)

def raw_to_4(im):
    native_1 = im[0::2,0::2] # top left (R)
    native_2 = im[1::2,0::2] # bottom left (IR)
    native_3 = im[0::2,1::2] # top right (G)
    native_4 = im[1::2,1::2] # botom right (B)

    col_list = [native_1,  native_3, native_4, native_2]
    return np.asarray(col_list).transpose(1,2,0)