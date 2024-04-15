import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    # print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


if __name__ == '__main__':
    # 分解案例
    dwt_module=DWT()
    x=Image.open('./2.png')
    # x=Image.open('./mountain.png')
    x=transforms.ToTensor()(x)
    print(torch.max(x),torch.min(x))
    x=torch.unsqueeze(x,0)
    x=transforms.Resize(size=(256,256))(x)
    subbands=dwt_module(x)

    title=['LL','HL','LH','HH']

    plt.figure()
    for i in range(4):
        plt.subplot(2,2,i+1)
        temp=torch.permute(subbands[0,3*i:3*(i+1),:,:],dims=[1,2,0])
        print(torch.max(temp),torch.min(temp))
        plt.imshow(temp)
        plt.title(title[i])
        plt.axis('off')
    # plt.show()
    plt.savefig('after_dwt2.png')

    # 重构案例
    title=['Original Image','Reconstruction Image']
    reconstruction_img=IWT()(subbands).cpu()
    # ssim_value=ssim(x,reconstruction_img)  # 计算原图与重构图之间的结构相似度
    # print("SSIM Value:",ssim_value) # tensor(1.)
    show_list=[torch.permute(x[0],dims=[1,2,0]),torch.permute(reconstruction_img[0],dims=[1,2,0])]
    print(torch.max(show_list[1]-show_list[0]),torch.min(show_list[1]-show_list[0]))

    plt.figure()
    for i in range(2):
        plt.subplot(1,2,i+1)
        plt.imshow(show_list[i])
        plt.title(title[i])
        plt.axis('off')
    # plt.show()
    plt.savefig('after_iwt2.png')

