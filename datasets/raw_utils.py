import numpy as np
import argparse
import cv2
import os


def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = os.path.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(
                        entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


def unpack_raw(im):
    h, w, _ = im.shape
    H, W = h * 2, w * 2
    img2 = np.zeros((H, W))
    img2[0:H:2, 0:W:2] = im[:, :, 0]
    img2[0:H:2, 1:W:2] = im[:, :, 1]
    img2[1:H:2, 0:W:2] = im[:, :, 2]
    img2[1:H:2, 1:W:2] = im[:, :, 3]
    return img2


def pack_raw(im):
    H = im.shape[0]
    W = im.shape[1]
    im = im[:, :, None]
    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :],
                          im[1:H:2, 1:W:2, :]), axis=2)
    return out



def read_raw(path, img_h=1536, img_w=2048):
    raw = np.fromfile(path, dtype='uint8')
    raw = raw.reshape(img_h, img_w)
    return raw


def save_as_png(raw, path, img_name):
    bgr = cv2.cvtColor(raw, cv2.COLOR_BayerRG2RGB_EA)
    cv2.imwrite(path + '/' + img_name + '.png', bgr)
    print(path + '/' + img_name + '.png')


def save_as_raw(raw, path, img_name):
    raw.tofile(path + '/' + img_name + '.raw')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch BM3D')
    parser.add_argument('--raw_dir', type=str, default='./test_video')
    parser.add_argument('--save_dir', type=str, default='./test_video')  # 存放去噪结果的路径

    parser.add_argument('--image_height', type=int, default=2048)
    parser.add_argument('--image_width', type=int, default=2448)

    # 噪声分布的标准差，BM3D有其它形式，可参考官方的实现 https://webpages.tuni.fi/foi/GCF-BM3D/
    parser.add_argument("--noise_sd", type=float, default=0.1)
    # 将Bayer阵列划分四通道，分别去噪。(建议使用，否则BM3D 单通道去噪后，存在颜色错误的可能。试了4次，有1次出来的类似灰度图)
    parser.add_argument('--split_channel', action='store_true')
    opt = parser.parse_args()

    # 下面两个步骤分别运行，使用其中一个时记得注释掉另一个

    # 1.寻找合适的噪声标准差：
    # raw_dir 为一张图片的路径，如 './25db/1.Raw'
    # save_dir 中会生成一系列rgb图片，图片名为使用的噪声标准差，人眼从中选择最优
    # 搜索范围为 [start/div, stop/div) 步长为 step/div

    # start, stop, step, div = 1, 2, 1, 100
    #
    # raw = read_raw(opt.raw_dir, opt.image_height, opt.image_width)
    # save_as_png(raw, opt.save_dir, 'noisy')
    # for i in range(start, stop, step):
    #     noise_sd = i * 1.0 / div
    #     raw_denoise = process_one(raw, noise_sd, opt.split_channel)
    #     save_as_png(raw_denoise, opt.save_dir, '%.4f' % (noise_sd))

    # 2.使用给定的噪声标准差给场景的一个噪声等级去噪：
    # raw_dir 为场景的一个噪声等级的路径，如 './25db'
    # save_dir 中保存去噪后的Raw文件，文件名不变
    # raw_dir_list = scandir(opt.raw_dir, suffix='Raw', full_path=True)
    # for cur_dir in raw_dir_list:
    #     raw_name = os.path.basename(cur_dir)[:-4]
    #     raw = read_raw(cur_dir, opt.image_height, opt.image_width)
    #     raw_denoise = process_one(raw, opt.noise_sd, opt.split_channel)
    #     save_as_raw(raw_denoise, opt.save_dir, raw_name)

    # 3. 批量转png
    # raw_dir为根目录，将目录下的所有递归子文件夹内的raw保存为同名的png
    # save_dir为保存的根目录
    opt.raw_dir = os.path.normpath(opt.raw_dir)
    opt.save_dir = os.path.normpath(opt.save_dir)
    raw_dir_list = scandir(opt.raw_dir, recursive='True', suffix='Raw', full_path=True)
    for cur_dir in raw_dir_list:
        cur_dir = os.path.normpath(cur_dir)
        raw_name = os.path.basename(cur_dir)[:-4]
        cur_save_dir = os.path.dirname(cur_dir.replace(opt.raw_dir, opt.save_dir))
        if not os.path.exists(cur_save_dir):
            os.makedirs(cur_save_dir)
        raw = read_raw(cur_dir, opt.image_height, opt.image_width)
        save_as_png(raw, cur_save_dir, raw_name)