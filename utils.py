import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np
import datetime
import os
import sys
import cv2
from math import exp
from pytorch_msssim import ssim
import importlib
import matplotlib.pyplot as plt
import random


def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to YCbCr.

    Args:
        image (torch.Tensor): RGB Image to be converted to YCbCr.

    Returns:
        torch.Tensor: YCbCr version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}".format(image.shape))

    image = image / 255. ## image in range (0, 1)
    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    y: torch.Tensor = 65.481 * r + 128.553 * g + 24.966 * b + 16.0
    cb: torch.Tensor = -37.797 * r + -74.203 * g + 112.0 * b + 128.0
    cr: torch.Tensor = 112.0 * r + -93.786 * g + -18.214 * b + 128.0

    return torch.stack((y, cb, cr), -3)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def dis_result(name, experiment_path, epoch, stat_dict):
    
    dis_name = name + '_' + str(epoch)
    path = os.path.join(experiment_path, 'display')
    if not os.path.exists(path):
        os.makedirs(path)
    l1 = stat_dict[name]['l1']
    l1 = torch.tensor(list(l1.values()))
    lpips = stat_dict[name]['lpipss']
    lpips = torch.tensor(list(lpips.values()))
    print(l1)
    print(lpips)
    plt.figure(dis_name)
    plt.plot(l1, lpips, 'bo')
    plt.xlabel('L_1')
    plt.ylabel('lpips')
    plt.title(dis_name)
    plt.grid(True)
    plt.draw()
    plt.savefig(path + '/' + dis_name + '.jpg')
    plt.cla()


def prepare_qat(model):
    ## fuse model
    model.module.fuse_model()
    ## qconfig and qat-preparation & per-channel quantization
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    model = torch.quantization.prepare_qat(model, inplace=True)
    return model
    
def imwrite(img, img_path):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)

def import_module(name):
    return importlib.import_module(name)

def calc_psnr(sr, hr):
    sr, hr = sr.double(), hr.double()
    diff = (sr - hr) / 255.00
    mse  = diff.pow(2).mean()
    psnr = -10 * math.log10(mse)                    
    return float(psnr)

def calc_ssim(sr, hr):
    ssim_val = ssim(sr, hr, size_average=True)
    return float(ssim_val)
    
def ndarray2tensor(ndarray_hwc):
    ndarray_chw = np.ascontiguousarray(ndarray_hwc.transpose((2, 0, 1)))
    ndarray_chw = ndarray_chw.copy()
    tensor = torch.from_numpy(ndarray_chw).float()
    return tensor

def cur_timestamp_str():
    now = datetime.datetime.now()
    year = str(now.year)
    month = str(now.month).zfill(2)
    day = str(now.day).zfill(2)
    hour = str(now.hour).zfill(2)
    minute = str(now.minute).zfill(2)

    content = "{}-{}{}-{}{}".format(year, month, day, hour, minute)
    return content


class ExperimentLogger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
	    self.terminal = stream
	    self.log = open(filename, 'a')
    def write(self, message):
	    self.terminal.write(message)
	    self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def get_stat_dict(args):
    stat_dict = {
        'epochs': 0,
        'losses': [],
    }
    _per_eval = {'psnrs': {},
            'ssims': {},
            'lpipss': {},
            'best_psnr': {
                'value': {},
                'epoch': {}
            },
            'best_ssim': {
                'value': {},
                'epoch': {}
            },
            'best_lpips':{
                'value': {},
                'epoch': {}
            }}
    evl_names = args.eval_sets.get('names')
    for name in evl_names:
        a = {name: _per_eval}
        stat_dict.update(a)
    return stat_dict
