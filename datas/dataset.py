import os
import glob
import random
import pickle

import numpy as np
import imageio
import torch
import torch.utils.data as data
import skimage.color as sc
import time

def ndarray2tensor(ndarray_hwc):
    ndarray_chw = np.ascontiguousarray(ndarray_hwc.transpose((2, 0, 1)))
    ndarray_chw = ndarray_chw.copy()
    tensor = torch.from_numpy(ndarray_chw).float()
    return tensor

def crop_patch(lr, hr, patch_size, scale, augment=True):
    # crop patch randomly
    lr_h, lr_w, _ = lr.shape
    hp = patch_size
    lp = patch_size // scale
    lx, ly = random.randrange(0, lr_w - lp + 1), random.randrange(0, lr_h - lp + 1)
    hx, hy = lx * scale, ly * scale
    lr_patch, hr_patch = lr[ly:ly+lp, lx:lx+lp, :], hr[hy:hy+hp, hx:hx+hp, :]
    # augment data
    if augment:
        hflip = random.random() > 0.5
        vflip = random.random() > 0.5
        rot90 = random.random() > 0.5
        if hflip: lr_patch, hr_patch = lr_patch[:, ::-1, :], hr_patch[:, ::-1, :]
        if vflip: lr_patch, hr_patch = lr_patch[::-1, :, :], hr_patch[::-1, :, :]
        if rot90: lr_patch, hr_patch = lr_patch.transpose(1,0,2), hr_patch.transpose(1,0,2)
        # numpy to tensor
    lr_patch, hr_patch = ndarray2tensor(lr_patch), ndarray2tensor(hr_patch)
    return lr_patch, hr_patch

def crop_EA_patch(lr, hr, patch_size, scale):

    hr, lr = ndarray2tensor(hr), ndarray2tensor(lr)
    hp = patch_size
    lp = patch_size // scale

    # crop patch
    unfolded_hrpatch = hr.unfold(1, hp, hp).unfold(2, hp, hp)
    unfolded_hrpatch = unfolded_hrpatch.reshape(3,unfolded_hrpatch.shape[1]*unfolded_hrpatch.shape[2],hp,hp).permute(1,0,2,3)
    # max_b = 50
    if unfolded_hrpatch.shape[0]>100:
        unfolded_hrpatch = unfolded_hrpatch[25:75,...]

    unfolded_lrpatch = lr.unfold(1, lp, lp).unfold(2, lp, lp)
    unfolded_lrpatch = unfolded_lrpatch.reshape(3,unfolded_lrpatch.shape[1]*unfolded_lrpatch.shape[2],lp,lp).permute(1,0,2,3)
    if unfolded_lrpatch.shape[0]>100:
        unfolded_lrpatch = unfolded_lrpatch[25:75,...]

    return unfolded_lrpatch, unfolded_hrpatch

class DATASET(data.Dataset):
    def __init__(
        self, HR_folder, LR_folder, 
        train=True, train_for_EA = True, augment=True, scale=2, colors=1, 
        patch_size=96, repeat=168
    ):
        super(DATASET, self).__init__()
        self.HR_folder = HR_folder
        self.LR_folder = LR_folder
        self.augment   = augment
        self.scale = scale
        self.colors = colors
        self.patch_size = patch_size
        self.repeat = repeat
        self.nums_trainset = 0
        self.train = train
        self.train_for_EA = train_for_EA

        ## for raw png images
        self.hr_filenames = []
        self.lr_filenames = []

        ## generate dataset
        if self.train:
            if self.train_for_EA:
                hr_folders = os.listdir(self.HR_folder)
                lr_folders = os.listdir(self.LR_folder)
                j=0
                for i in range(len(hr_folders)):
                    hr_filename = os.path.join(self.HR_folder, hr_folders[i])
                    if 'x4' in lr_folders[i]:
                        lr_name = hr_folders[i].split('.png')[0] + 'x4.png'
                        lr_filename = os.path.join(self.LR_folder, lr_name)
                    else:
                        lr_filename = os.path.join(self.LR_folder, lr_folders[i])
                    self.hr_filenames.append(hr_filename)
                    self.lr_filenames.append(lr_filename)
                    j = j + 1
                    if j > 15: 
                        break
                self.nums_trainset = len(self.hr_filenames)
            else:
                hr_folders = os.listdir(self.HR_folder)
                lr_folders = os.listdir(self.LR_folder)
                for i in range(len(hr_folders)):
                    hr_filename = os.path.join(self.HR_folder, hr_folders[i])
                    if 'x4' in lr_folders[i]:
                        lr_name = hr_folders[i].split('.png')[0] + 'x4.png'
                        lr_filename = os.path.join(self.LR_folder, lr_name)
                    else:
                        lr_filename = os.path.join(self.LR_folder, lr_folders[i])
                    self.hr_filenames.append(hr_filename)
                    self.lr_filenames.append(lr_filename)
                self.nums_trainset = len(self.hr_filenames)
        else:
            hr_folders = os.listdir(self.HR_folder)
            lr_folders = os.listdir(self.LR_folder)
            for i in range(len(hr_folders)):
                hr_filename = os.path.join(self.HR_folder, hr_folders[i])
                if 'x4' in lr_folders[i]:
                    lr_name = hr_folders[i].split('.png')[0] + 'x4.png'
                    lr_filename = os.path.join(self.LR_folder, lr_name)
                else:
                    lr_filename = os.path.join(self.LR_folder, lr_folders[i])
                self.hr_filenames.append(hr_filename)
                self.lr_filenames.append(lr_filename)
            self.nums_trainset = len(self.hr_filenames)
            

    def __len__(self):
        if self.train_for_EA:
            return self.nums_trainset
        if self.train:
            return self.nums_trainset * self.repeat
        else:
            return self.nums_trainset

    def __getitem__(self, idx):
        # get periodic index
        idx = idx % self.nums_trainset
        # get whole image
        hr, lr = imageio.imread(self.hr_filenames[idx], pilmode="RGB"), imageio.imread(self.lr_filenames[idx], pilmode="RGB")
        if self.colors == 1:
            hr = sc.rgb2ycbcr(hr)[:, :, 0:1]
            lr = sc.rgb2ycbcr(lr)[:, :, 0:1]
            
        if self.train:
            if not self.train_for_EA:
                train_lr_patch, train_hr_patch = crop_patch(lr, hr, self.patch_size, self.scale, True)
                if train_lr_patch.shape[0]!=3 or train_lr_patch.shape[1]!=32 or train_lr_patch.shape[2]!=32:
                    print('error lr')
                    
                if train_hr_patch.shape[0]!=3 or train_hr_patch.shape[1]!=128 or train_hr_patch.shape[2]!=128:
                    print('error hr')
                return train_lr_patch, train_hr_patch
            else:
                train_EA_lr_patch, train_EA_hr_patch = crop_EA_patch(lr, hr, self.patch_size, self.scale)
                return train_EA_lr_patch, train_EA_hr_patch

        hr_image, lr_image = ndarray2tensor(hr_image), ndarray2tensor(lr_image)
        return lr_image, hr_image

if __name__ == '__main__':
    HR_folder = '/home/notebook/data/group/SunLingchen/dataset/Flickr2K/Flickr2K_HR'
    LR_folder = '/home/notebook/data/group/SunLingchen/dataset/Flickr2K/Flickr2K_LR_bicubic/X4'
    argment   = True
    training_data = DATASET(HR_folder, LR_folder, augment=True, scale=4, colors=3, patch_size=128, repeat=8)

    print("numner of sample: {}".format(len(training_data)))
    start = time.time()
    for idx in range(10):
        lr, hr = training_data[idx]
        print(hr.shape, lr.shape)
    end = time.time()
    print(end - start)