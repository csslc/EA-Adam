import math
import argparse, yaml
import os
from tqdm import tqdm
import logging
import sys
import time
import importlib
import utils
import glob
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from losses.losses import PerceptualLoss, SSIMLoss
import copy
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F

parser = argparse.ArgumentParser(description='EasySR')
parser.add_argument('--config', type=str, default='configs/swinIR_gan.yml', help = 'pre-config file for training')
parser.add_argument('--test_model_path', type=str, default='pretrained_models/final_model_swinIR_EAdam_195_42_x4_swinIR.pt', help = 'pre-config file for training')
parser.add_argument('--root_img', type=str, default='output/EA-Adam-test-ori', help = 'pre-config file for training')
parser.add_argument('--input_image', type=str, default='test_input', help = 'pre-config file for training')

if __name__ == '__main__':
    import torch
    import torch.nn as nn
    from torch.optim.lr_scheduler import MultiStepLR, StepLR
    from datas.utils import create_datasets
    import matplotlib.pyplot as plt

    args = parser.parse_args()

    if args.config:
       opt = vars(args)
       yaml_args = yaml.load(open(args.config), Loader=yaml.FullLoader)
       opt.update(yaml_args)
    args.popsize = 1
    ## select active gpu devices
    device = None
    if args.gpu_ids is not None and torch.cuda.is_available():
        print('use cuda & cudnn for acceleration!')
        print('the gpu id is: {}'.format(args.gpu_ids))
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        print('use cpu for training!')
        device = torch.device('cpu')
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)

    # load test model
    print('load test model!')
    ## definitions of model
    try:
        model = utils.import_module('models.{}.{}_network'.format(args.model, args.model)).create_model(args)
    except Exception:
        raise ValueError('not supported model type! or something')
    model = model.to(device)
    ckpt = torch.load(args.test_model_path)
    model_state_dict = ckpt['model_state_dict']
    # model_state_dict = ckpt['params']
    if args.model == 'srresnet' or args.model == 'swinIR':
        model_state_dict_new = {}
        for state in model_state_dict:
            name = state.split('module.')[1]
            model_state_dict_new[name] = model_state_dict[state]
        model.load_state_dict(model_state_dict_new, strict=True)
    elif args.model == 'rrdb':
        model.load_state_dict(model_state_dict, strict=True)
    model = model.to(device)
    model = model.eval()
    
    if not os.path.exists(args.root_img):
        os.makedirs(args.root_img)

    # get all input images
    if os.path.isdir(args.input_image):
        image_names = sorted(glob.glob(f'{args.input_image}/*.png'))
    else:
        image_names = [args.input_image]

    # for valid_dataloader in valid_dataloaders:
    for image in image_names:
        with torch.no_grad():
            input_image = Image.open(image).convert('RGB')
            input_image = F.to_tensor(input_image).unsqueeze(0).to(device)
            sr = model(input_image)
            sr = torch.clip(sr, 0, 1)

            bname = os.path.basename(image)

            output_pil = transforms.ToPILImage()(sr[0].cpu())

            output_pil.save(os.path.join(args.root_img, bname))