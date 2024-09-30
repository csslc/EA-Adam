import math
import argparse, yaml
import utils
import os
from tqdm import tqdm
import logging
import sys
import time
import importlib
import glob
import lpips
from losses.losses import PerceptualLoss, SSIMLoss, GANLoss
import numpy as np
from optimizer.Adam_gn import Adam
import copy
from collections import OrderedDict

parser = argparse.ArgumentParser(description='EasySR')
## yaml configuration files
parser.add_argument('--config', type=str, default='configs/fusion_srresnet.yml', help = 'pre-config file for training')
parser.add_argument('--test_model_path', type=str, default='/home/notebook/data/group/SunLingchen/code/EA-Adam-main/experiments/test_fusion_srgan_Adam-2024-0923-1511/models/model_x4_10.pt', help = 'the fusion model for N expert models')
parser.add_argument('--expert_path', type=str, default='experiments/EAdam_srresnet_EAiter200-2024-0911-1014/fusion_experts', help = 'N (popsize) expert models')
parser.add_argument('--final_model_path', type=str, default='experiments/fusion_srresnet', help = 'the path to save the final fusion model')

if __name__ == '__main__':
    args = parser.parse_args()
    if args.config:
       opt = vars(args)
       yaml_args = yaml.load(open(args.config), Loader=yaml.FullLoader)
       opt.update(yaml_args)
    ## set visibel gpu   
    gpu_ids_str = str(args.gpu_ids).replace('[','').replace(']','')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_ids_str)
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim.lr_scheduler import MultiStepLR, StepLR
    from datas.utils import create_datasets

    if args.config:
       opt = vars(args)
       yaml_args = yaml.load(open(args.config), Loader=yaml.FullLoader)
       opt.update(yaml_args)

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

    ## create dataset for calculating weight
    _, _, _, test_dataloaders = create_datasets(args)

    ## definition of loss and optimizer
    loss_func = nn.L1Loss()
    loss_lpips_func = lpips.LPIPS(net='alex', spatial = False).to(device)
    perceptual_loss = PerceptualLoss(args.perceptual_opt['layer_weights']).to(device)
    ssim_loss = SSIMLoss().to(device)
    
    ## load expert model
    ## definitions of expert model
    try:
        m_expert = utils.import_module('models.{}.{}_network'.format(args.experts_model, args.experts_model)).create_model(args)
    except Exception:
        raise ValueError('not supported model type! or something')
    m_expert = nn.DataParallel(m_expert).to(device)
    m_expert = m_expert.eval()

    # optimizer = torch.optim.Adam(m_expert.parameters(), lr=args.lr)

    # if args.is_qat:
    #     scheduler = StepLR(optimizer, step_size=args.decays, gamma=args.gamma)
    # else:
    #     scheduler = MultiStepLR(optimizer, milestones=args.decays, gamma=args.gamma)
    
    ckpt = torch.load(args.test_model_path)
    m_expert.load_state_dict(ckpt['model_state_dict'], strict = True)

    ## start recording weight
    weights_all = []
    weights_data =[]
    for test_dataloader in test_dataloaders:
        name = test_dataloader['name']
        if name == 'div2k':
            loader = test_dataloader['dataloader']
            ind_img = 0
            num_data = 100 # only 100 images are used
            t = 0
            for lr, hr in tqdm(loader, ncols=80):
                # lr, hr = batch
                weights = []
                if t >= num_data:
                    break
                if args.range == 1:
                    lr = lr / 255.
                lr = lr.to(device)
                lr = lr[:, :, int(lr.shape[2]/4):int(lr.shape[2]*3/4), int(lr.shape[3]/4):int(lr.shape[3]*3/4)]
                
                _, weights = m_expert(lr, weights)
                
                if t!= 0:
                    weights_all = [weights_all[i] + weights[i] for i in range(len(weights_all))]
                else:
                    weights_all = weights
                del weights, _, lr, hr
                torch.cuda.empty_cache()
                t += 1
    
    ## start calculating avarage weight
    avg_weights = [weights_all[i]/num_data for i in range(len(weights_all))]

    ## validation
    # load all expert models in expert_models
    expert_path = args.expert_path
    files = os.listdir(expert_path)
    expert_models = []
    for file in files:
        try:
            expert_model = utils.import_module('models.{}.{}_network'.format(args.expert_model, args.expert_model)).create_model(args)
        except Exception:
            raise ValueError('not supported model type! or something')
        expert_model = nn.DataParallel(expert_model).to(device)
        fm = expert_path + '/' + file
        ckpt = torch.load(fm)
        expert_model.load_state_dict(ckpt['model_state_dict'])
        expert_model = expert_model.eval()
        expert_models.append(expert_model)
    
    # get the final model via avg_weights and expert_models
    expert_model_0= copy.deepcopy(expert_models[0])
    expert_model_0_par = dict(expert_model_0.named_parameters())
    expert_model_final = copy.deepcopy(expert_models[args.num_network-1])
    expert_model_final_par = dict(expert_model_final.named_parameters())
    num_weight = len(expert_model_0_par.keys()) // 2

    for j in range(args.num_network):
        if (j==args.num_network-1):
            continue
        if j == 0:
            nw = 0
            for k in expert_model_0_par.keys():
                if k[-1] == 't':
                    expert_model_final_par[k].data.mul_(avg_weights[nw][0][args.num_network-1]).add_(expert_model_0_par[k].data, alpha=avg_weights[nw][0][j])
                if k[-1] == 's':
                    expert_model_final_par[k].data.mul_(avg_weights[nw][0][args.num_network-1]).add_(expert_model_0_par[k].data, alpha=avg_weights[nw][0][j])
                    nw += 1
            print(nw)
        else:
            nw = 0
            expert_model_j= copy.deepcopy(expert_models[j])
            expert_model_j_par = dict(expert_model_j.named_parameters())
            
            for k in expert_model_0_par.keys():
                if k[-1] == 't':
                    expert_model_final_par[k].data.add_(expert_model_j_par[k].data, alpha=avg_weights[nw][0][j])
                if k[-1] == 's':
                    expert_model_final_par[k].data.add_(expert_model_j_par[k].data, alpha=avg_weights[nw][0][j])
                    nw += 1
                # model_final_par[k].data.add_(model_j_par[k].data, alpha=w_fuse[j])
    model = copy.deepcopy(expert_model_final)

    del expert_models, expert_model_0, expert_model_final, expert_model_j
    torch.cuda.empty_cache()

    # save the final model
    if not os.path.exists(args.final_model_path):
        os.makedirs(args.final_model_path)
    saved_model_path = os.path.join(args.final_model_path, 'final_model_EAdam_x{}_{}.pt'.format(args.scale, args.expert_model))
                    # torch.save(model.state_dict(), saved_model_path)
    torch.save({
        'model_state_dict': model.state_dict()
    }, saved_model_path)