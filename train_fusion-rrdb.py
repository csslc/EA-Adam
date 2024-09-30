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
import copy
from losses.DISTS.DISTS_pytorch.DISTS_pt import DISTS
from EA_operator.sel_model import model_ema
from losses.LDL_loss import get_refined_artifact_map

parser = argparse.ArgumentParser(description='EasySR')
## yaml configuration files
parser.add_argument('--config', type=str, default='configs/fusion_rrdb.yml', help = 'pre-config file for training')
parser.add_argument('--expert_path', type=str, default='/home/notebook/data/group/SunLingchen/code/SimpleIR-main/experiments/RRDB_EA-Adam_DF2K_EAiter200-2024-0522-1736/5_23_fusion_13', help = 'resume training or not')
parser.add_argument('--resume_expert', type=str, default=None, help = 'resume the expert model')
parser.add_argument('--resume_d', type=str, default='/home/notebook/data/group/SunLingchen/code/SimpleIR-main/experiments/RRDB_EA-Adam_DF2K_EAiter200-2024-0522-1736/models/model_d_x4_13_4.pt', help = 'resume training or not')

parser.add_argument('--is_l1loss', type=float, default= 0.1, help = 'use l1 loss for optimization')
parser.add_argument('--is_lpips', type=float, default=False, help = 'use lpips loss for optimization')
parser.add_argument('--is_dists', type=float, default=False, help = 'use dists loss for optimization')
parser.add_argument('--is_perceptual', type=float, default= 1.0, help = 'use perceptual loss for optimization')
parser.add_argument('--is_ssim', type=float, default=False, help = 'use ssim loss for optimization')
parser.add_argument('--is_LDL', type=float, default=False, help = 'use LDL loss for optimization')
parser.add_argument('--is_gan', type=float, default=0.005, help = 'use gan loss for optimization')


if __name__ == '__main__':
    args = parser.parse_args()
    utils.setup_seed(0)
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

    ## create dataset for training and validating
    train_dataloader, _, valid_dataloaders = create_datasets(args)

    ## definitions of model
    try:
        model = utils.import_module('models.{}.{}_network'.format(args.model, args.model)).create_model(args)
    except Exception:
        raise ValueError('not supported model type! or something')
    model = nn.DataParallel(model).to(device)

    model_d = utils.import_module('models.discriminator.{}_network'.format(args.network_d.get('type'))).create_model(args)
    model_d = nn.DataParallel(model_d).to(device)

    ## definition of loss and optimizer
    loss_func = nn.L1Loss()
    loss_lpips_func = lpips.LPIPS(net='alex', spatial = False).to(device)
    perceptual_loss = PerceptualLoss(args.perceptual_opt['layer_weights']).to(device)
    ssim_loss = SSIMLoss().to(device)
    DISTS_func = DISTS().to(device)
    gan_loss = GANLoss(args.gan_opt.get('gan_type')).to(device)
    
    ## load expert model
    ## definitions of expert model
    try:
        m_expert = utils.import_module('models.{}.{}_network'.format(args.experts_model, args.experts_model)).create_model(args)
    except Exception:
        raise ValueError('not supported model type! or something')
    # m_expert = nn.DataParallel(m_expert).to(device)
    m_expert = m_expert.to(device)
    optimizer = torch.optim.Adam(m_expert.parameters(), lr=args.lr)
    if args.is_qat:
        scheduler = StepLR(optimizer, step_size=args.decays, gamma=args.gamma)
    else:
        scheduler = MultiStepLR(optimizer, milestones=args.decays, gamma=args.gamma)

    optimizer_d = torch.optim.Adam(model_d.parameters(), lr=args.lr)
    if args.is_qat:
        scheduler_d = StepLR(optimizer_d, step_size=args.decays, gamma=args.gamma)
    else:
        scheduler_d = MultiStepLR(optimizer_d, milestones=args.decays, gamma=args.gamma)

    expert_path = args.expert_path
    files = os.listdir(expert_path)
    i = 0
    paras_dict = []
    for file in files:
        if not os.path.isdir(file):
            fm = expert_path + '/' + file
            ckpt = torch.load(fm)
            # models_expert[i].load_state_dict(ckpt['model_state_dict'])
            # para_dict = dict(models_expert[i].named_parameters())
            paras_dict.append(ckpt['model_state_dict'])
            i += 1
    
    experts_model_para = {}
    for k, v in paras_dict[0].items():
        for i in range(args.num_network):
            if i == 0:
                up_weight = v.unsqueeze(0)
            else:
                up_weight = torch.cat((up_weight, paras_dict[i][k].unsqueeze(0)), 0)
        experts_model_para[k] = up_weight
    m_expert.load_state_dict(experts_model_para, strict = False)

    # definition of EMA
    if args.is_LDL:    
        m_expert_ema = copy.deepcopy(m_expert)
        for p in m_expert_ema.parameters():
            p.requires_grad = False  # copy net_g weight
        m_expert_ema.eval()

    ## resume training
    start_epoch = 1
    if args.resume_expert is not None:
        ckpt_files = glob.glob(os.path.join(args.resume, 'models', "*.pt"))
        if len(ckpt_files) != 0:
        # if True:
            ckpt_files = sorted(ckpt_files, key=lambda x: int(x.replace('.pt','').split('_')[-1]))
            ckpt = torch.load(ckpt_files[-1])
            prev_epoch = ckpt['epoch']

            start_epoch = prev_epoch + 1
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            stat_dict = ckpt['stat_dict']
            ## reset folder and param
            experiment_path = args.resume
            log_name = os.path.join(experiment_path, 'log.txt')
            experiment_model_path = os.path.join(experiment_path, 'models')
            print('select {}, resume training from epoch {}.'.format(ckpt_files[-1], start_epoch))
    else:
        ## auto-generate the output logname
        experiment_name = None
        timestamp = utils.cur_timestamp_str()
        if args.log_name is None:
            experiment_name = '{}-{}-x{}-{}'.format(args.model, 'int8' if args.is_qat else 'fp32', args.scale, timestamp)
        else:
            experiment_name = '{}-{}'.format(args.log_name, timestamp)
        experiment_path = os.path.join(args.log_path, experiment_name)
        log_name = os.path.join(experiment_path, 'log.txt')
        stat_dict = utils.get_stat_dict(args)
        ## create folder for ckpt and stat
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)
        experiment_model_path = os.path.join(experiment_path, 'models')
        if not os.path.exists(experiment_model_path):
            os.makedirs(experiment_model_path)
        ## save training paramters
        exp_params = vars(args)
        exp_params_name = os.path.join(experiment_path, 'config.yml')
        with open(exp_params_name, 'w') as exp_params_file:
            yaml.dump(exp_params, exp_params_file, default_flow_style=False)

    if args.resume_d is not None:
        
        ckpt = torch.load(args.resume_d)
        # model_state = {}
        # for k in ckpt['model_state_dict']:
        #     k_name = 'module.' + k
        #     model_state[k_name] = ckpt['model_state_dict'][k]
        model_d.load_state_dict(ckpt['model_state_dict'])
            
    ## print architecture of model
    time.sleep(3) # sleep 3 seconds 
    sys.stdout = utils.ExperimentLogger(log_name, sys.stdout)
    print(model)
    sys.stdout.flush()

    # 初始化 best
    for valid_dataloader in valid_dataloaders:
        name = valid_dataloader['name']
        stat_dict[name]['best_psnr']['value'], stat_dict[name]['best_ssim']['value'], stat_dict[name]['best_lpips']['value'] = 0.0, 0.0, 1.0

    ## start training
    timer_start = time.time()
    avg_pgrd = 0.0
    cur_steps = 0.0
    for epoch in range(start_epoch, args.epochs+1):
        epoch_loss = 0.0
        epoch_loss_l1 = 0.0
        epoch_loss_lpips = 0.0
        stat_dict['epochs'] = epoch
        m_expert = m_expert.train()
        opt_lr = scheduler.get_last_lr()
        print('##==========={}-training, Epoch: {}, lr: {} =============##'.format('int8' if args.is_qat else 'fp32', epoch, opt_lr))
        weights = []
        
        for iter, batch in enumerate(train_dataloader):
            for p in model_d.parameters():
                p.requires_grad = False
            if args.is_LDL:
                for p in m_expert_ema.parameters():
                    p.requires_grad = False

            optimizer.zero_grad()
            lr, hr = batch
            if args.range == 1:
                lr, hr = lr / 255., hr / 255.
            lr, hr = lr.to(device), hr.to(device)
            # _, weight = model(lr)
            # weight = weight/torch.sum(weight, 1).unsqueeze(1).repeat([1,5])
            sr, _ = m_expert(lr, weights)
            if args.is_LDL:
                sr_ema, _ = m_expert_ema(lr, weights)
            # sr = m_expert(lr)

            loss_l1 = loss_func(sr, hr)

            # perceptual loss
            loss_p = torch.zeros(1).to(device)
            loss_p += args.is_l1loss * loss_l1
            if args.is_lpips:
            # convert to [-1,1]
                if args.range == 1:
                    sr_con, hr_con = sr*2-1, hr*2-1
                else:
                    sr_con, hr_con = (sr/255.)*2-1, (hr/255.)*2-1
                loss_p += args.is_lpips * (torch.mean(loss_lpips_func(sr_con, hr_con)))
            if args.is_perceptual:
                per, _ = perceptual_loss(sr, hr)
                loss_p += args.is_perceptual * per
            if args.is_ssim:
                if args.range == 1:
                    loss_p += args.is_ssim * (1 - ssim_loss(sr*255., hr*255.))
                else:
                    loss_p += args.is_ssim * (1 - ssim_loss(sr, hr))
            if args.is_dists:
            # convert to [-1,1]
                if args.range == 1:
                    sr_con, hr_con = sr, hr
                else:
                    sr_con, hr_con = sr/255., hr/255.
                loss_p += args.is_dists * DISTS_func(hr_con, sr_con, require_grad=True, batch_average=True)
            if args.is_gan:
                real_d_pred =  model_d(hr).detach()
                fake_g_pred = model_d(sr)
                l_g_real = gan_loss(real_d_pred - torch.mean(fake_g_pred), False, is_disc=False)
                l_g_fake = gan_loss(fake_g_pred - torch.mean(real_d_pred), True, is_disc=False)
                l_g_gan = (l_g_real + l_g_fake) / 2
                loss_p += args.is_gan * l_g_gan
            
            if args.is_LDL:
                pixel_weight = get_refined_artifact_map(hr, sr, sr_ema, 7)
                l_g_artifacts = loss_func(torch.mul(pixel_weight, sr), torch.mul(pixel_weight, hr))
                loss_p += l_g_artifacts

            loss_p.backward()
            optimizer.step()
            epoch_loss += float(loss_p)

            if args.is_LDL:
                # update model ema
                model_ema(m_expert, m_expert_ema, args.ema_decay)

            # update model_d
            if args.is_gan:
                for p in model_d.parameters():
                    p.requires_grad = True
                optimizer_d.zero_grad()
                # real
                fake_d_pred = model_d(sr).detach()
                real_d_pred = model_d(hr)
                l_d_real = gan_loss(real_d_pred - torch.mean(fake_d_pred), True, is_disc=True) * 0.5
                l_d_real.backward()
                # fake
                fake_d_pred = model_d(sr.detach())
                l_d_fake = gan_loss(fake_d_pred - torch.mean(real_d_pred.detach()), False, is_disc=True) * 0.5
                l_d_fake.backward()
                optimizer_d.step()

                del _, loss_l1, loss_p, lr, hr, sr, l_d_fake, l_d_real, fake_d_pred, real_d_pred
                torch.cuda.empty_cache()
            else:
                del _, loss_l1, loss_p, lr, hr, sr
                torch.cuda.empty_cache()


            if (iter + 1) % args.log_every == 0:
                cur_steps = (iter+1)*args.batch_size
                total_steps = len(train_dataloader.dataset)
                fill_width = math.ceil(math.log10(total_steps))
                cur_steps = str(cur_steps).zfill(fill_width)

                epoch_width = math.ceil(math.log10(args.epochs))
                cur_epoch = str(epoch).zfill(epoch_width)

                avg_loss = epoch_loss / (iter + 1)
                stat_dict['losses'].append(avg_loss)

                timer_end = time.time()
                duration = timer_end - timer_start
                timer_start = timer_end
                print('Epoch:{}, {}/{}, loss: {:.4f}, time: {:.3f}'.format(cur_epoch, cur_steps, total_steps, avg_loss, duration))

        if epoch % args.test_every == 0:
            # best practice for qat
            if epoch > 2 and args.is_qat:
                m_expert.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
            if epoch > 3 and args.is_qat:
                m_expert.apply(torch.quantization.disable_observer)

            torch.set_grad_enabled(False)
            test_log = ''
            m_expert = m_expert.eval()
            for valid_dataloader in valid_dataloaders:
                avg_psnr, avg_ssim, avg_lpips = 0.0, 0.0, 0.0
                name = valid_dataloader['name']
                loader = valid_dataloader['dataloader']
                weights = []
                for lr, hr in tqdm(loader, ncols=80):
                    if args.range == 1:
                        lr = lr / 255.
                    lr, hr = lr.to(device), hr.to(device)
                    # crop image for evaluation
                    hr = hr.clamp(0, 255)
                    hr = hr[:, :, args.scale:-args.scale, args.scale:-args.scale]
                    sr, _ = m_expert(lr, weights)
                    if args.range == 1:
                        sr = sr * 255.
                    # quantize output to [0, 255]
                    sr = sr.clamp(0, 255)
                    sr = sr[:, :, args.scale:-args.scale, args.scale:-args.scale]
                    
                    # conver to ycbcr
                    if args.colors == 3:
                        hr_ycbcr = utils.rgb_to_ycbcr(hr)
                        sr_ycbcr = utils.rgb_to_ycbcr(sr)
                        hr_ycbcr = hr_ycbcr[:, 0:1, :, :]
                        sr_ycbcr = sr_ycbcr[:, 0:1, :, :]
                    
                    # calculate psnr and ssim
                    psnr = utils.calc_psnr(sr_ycbcr, hr_ycbcr)       
                    ssim = utils.calc_ssim(sr_ycbcr, hr_ycbcr) 
                    lpips_m = torch.mean(loss_lpips_func((sr/255.)*2-1, (hr/255.)*2-1)).item()
                    pieapp_m = 0
                    avg_psnr += psnr
                    avg_ssim += ssim
                    avg_lpips += lpips_m

                    del sr, lr, hr, hr_ycbcr, sr_ycbcr, psnr, ssim, lpips_m
                    torch.cuda.empty_cache()
                avg_psnr = round(avg_psnr/len(loader) + 5e-3, 2)
                avg_ssim = round(avg_ssim/len(loader) + 5e-5, 4)
                avg_lpips = round(avg_lpips/len(loader) + 5e-5, 4)
                stat_dict[name]['psnrs'] = avg_psnr
                stat_dict[name]['ssims'] = avg_ssim
                stat_dict[name]['lpips'] = avg_lpips
                if stat_dict[name]['best_psnr']['value'] < avg_psnr:
                    stat_dict[name]['best_psnr']['value'] = avg_psnr
                    stat_dict[name]['best_psnr']['epoch'] = epoch
                if stat_dict[name]['best_ssim']['value'] < avg_ssim:
                    stat_dict[name]['best_ssim']['value'] = avg_ssim
                    stat_dict[name]['best_ssim']['epoch'] = epoch
                if stat_dict[name]['best_lpips']['value'] > avg_lpips:
                    stat_dict[name]['best_lpips']['value'] = avg_lpips
                    stat_dict[name]['best_lpips']['epoch'] = epoch
                test_log += '[{}-X{}], PSNR/SSIM/LPIPS: {:.2f}/{:.4f}/{:.4f} (Best: {:.2f}/{:.4f}/{:.4f}, Epoch: {}/{}/{})\n'.format(
                    name, args.scale, float(avg_psnr), float(avg_ssim), float(avg_lpips),
                    stat_dict[name]['best_psnr']['value'], stat_dict[name]['best_ssim']['value'], stat_dict[name]['best_lpips']['value'],
                    stat_dict[name]['best_psnr']['epoch'], stat_dict[name]['best_ssim']['epoch'], stat_dict[name]['best_lpips']['epoch'])
            # print log & flush out
            print(test_log)
            sys.stdout.flush()
            # save model
            saved_model_path = os.path.join(experiment_model_path, 'model_x{}_{}.pt'.format(args.scale, epoch))
            # torch.save(model.state_dict(), saved_model_path)
            torch.save({
                'epoch': epoch,
                'model_state_dict': m_expert.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'stat_dict': stat_dict
            }, saved_model_path)
            torch.set_grad_enabled(True)
            # save stat dict
            ## save training paramters
            stat_dict_name = os.path.join(experiment_path, 'stat_dict.yml')
            with open(stat_dict_name, 'w') as stat_dict_file:
                yaml.dump(stat_dict, stat_dict_file, default_flow_style=False)
        ## update scheduler
        scheduler.step()
        scheduler_d.step()