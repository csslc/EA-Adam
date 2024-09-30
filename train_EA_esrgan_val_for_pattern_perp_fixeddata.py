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
from EA_operator.gen import gen_cross, gen_mut
from optimizer.norm_grd import normalized_gradient
from collections import OrderedDict
import copy
from utils import dis_result
import numpy as np
from losses.losses import PerceptualLoss, SSIMLoss, GANLoss

parser = argparse.ArgumentParser(description='EasySR')
## yaml configuration files
parser.add_argument('--config', type=str, default='configs/rrdb_gan.yml', help = 'pre-config file for training')
parser.add_argument('--resume', type=str, default='/BasicSR_model/ESRGAN_PSNR_SRx4_DF2K_official-150ff491.pth', help = 'resume training or not')
parser.add_argument('--resume_d', type=str, default=None, help = 'resume training or not')
parser.add_argument('--is_l1loss', type=float, default= 0.01, help = 'resume training or not')
parser.add_argument('--is_lpips', type=float, default=False, help = 'resume training or not')
parser.add_argument('--is_perceptual', type=float, default=1.0, help = 'resume training or not')
parser.add_argument('--is_ssim', type=float, default=False, help = 'resume training or not')
parser.add_argument('--is_gan', type=float, default=0.005, help = 'resume training or not')

# val methods
parser.add_argument('--is_pretrain', type=str, default=True, help = 'if the pretrain model is used')
parser.add_argument('--is_all_gen_model', type=str, default=True, help = 'if the popsize generator models are optimized')
parser.add_argument('--is_dir_gen', type=str, default=True, help = 'if the generator models are directly added in EA optimizer')
parser.add_argument('--is_all_dis_model', type=str, default= True, help = 'if the popsize discriminator models are optimized')

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

    ## create dataset for training and testing
    train_dataloader, train_for_EA_dataloader, test_dataloaders, _ = create_datasets(args)

    ## definition of loss functions
    loss_func = nn.L1Loss()
    loss_lpips_func = lpips.LPIPS(net='alex', spatial = False).to(device)
    perceptual_loss = PerceptualLoss(args.perceptual_opt['layer_weights']).to(device)
    ssim_loss = SSIMLoss().to(device)
    gan_loss = GANLoss(args.gan_opt.get('gan_type')).to(device)

    ## definitions of model and the corresponding optimizer
    models = []
    optimizers = []
    schedulers = []
    loss_value = np.zeros([2, int(args.popsize)])
    for i in range(args.popsize):
        try:
            model = utils.import_module('models.{}.{}_network'.format(args.model, args.model)).create_model(args)
        except Exception:
            raise ValueError('not supported model type! or something')
        # model = nn.DataParallel(model).to(device)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        if args.is_qat:
            scheduler = StepLR(optimizer, step_size=args.decays, gamma=args.gamma)
        else:
            scheduler = MultiStepLR(optimizer, milestones=args.decays, gamma=args.gamma)
        models.append(model)
        optimizers.append(optimizer)
        schedulers.append(scheduler)

    # definitions of discriminator model
    if args.is_all_dis_model:
        models_d = []
        optimizers_d = []
        schedulers_d = []
        for i in range(args.popsize):
            model_d = utils.import_module('models.discriminator.{}_network'.format(args.network_d.get('type'))).create_model(args)
            model_d = nn.DataParallel(model_d).to(device)
            optimizer_d = torch.optim.Adam(model_d.parameters(), lr=args.lr)
            if args.is_qat:
                scheduler_d = StepLR(optimizer_d, step_size=args.decays, gamma=args.gamma)
            else:
                scheduler_d = MultiStepLR(optimizer_d, milestones=args.decays, gamma=args.gamma)
            models_d.append(model_d)
            optimizers_d.append(optimizer_d)
            schedulers_d.append(scheduler_d)
    else:
        model_d = utils.import_module('models.discriminator.{}_network'.format(args.network_d.get('type'))).create_model(args)
        model_d = nn.DataParallel(model_d).to(device)
        optimizer_d = torch.optim.Adam(model_d.parameters(), lr=args.lr)
        if args.is_qat:
            scheduler_d = StepLR(optimizer_d, step_size=args.decays, gamma=args.gamma)
        else:
            scheduler_d = MultiStepLR(optimizer_d, milestones=args.decays, gamma=args.gamma)

    ## definition of weight for objective function, calculate the neighboring index of each ind
    from EA_operator.uniform import uniform_point
    w = uniform_point(args.popsize, start_point=[1,0], end_point=[0,1])
    # w = torch.tensor([[0],[1]])
    B = np.zeros([args.neighborsize * 2 + 1, args.popsize])
    B[0,:] = np.array(range(args.popsize))
    B[1,:] = [max(x,0) for x in B[0,:]-args.neighborsize]
    B[2,:] = [min(x,args.popsize-1) for x in B[0,:]+args.neighborsize]
    B[2,0] = B[2,0] + 1
    B[1,args.popsize-1] = B[1,args.popsize-1] - 1


    ## load pretrain
    if args.pretrain is not None:
        print('load pretrained model: {}!'.format(args.pretrain))
        ckpt = torch.load(args.pretrain)
        model.load_state_dict(ckpt['model_state_dict'])
        ## if qat
    if args.is_qat:
        if args.pretrain is not None:
            print('start quantization-awared training !')
            model = utils.prepare_qat(model)
        else:
            raise ValueError('please provide pre-trained model for qat!')
    
    ## resume training
    start_epoch = 0
    if (args.resume is not None) and (args.is_pretrain):
        ckpt = torch.load(args.resume)
        pt2, pt3 = [], []
        for i in range(args.popsize):
            # models[i].load_state_dict(ckcp)
            models[i].load_state_dict(ckpt['params'])
        stat_dict = utils.get_stat_dict(args)
        timestamp = utils.cur_timestamp_str()
        if args.log_name is None:
            experiment_name = '{}-{}-x{}-{}'.format(args.model, 'int8' if args.is_qat else 'fp32', args.scale, timestamp)
        else:
            experiment_name = '{}-{}'.format(args.log_name, timestamp)
        experiment_path = os.path.join(args.log_path, experiment_name)
        log_name = os.path.join(experiment_path, 'log.txt')
        experiment_model_path = os.path.join(experiment_path, 'models')
        ## create folder for ckpt and stat
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)
        experiment_model_path = os.path.join(experiment_path, 'models')
        if not os.path.exists(experiment_model_path):
            os.makedirs(experiment_model_path)
        print('select {}, resume training from epoch {}.'.format(args.resume, start_epoch))
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

    ## print architecture of model
    time.sleep(3) # sleep 3 seconds 
    sys.stdout = utils.ExperimentLogger(log_name, sys.stdout)
    print(model)
    sys.stdout.flush()

    # 初始化 best
    for test_dataloader in test_dataloaders:
        name = test_dataloader['name']
        for i in range(args.popsize):
            stat_dict[name]['best_psnr']['value'][str(i)], \
            stat_dict[name]['best_ssim']['value'][str(i)], \
            stat_dict[name]['best_lpips']['value'][str(i)] = 0.0, 0.0, 1000000.0

    ## start training
    timer_start = time.time()
    is_EA = False
    is_Adam = True
    is_upd_loss_value = False
    type_opt = 'Adam'
    epoch_ea, epoch_adam = 0, 0
    z_min = 10 * torch.ones(2)
    z_max = 0.0001 * torch.ones(2)

    for epoch in range(start_epoch, args.epochs+1):
        stat_dict['epochs'] = epoch
        opt_lr = schedulers[0].get_last_lr()
        epoch_losses = []
        if is_Adam:
            epoch_adam += 1
            if epoch_adam>=args.adam_epoch:
                is_EA = True
                is_upd_loss_value = True
                is_Adam = False
                epoch_adam = 0
                type_opt = 'EA'
                # define the other models except 0 and popsize
                if not args.is_all_gen_model:
                    if args.is_dir_gen:
                        if args.is_all_dis_model:
                            for j in range(args.popsize):
                                if (j == 0) or (j==args.popsize-1):
                                    continue
                                if j < args.popsize//2:
                                    models[j] = copy.deepcopy(models[0])
                                    models_d[j] = copy.deepcopy(models_d[0])
                                else:
                                    models[j] = copy.deepcopy(models[args.popsize-1])
                                    models_d[j] = copy.deepcopy(models_d[args.popsize-1])
                                optimizers_d[j] = torch.optim.Adam(models_d[j].parameters(), lr=args.lr)
                                if args.is_qat:
                                    schedulers_d[j] = StepLR(optimizers_d[j], step_size=args.decays, gamma=args.gamma)
                                else:
                                    schedulers_d[j] = MultiStepLR(optimizers_d[j], milestones=args.decays, gamma=args.gamma)
                                for num_d in range(epoch):
                                    schedulers_d[j].step()
                        else:
                            for j in range(args.popsize):
                                if (j == 0) or (j==args.popsize-1):
                                    continue
                                if j < args.popsize//2:
                                    models[j] = copy.deepcopy(models[0])
                                else:
                                    models[j] = copy.deepcopy(models[args.popsize-1])
                    else:
                        if not args.is_all_dis_model:
                            model_0= copy.deepcopy(models[0])
                            model_0_par = dict(model_0.named_parameters())
                            for j in range(args.popsize):
                                if (j == 0) or (j==args.popsize-1):
                                    continue
                                model_final = copy.deepcopy(models[args.popsize-1])
                                model_final_par = dict(model_final.named_parameters())
                                for k in model_0_par.keys():
                                    model_final_par[k].data.mul_(w[1][j]).add_(model_0_par[k].data, alpha=w[0][j])
                                models[j] = copy.deepcopy(model_final)
                        else:
                            print('the setting is wrong!')

        elif is_EA:
            epoch_ea += 1
            if epoch_ea>=args.EA_epoch:
                is_Adam = True
                is_EA = False
                type_opt = 'Adam'
                epoch_ea = 0
                # redefine the optimizer and scheduler
                optimizers = []
                schedulers = []
                for i in range(args.popsize):
                    optimizer = torch.optim.Adam(models[i].parameters(), lr=args.lr)
                    if args.is_qat:
                        scheduler = StepLR(optimizer, step_size=args.decays, gamma=args.gamma)
                    else:
                        scheduler = MultiStepLR(optimizer, milestones=args.decays, gamma=args.gamma)
                    for j in range(epoch):
                        scheduler.step()
                    optimizers.append(optimizer)
                    schedulers.append(scheduler)
        
        for i in range(args.popsize):
            models[i] = models[i].train()
            epoch_losses.append(0.0)
        if args.is_all_dis_model:
            for i in range(args.popsize):
                models_d[i] = models_d[i].train()
        else:     
            model_d = model_d.train()

        print('##=========== {}-training, Epoch: {}, lr: {}, optimizer: {} =============##'.format('int8' if args.is_qat else 'fp32', epoch, opt_lr[0], type_opt))
        if is_EA:
            if is_upd_loss_value:
                loss_value = np.zeros([2, int(args.popsize)])
                loss_all_value = np.zeros([2, int(args.popsize)])
                for iter, batch in enumerate(train_for_EA_dataloader):
                    lr, hr = batch
                    lr, hr = lr.to(device), hr.to(device)
                    lr, hr = lr.squeeze(0), hr.squeeze(0)
                    if args.range == 1:
                        lr, hr = lr / 255., hr / 255.
                    for i in range(args.popsize):
                        sr = models[i](lr)
                        l1_t = loss_func(sr, hr).item()
                        if args.is_perceptual:
                            perp, _ = perceptual_loss(sr, hr)
                            l_per_t = perp.item()
                            # del perp
                            # torch.cuda.empty_cache()

                        # del sr
                        # torch.cuda.empty_cache()
                        loss_all_value[0][i] += l1_t
                        loss_all_value[1][i] += l_per_t
                    # del lr, hr
                    # torch.cuda.empty_cache()

                loss_value = loss_all_value/len(train_for_EA_dataloader)

                z_min[0] = min(loss_value[0])
                z_min[1] = min(loss_value[1])
                z_max[0] = max(loss_value[0])
                z_max[1] = max(loss_value[1])
                is_upd_loss_value = False

            for EA_iter in range(args.EA_iter_per_epoch):
                # initialize the loss of population
                for i in range(args.popsize):
                    if np.random.rand(1)<=args.pCrossover:
                        pt1 = copy.deepcopy(models[i])
                        if np.random.rand(1) <= 0.3:
                            lp = list(range(args.popsize))
                            del lp[i]
                            pt = np.random.choice(args.popsize-1, 2, replace=False)
                            pt2 = copy.deepcopy(models[lp[pt[0]]])
                            pt3 = copy.deepcopy(models[lp[pt[1]]])
                        else:
                            lp = list(range(int(B[1][i]),int(B[2][i])+1))
                            lp.remove(i)
                            pt = np.random.choice(args.neighborsize * 2, 2, replace=False)
                            pt2 = copy.deepcopy(models[lp[pt[0]]])
                            pt3 = copy.deepcopy(models[lp[pt[1]]])
                        indc_model1, indc_model2 = gen_cross(models[i], pt1, pt2, pt3, args.cro, args.mu, args.model)
                    else:
                        indc_model1 = copy.deepcopy(models[i])
                    if np.random.rand(1)<=args.pMutation:
                        pt2 = []
                        pt3 = []
                        indc_model1 = gen_mut(models[i], indc_model1, pt2, pt3, args.mut, args.mu, args.model, only_conv = True)
                    
                    # evaluate the newly-generated individual
                    l1_all_ind, l_per_all_ind = 0.0, 0.0                 
                    for iter, batch in enumerate(train_for_EA_dataloader):
                        lr, hr = batch
                        lr, hr = lr.to(device), hr.to(device)
                        lr, hr = lr.squeeze(0), hr.squeeze(0)
                        
                        if args.range == 1:
                            lr, hr = lr / 255., hr / 255.

                        sr_m_ind = indc_model1(lr)

                        # del lr
                        # torch.cuda.empty_cache()

                        l1_ind = loss_func(sr_m_ind, hr).item()
                        perp, _ = perceptual_loss(sr_m_ind, hr)
                        l_per_ind = perp.item()
                        # del perp
                        # torch.cuda.empty_cache()

                        # real_d_pred =  models_d[i](hr).detach()
                        # fake_g_pred = models_d[i](sr_m_ind)
                        # l_g_real = gan_loss(real_d_pred - torch.mean(fake_g_pred), False, is_disc=False)
                        # l_g_fake = gan_loss(fake_g_pred - torch.mean(real_d_pred), True, is_disc=False)
                        # l_g_gan = (l_g_real.item() + l_g_fake.item()) / 2
                        
                        l1_all_ind += l1_ind
                        l_per_all_ind += l_per_ind
                        # l_per_all_ind += args.is_gan * l_g_gan

                        # del sr_m_ind, hr
                        # torch.cuda.empty_cache()

                    l1_ind = l1_all_ind / len(train_for_EA_dataloader)
                    per_ind = l_per_all_ind / len(train_for_EA_dataloader)

                    if l1_ind < z_min[0]:
                        z_min[0] = l1_ind
                    if per_ind < z_min[1]:
                        z_min[1] = per_ind
                    
                    if l1_ind > z_max[0]:
                        z_max[0] = l1_ind
                    if per_ind > z_max[1]:
                        z_max[1] = per_ind
                    
                    # update the generator population                   
                    for ind in range(int(B[1][i]),int(B[2][i])+1):
                        loss_now = max(abs(l1_ind-z_min[0])* w[0][ind].numpy()/((z_max[0]-z_min[0])+0.00001), abs(per_ind-z_min[1])* w[1][ind].numpy()/((z_max[1]-z_min[1])+0.00001))
                        loss_ori_l1 = loss_value[0][ind]
                        loss_ori_per = loss_value[1][ind]
                        loss_ori = max(abs(loss_ori_l1-z_min[0])* w[0][ind].numpy()/((z_max[0]-z_min[0])+0.00001), abs(loss_ori_per-z_min[1])* w[1][ind].numpy()/((z_max[1]-z_min[1])+0.00001))
                        if loss_now < loss_ori:
                            models[ind] = copy.deepcopy(indc_model1)
                            loss_value[0][ind] = l1_ind
                            loss_value[1][ind] = per_ind
                            epoch_losses[ind] = loss_now
                        else:
                            epoch_losses[ind] = loss_ori
                        
                    # del indc_model1
                    # torch.cuda.empty_cache()
                               
                if (EA_iter + 1) % (args.EA_log_every) == 0:
                    cur_steps = (EA_iter+1)*args.batch_size_EA
                    total_steps = args.EA_iter_per_epoch
                    fill_width = math.ceil(math.log10(total_steps))
                    cur_steps = str(cur_steps).zfill(fill_width)

                    epoch_width = math.ceil(math.log10(args.epochs))
                    cur_epoch = str(epoch).zfill(epoch_width)

                    avg_losses = epoch_losses

                    stat_dict['losses'].append(avg_losses)

                    timer_end = time.time()
                    duration = timer_end - timer_start
                    timer_start = timer_end
                    print('Epoch:{}, {}/{}, time: {:.3f},\nloss: {}'.format(cur_epoch, cur_steps, total_steps, duration, avg_losses))

        elif is_Adam:
            for iter, batch in enumerate(train_dataloader):
                lr, hr = batch
                if args.range == 1:
                    lr, hr = lr / 255., hr / 255.
                lr, hr = lr.to(device), hr.to(device)

                for i in range(args.popsize):
                    if not args.is_all_gen_model:
                        if (i != 0) and (i != args.popsize-1):
                            continue
                    if (args.is_pretrain) and (i==0):
                        continue

                    if args.is_all_dis_model:
                        for p in models_d[i].parameters():
                            p.requires_grad = False
                    else:
                        for p in model_d.parameters():
                            p.requires_grad = False
                    optimizers[i].zero_grad()
                    sr = models[i](lr) 
                    loss_l1 = loss_func(sr, hr)

                    # perceptual loss
                    loss_p = torch.zeros(1).to(device)
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
                    if args.is_gan:
                        if args.is_all_dis_model:
                            real_d_pred =  models_d[i](hr).detach()
                            fake_g_pred = models_d[i](sr)
                            l_g_real = gan_loss(real_d_pred - torch.mean(fake_g_pred), False, is_disc=False)
                            l_g_fake = gan_loss(fake_g_pred - torch.mean(real_d_pred), True, is_disc=False)
                            l_g_gan = (l_g_real + l_g_fake) / 2
                            loss_p += args.is_gan * l_g_gan
                        else:
                            real_d_pred =  model_d(hr).detach()
                            fake_g_pred =model_d(sr)
                            l_g_real = gan_loss(real_d_pred - torch.mean(fake_g_pred), False, is_disc=False)
                            l_g_fake = gan_loss(fake_g_pred - torch.mean(real_d_pred), True, is_disc=False)
                            l_g_gan = (l_g_real + l_g_fake) / 2
                            loss_p += args.is_gan * l_g_gan
                        
                    loss_l1.backward(retain_graph=True)
                    loss = w[0][i]*loss_l1.item() + w[1][i]*loss_p.item()
                    g1_norm = OrderedDict()
                    for name, par in models[i].named_parameters():
                        if par.grad is not None:
                            g1_norm[name] = copy.deepcopy(normalized_gradient(par.grad,use_norm=True, norm_conv_only=False))
                    optimizers[i].zero_grad()
                    loss_p.backward(retain_graph=True)
                    g2_norm = OrderedDict()
                    for name, par in models[i].named_parameters():
                        if par.grad is not None:
                            g2_norm[name] = copy.deepcopy(normalized_gradient(par.grad,use_norm=True, norm_conv_only=False))
                            g = w[0][i]*g1_norm[name]+w[1][i]*g2_norm[name]
                            par.grad = g

                    optimizers[i].step()
                    epoch_losses[i] += float(loss)

                    # del loss_p, loss_l1
                    # torch.cuda.empty_cache()

                    # optimize model_d
                    if args.is_all_dis_model:
                        # optimize model_d
                        for p in models_d[i].parameters():
                            p.requires_grad = True
                        optimizers_d[i].zero_grad()
                        # real
                        fake_d_pred = models_d[i](sr).detach()
                        real_d_pred = models_d[i](hr)
                        l_d_real = gan_loss(real_d_pred - torch.mean(fake_d_pred), True, is_disc=True) * 0.5
                        l_d_real.backward()
                        # fake
                        fake_d_pred = models_d[i](sr.detach())
                        l_d_fake = gan_loss(fake_d_pred - torch.mean(real_d_pred.detach()), False, is_disc=True) * 0.5
                        l_d_fake.backward()
                        optimizers_d[i].step()

                        # del real_d_pred, fake_d_pred, l_d_real, l_d_fake, sr
                        # torch.cuda.empty_cache()
                       
                    else:
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
                        
                        # del real_d_pred, fake_d_pred, l_d_real, l_d_fake
                        # torch.cuda.empty_cache()
                        
                if (iter + 1) % args.log_every == 0:
                    cur_steps = (iter+1)*args.batch_size
                    total_steps = len(train_dataloader.dataset)
                    fill_width = math.ceil(math.log10(total_steps))
                    cur_steps = str(cur_steps).zfill(fill_width)

                    epoch_width = math.ceil(math.log10(args.epochs))
                    cur_epoch = str(epoch).zfill(epoch_width)

                    # avg_loss = epoch_loss / (iter + 1)
                    avg_losses = [round(x/(iter + 1), 4) for x in epoch_losses]

                    stat_dict['losses'].append(avg_losses)

                    timer_end = time.time()
                    duration = timer_end - timer_start
                    timer_start = timer_end
                    print('Epoch:{}, {}/{}, time: {:.3f},\nloss: {}'.format(cur_epoch, cur_steps, total_steps, duration, avg_losses))

        if (epoch % args.test_every==0) or (is_EA):
            # best practice for qat
            if epoch > 2 and args.is_qat:
                model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
            if epoch > 3 and args.is_qat:
                model.apply(torch.quantization.disable_observer)

            torch.set_grad_enabled(False)
            test_log = ''
            psnr, ssim, lpips = OrderedDict(), OrderedDict(), OrderedDict()
            for i in range(args.popsize):
                models[i] = models[i].eval()
                psnr[str(i)], ssim[str(i)],lpips[str(i)] = OrderedDict(),OrderedDict(),OrderedDict()
            if not args.is_all_dis_model:
                model_d = model_d.eval()
            else:
                for i in range(args.popsize):
                    models_d[i] = models_d[i].eval()

            patch_size = args.patch_size
            for test_dataloader in test_dataloaders:
                name = test_dataloader['name']
                for i in range(args.popsize):
                    psnr[str(i)][name],ssim[str(i)][name], lpips[str(i)][name] = 0.0, 0.0, 0.0

                loader = test_dataloader['dataloader']
                for lr, hr in tqdm(loader, ncols=80):
                    if args.range == 1:
                        lr = lr / 255.

                    # quantize output to [0, 255]
                    hr = hr.clamp(0, 255)
                    # crop image for evaluation
                    hr = hr[:, :, args.scale:-args.scale, args.scale:-args.scale]

                    lr, hr = lr.to(device), hr.to(device)
                    
                    for i in range(args.popsize):
                        sr = models[i](lr)
                        if args.range == 1:
                            sr = sr * 255.
                        # quantize output to [0, 255]
                        sr = sr.clamp(0, 255)
                        # crop image for evaluation
                        sr = sr[:, :, args.scale:-args.scale, args.scale:-args.scale]
                        
                        # calculate the metrics
                        lpips_m = torch.mean(loss_lpips_func((sr/255.)*2-1, (hr/255.)*2-1)).item()

                        if (args.colors == 3):
                            sr_ycbcr = utils.rgb_to_ycbcr(sr)
                            hr_ycbcr = utils.rgb_to_ycbcr(hr)
                            hr_ycbcr = hr_ycbcr[:, 0:1, :, :]
                            sr_ycbcr = sr_ycbcr[:, 0:1, :, :]
                        psnr_m = utils.calc_psnr(sr_ycbcr, hr_ycbcr)
                        ssim_m = ssim_loss(sr_ycbcr, hr_ycbcr).item()

                        psnr[str(i)][name] += psnr_m
                        ssim[str(i)][name] += ssim_m
                        lpips[str(i)][name] +=lpips_m

                for i in range(args.popsize):
                    avg_psnr = round(psnr[str(i)][name] /len(loader) + 5e-3, 2)
                    avg_ssim = round(ssim[str(i)][name]/len(loader) + 5e-5, 4)
                    avg_lpips = round(lpips[str(i)][name]/len(loader) + 5e-5, 4)

                    stat_dict[name]['psnrs'][str(i)] = avg_psnr
                    stat_dict[name]['ssims'][str(i)] = avg_ssim
                    stat_dict[name]['lpipss'][str(i)] = avg_lpips
                    if stat_dict[name]['best_psnr']['value'][str(i)] < avg_psnr:
                        stat_dict[name]['best_psnr']['value'][str(i)] = avg_psnr
                        stat_dict[name]['best_psnr']['epoch'][str(i)] = epoch
                    if stat_dict[name]['best_ssim']['value'][str(i)] < avg_ssim:
                        stat_dict[name]['best_ssim']['value'][str(i)] = avg_ssim
                        stat_dict[name]['best_ssim']['epoch'][str(i)] = epoch
                    if stat_dict[name]['best_lpips']['value'][str(i)] > avg_lpips:
                        stat_dict[name]['best_lpips']['value'][str(i)] = avg_lpips
                        stat_dict[name]['best_lpips']['epoch'][str(i)] = epoch
                    test_log += 'the current ind is: {}, \n[{}-X{}], PSNR/SSIM/LPIPS: {:.2f}/{:.4f}/{:.4f} (Best: {:.2f}/{:.4f}/{:.4f}, Epoch: {}/{}/{})\n'.format(i,
                        name, args.scale, float(avg_psnr), float(avg_ssim), float(avg_lpips), 
                        stat_dict[name]['best_psnr']['value'][str(i)], stat_dict[name]['best_ssim']['value'][str(i)], stat_dict[name]['best_lpips']['value'][str(i)],
                        stat_dict[name]['best_psnr']['epoch'][str(i)], stat_dict[name]['best_ssim']['epoch'][str(i)], stat_dict[name]['best_lpips']['epoch'][str(i)])
                # display results
                dis_result(name, experiment_path, epoch, stat_dict)
                # save model
                for i in range(args.popsize):
                    saved_model_path = os.path.join(experiment_model_path, 'model_x{}_{}_{}.pt'.format(args.scale, epoch, i))
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': models[i].state_dict(),
                        'optimizer_state_dict': optimizers[i].state_dict(),
                        'scheduler_state_dict': schedulers[i].state_dict()
                    }, saved_model_path)
                    saved_model_path = os.path.join(experiment_model_path, 'model_d_x{}_{}_{}.pt'.format(args.scale, epoch, i))
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': models_d[i].state_dict(),
                        'optimizer_state_dict': optimizers_d[i].state_dict(),
                        'scheduler_state_dict': schedulers_d[i].state_dict()
                    }, saved_model_path)
                # print log & flush out
            print(test_log)
            sys.stdout.flush()
            torch.set_grad_enabled(True)
            # save stat dict
            ## save training paramters
            stat_dict_name = os.path.join(experiment_path, 'stat_dict.yml')
            with open(stat_dict_name, 'w') as stat_dict_file:
                yaml.dump(stat_dict, stat_dict_file, default_flow_style=False)
        ## update scheduler
        # scheduler.step()
        if args.is_all_dis_model:
            for i in range(args.popsize):
                schedulers[i].step()
                schedulers_d[i].step()
        else:
            for i in range(args.popsize):
                schedulers[i].step()
            scheduler_d.step()