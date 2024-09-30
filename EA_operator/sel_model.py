from tqdm import tqdm
import torch
import numpy as np
import copy
from torch.nn.parallel import DataParallel, DistributedDataParallel

def sel_model(device, perceptual_loss, loss_func, loss_lpips_func, net_ori, indc_model1, valid_dataloaders, val_name, args):

    loss_full_ori_l1 = 0.0
    loss_full_ori_lpips = 0.0
    loss_full_now_l1 = 0.0
    loss_full_now_lpips = 0.0
    for valid_dataloader in valid_dataloaders:
        name = valid_dataloader['name']
        if name == val_name:
            loader = valid_dataloader['dataloader']
            for lr, hr in loader:
                if args.range == 1:
                        lr,hr = lr / 255., hr / 255.
                lr, hr = lr.to(device), hr.to(device)
                sr_ori = net_ori(lr)
                sr_now = indc_model1(lr)

                if args.is_lpips:
                # convert to [-1,1]
                    lpips_ori = torch.mean(loss_lpips_func((sr_ori)*2-1, (hr)*2-1)).item()
                elif args.is_perceptual:
                    lpips_ori, _ = perceptual_loss(sr_ori, hr)
                l1_ori = loss_func(sr_ori, hr).item()
                if args.is_lpips:
                # convert to [-1,1]
                    lpips_now = torch.mean(loss_lpips_func((sr_now)*2-1, (hr)*2-1)).item()
                elif args.is_perceptual:
                    lpips_now, _ = perceptual_loss(sr_now, hr)

                l1_now = loss_func(sr_now, hr).item()
                loss_full_ori_l1 += l1_ori
                loss_full_ori_lpips += lpips_ori
                loss_full_now_l1 += l1_now
                loss_full_now_lpips +=lpips_now
            loss_ori_l1 =  loss_full_ori_l1 /len(loader)
            loss_ori_lpips = loss_full_ori_lpips /len(loader)
            loss_now_l1 = loss_full_now_l1 /len(loader)
            loss_now_lpips = loss_full_now_lpips /len(loader)

            
    return loss_ori_l1, loss_ori_lpips, loss_now_l1, loss_now_lpips

def evl_gan_model(device, gan_loss, model_ema_d, ssim_loss, perceptual_loss, loss_func, loss_lpips_func, net_ori, train_for_EA_dataloader, args):
    loss_full_ori_l1 = 0.0
    loss_full_ori_lpips = 0.0
    patch_size = args.patch_size
    l = 0
    sr_output = []
    hr_output = []
    for iter, batch in enumerate(train_for_EA_dataloader):
        lr, hr = batch
        lr, hr = lr.to(device), hr.to(device)
        # lr, hr = lr.reshape(lr.shape[0]*lr.shape[1], 3, patch_size//args.scale, patch_size//args.scale), hr.reshape(hr.shape[0]*hr.shape[1], 3, patch_size, patch_size)
        
    # for valid_dataloader in valid_dataloaders:
        # name = valid_dataloader['name']
        # if name == val_name:
        #     loader = valid_dataloader['dataloader']
        #     for lr, hr in loader:
        if args.range == 1:
            lr, hr = lr / 255., hr / 255.
        # 切割 patch
        # pad_hr = (patch_size - 1) // 2
        # residual_pad = F.pad(residual, pad=[pad, pad, pad, pad], mode='reflect')
        # unfolded_hrpatch = hr.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        # unfolded_hrpatch = unfolded_hrpatch.squeeze(0).reshape(3,unfolded_hrpatch.shape[2]*unfolded_hrpatch.shape[3],patch_size,patch_size).permute(1,0,2,3)

        # unfolded_hrpatch, lr = unfolded_hrpatch.to(device), lr.to(device)

        lr_1, hr_1 = lr[0:lr.shape[0]//2], hr[0:hr.shape[0]//2]
        lr_2, hr_2 = lr[lr.shape[0]//2:lr.shape[0]], hr[hr.shape[0]//2:hr.shape[0]]

        sr_m_ori_1, sr_m_ori_2 = net_ori(lr_1), net_ori(lr_2)
        # sr_m_ori = sr_m_ori.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        # sr_m_ori = sr_m_ori.squeeze(0).reshape(3,sr_m_ori.shape[2]*sr_m_ori.shape[3],patch_size,patch_size).permute(1,0,2,3)
        
        l1_ori = loss_func(sr_m_ori_1, hr_1).item() + loss_func(sr_m_ori_2, hr_2).item()

        lpips_ori = 0.0
        if args.is_lpips:
        # convert to [-1,1]
            lpips_ori += args.is_lpips * (torch.mean(loss_lpips_func((sr_m_ori)*2-1, (hr)*2-1)).item())
        if args.is_ssim:
            lpips_ori = args.is_ssim * (1 - ssim_loss(sr_m_ori*255., hr*255.).item())
        if args.is_perceptual:
            # perp, _ = perceptual_loss(sr_m_ori, hr)
            # lpips_ori += args.is_perceptual * (perp.item())
            perp, _ = perceptual_loss(sr_m_ori_1, hr_1)
            lpips_ori += args.is_perceptual * (perp.item())
            perp, _ = perceptual_loss(sr_m_ori_2, hr_2)
            lpips_ori += args.is_perceptual * (perp.item())
        if args.is_gan:
            # fake_g_pred = model_ema_d(sr_m_ori)
            # l_g_gan = gan_loss(fake_g_pred, True, is_disc=False)
            # lpips_ori += args.is_gan * (l_g_gan.item())
            fake_g_pred = model_ema_d(sr_m_ori_1)
            l_g_gan = gan_loss(fake_g_pred, True, is_disc=False)
            lpips_ori += args.is_gan * (l_g_gan.item())
            fake_g_pred = model_ema_d(sr_m_ori_2)
            l_g_gan = gan_loss(fake_g_pred, True, is_disc=False)
            lpips_ori += args.is_gan * (l_g_gan.item())
        lpips_ori += args.is_l1loss * (l1_ori)

        loss_full_ori_l1 += l1_ori
        loss_full_ori_lpips += lpips_ori

        sr_output.append(sr_m_ori_1)
        sr_output.append(sr_m_ori_2)
        hr_output.append(hr_1)
        hr_output.append(hr_2)

    loss_ori_l1 =  loss_full_ori_l1 / len(train_for_EA_dataloader) / 2
    loss_ori_lpips = loss_full_ori_lpips / len(train_for_EA_dataloader) / 2
        
    return loss_ori_l1, loss_ori_lpips, sr_output, hr_output

def evl_gan_model(device, gan_loss, model_ema_d, ssim_loss, perceptual_loss, loss_func, loss_lpips_func, net_ori, train_for_EA_dataloader, args):
    loss_full_ori_l1 = 0.0
    loss_full_ori_lpips = 0.0
    patch_size = args.patch_size
    l = 0
    sr_output = []
    hr_output = []
    for iter, batch in enumerate(train_for_EA_dataloader):
        lr, hr = batch
        lr, hr = lr.to(device), hr.to(device)
        # lr, hr = lr.reshape(lr.shape[0]*lr.shape[1], 3, patch_size//args.scale, patch_size//args.scale), hr.reshape(hr.shape[0]*hr.shape[1], 3, patch_size, patch_size)
        
    # for valid_dataloader in valid_dataloaders:
        # name = valid_dataloader['name']
        # if name == val_name:
        #     loader = valid_dataloader['dataloader']
        #     for lr, hr in loader:
        if args.range == 1:
            lr, hr = lr / 255., hr / 255.
        # 切割 patch
        # pad_hr = (patch_size - 1) // 2
        # residual_pad = F.pad(residual, pad=[pad, pad, pad, pad], mode='reflect')
        # unfolded_hrpatch = hr.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        # unfolded_hrpatch = unfolded_hrpatch.squeeze(0).reshape(3,unfolded_hrpatch.shape[2]*unfolded_hrpatch.shape[3],patch_size,patch_size).permute(1,0,2,3)

        # unfolded_hrpatch, lr = unfolded_hrpatch.to(device), lr.to(device)

        lr_1, hr_1 = lr[0:lr.shape[0]//2], hr[0:hr.shape[0]//2]
        lr_2, hr_2 = lr[lr.shape[0]//2:lr.shape[0]], hr[hr.shape[0]//2:hr.shape[0]]

        sr_m_ori_1, sr_m_ori_2 = net_ori(lr_1), net_ori(lr_2)
        # sr_m_ori = sr_m_ori.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        # sr_m_ori = sr_m_ori.squeeze(0).reshape(3,sr_m_ori.shape[2]*sr_m_ori.shape[3],patch_size,patch_size).permute(1,0,2,3)
        
        l1_ori = loss_func(sr_m_ori_1, hr_1).item() + loss_func(sr_m_ori_2, hr_2).item()

        lpips_ori = 0.0
        if args.is_lpips:
        # convert to [-1,1]
            lpips_ori += args.is_lpips * (torch.mean(loss_lpips_func((sr_m_ori)*2-1, (hr)*2-1)).item())
        if args.is_ssim:
            lpips_ori = args.is_ssim * (1 - ssim_loss(sr_m_ori*255., hr*255.).item())
        if args.is_perceptual:
            # perp, _ = perceptual_loss(sr_m_ori, hr)
            # lpips_ori += args.is_perceptual * (perp.item())
            perp, _ = perceptual_loss(sr_m_ori_1, hr_1)
            lpips_ori += args.is_perceptual * (perp.item())
            perp, _ = perceptual_loss(sr_m_ori_2, hr_2)
            lpips_ori += args.is_perceptual * (perp.item())
        if args.is_gan:
            # fake_g_pred = model_ema_d(sr_m_ori)
            # l_g_gan = gan_loss(fake_g_pred, True, is_disc=False)
            # lpips_ori += args.is_gan * (l_g_gan.item())
            fake_g_pred = model_ema_d(sr_m_ori_1)
            l_g_gan = gan_loss(fake_g_pred, True, is_disc=False)
            lpips_ori += args.is_gan * (l_g_gan.item())
            fake_g_pred = model_ema_d(sr_m_ori_2)
            l_g_gan = gan_loss(fake_g_pred, True, is_disc=False)
            lpips_ori += args.is_gan * (l_g_gan.item())
        lpips_ori += args.is_l1loss * (l1_ori)

        loss_full_ori_l1 += l1_ori
        loss_full_ori_lpips += lpips_ori

        sr_output.append(sr_m_ori_1)
        sr_output.append(sr_m_ori_2)
        hr_output.append(hr_1)
        hr_output.append(hr_2)

    loss_ori_l1 =  loss_full_ori_l1 / len(train_for_EA_dataloader) / 2
    loss_ori_lpips = loss_full_ori_lpips / len(train_for_EA_dataloader) / 2
        
    return loss_ori_l1, loss_ori_lpips, sr_output, hr_output
    
def evl_model(device, ssim_loss, perceptual_loss, loss_func, loss_lpips_func, net_ori, valid_dataloaders, val_name, args):
    loss_full_ori_l1 = 0.0
    loss_full_ori_lpips = 0.0
    patch_size = args.patch_size
    inter = patch_size//args.scale
    l = 0
    for valid_dataloader in valid_dataloaders:
        name = valid_dataloader['name']
        if name == val_name:
            loader = valid_dataloader['dataloader']
            for lr, hr in loader:
                if args.range == 1:
                        lr,hr = lr / 255., hr / 255.
                
                lr, hr = lr.to(device), hr.to(device)
                sr_ori = net_ori(lr)
                l1_ori = loss_func(sr_ori, hr).item()
                if args.is_lpips:
                # convert to [-1,1]
                    lpips_ori += torch.mean(loss_lpips_func((sr_ori)*2-1, (hr)*2-1)).item()
                elif args.is_ssim:
                    lpips_ori = 1 - ssim_loss(sr_ori*255., hr*255.).item()
                elif args.is_perceptual:
                    lpips_ori, _ = perceptual_loss(sr_ori, hr)

                loss_full_ori_l1 += l1_ori
                loss_full_ori_lpips += lpips_ori

            loss_ori_l1 =  loss_full_ori_l1 / len(loader)
            loss_ori_lpips = loss_full_ori_lpips / len(loader)
        
    return loss_ori_l1, loss_ori_lpips


def model_ema(model_d, model_ema_d, decay):
    model_d_copy = copy.deepcopy(model_d)

    net_g_params = dict(model_d_copy.named_parameters())
    net_g_ema_params = dict(model_ema_d.named_parameters())

    for k in net_g_ema_params.keys():
        net_g_ema_params[k].data.mul_(decay).add_(net_g_params[k].data, alpha=1 - decay)

def get_bare_model(net):
    """Get bare model, especially under wrapping with
    DistributedDataParallel or DataParallel.
    """
    if isinstance(net, (DataParallel, DistributedDataParallel)):
        net = net.module
    return net
