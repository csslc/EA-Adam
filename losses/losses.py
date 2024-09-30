import math
import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F

from losses.vgg_arch import VGGFeatureExtractor
# from vgg_arch import VGGFeatureExtractor
from torchvision.transforms.functional import normalize
from math import exp
from torch.autograd import Variable
import numpy as np
import lpips
import time
import cv2


class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss from basicsr
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram

class SSIMLoss(nn.Module):
    """
    Calculate SSIM (structural similarity) from BasicSR

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.

    Returns:
        float: ssim result.
    """
    def __init__(self, input_order='CHW', crop_border = 0):
        super(SSIMLoss, self).__init__()
        self.input_order = input_order
        self.crop_border = crop_border
        self.channel = 3

    def forward(self, img, img2):
        assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
        if self.input_order not in ['HWC', 'CHW']:
            raise ValueError(f'Wrong input_order {self.input_order}. Supported input_orders are "HWC" and "CHW"')
        b = img.shape[0]
        ssims_batch = torch.zeros(1).cuda()
        for j in range(b):
            img_b, img2_b = img[j], img2[j]
            img_b = self.reorder_image(img_b, input_order=self.input_order)
            img2_b = self.reorder_image(img2_b, self.input_order)

            if self.crop_border != 0:
                img_b = img_b[self.crop_border:-self.crop_border, self.crop_border:-self.crop_border, ...]
                img2_b = img2_b[self.crop_border:-self.crop_border, self.crop_border:-self.crop_border, ...]

            ssims = torch.zeros(1).cuda()
            for i in range(img_b.shape[2]):
                ssims += self._ssim(img_b[..., i], img2_b[..., i])
            ssims_batch += ssims/img_b.shape[2]
        
        return ssims_batch/b

    def _ssim(self, img, img2):
        """Calculate SSIM (structural similarity) for one channel images.

        It is called by func:`calculate_ssim`.

        Args:
            img (ndarray): Images with range [0, 255] with order 'HWC'.
            img2 (ndarray): Images with range [0, 255] with order 'HWC'.

        Returns:
            float: ssim result.
        """

        w, h = img.size()
        img, img2 = img.unsqueeze(0).unsqueeze(0), img2.unsqueeze(0).unsqueeze(0)
        window_size = 11
        sigma = 1.5 * window_size / 11
        window = self.create_window(window_size, sigma, 1).cuda()

        mu1 = F.conv2d(img, window, padding = window_size//2, groups = 1)
        mu2 = F.conv2d(img2, window, padding = window_size//2, groups = 1)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img*img, window, padding = window_size//2, groups = 1) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = 1) - mu2_sq
        sigma12 = F.conv2d(img*img2, window, padding = window_size//2, groups = 1) - mu1_mu2

        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2
        V1 = 2.0 * sigma12 + C2
        V2 = sigma1_sq + sigma2_sq + C2
        ssim_map = ((2*mu1_mu2 + C1)*V1)/((mu1_sq + mu2_sq + C1)*V2)
        mcs_map = V1 / V2
        return ssim_map.mean()


    def reorder_image(self, img, input_order):
        """Reorder images to 'HWC' order.

        If the input_order is (h, w), return (h, w, 1);
        If the input_order is (c, h, w), return (h, w, c);
        If the input_order is (h, w, c), return as it is.

        Args:
            img (ndarray): Input image.
            input_order (str): Whether the input order is 'HWC' or 'CHW'.
                If the input image shape is (h, w), input_order will not have
                effects. Default: 'HWC'.

        Returns:
            ndarray: reordered image.
        """

        if input_order not in ['HWC', 'CHW']:
            raise ValueError(f"Wrong input_order {input_order}. Supported input_orders are 'HWC' and 'CHW'")
        if len(img.shape) == 2:
            img = img[..., None]
        if input_order == 'CHW':
            # img = img.transpose(1, 2, 0)
            img = img.permute(1, 2, 0)
        return img

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, sigma, channel):
        _1D_window = self.gaussian(window_size, sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window


class GANLoss(nn.Module):
    """Define GAN loss.

    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'wgan_softplus':
            self.loss = self._wgan_softplus_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):
        """wgan loss.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return -input.mean() if target else input.mean()

    def _wgan_softplus_loss(self, input, target):
        """wgan loss with soft plus. softplus is a smooth approximation to the
        ReLU function.

        In StyleGAN2, it is called:
            Logistic loss for discriminator;
            Non-saturating loss for generator.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return F.softplus(-input).mean() if target else F.softplus(input).mean()

    def get_target_label(self, input, target_is_real):
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        if self.gan_type in ['wgan', 'wgan_softplus']:
            return target_is_real
        target_val = (self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.

        Returns:
            Tensor: GAN loss value.
        """
        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'hinge':
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight

