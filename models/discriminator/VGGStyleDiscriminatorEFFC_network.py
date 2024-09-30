import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

def create_model(args):
    return VGGStyleDiscriminator(args)

class VGGStyleDiscriminator(nn.Module):
    """VGG style discriminator with input size 128 x 128 or 256 x 256.

    It is used to train SRGAN, ESRGAN, and VideoGAN.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features.Default: 64.
    """

    def __init__(self, args):
        super(VGGStyleDiscriminator, self).__init__()

        num_in_ch = args.network_d.get('num_in_ch')
        num_feat = args.network_d.get('num_feat')
        self.groups = args.popsize
        self.num_feat = num_feat

        self.input_size = args.patch_size
        assert self.input_size == 128 or self.input_size == 256, (
            f'input size must be 128 or 256, but received {input_size}')

        self.conv0_0 = nn.Conv2d(num_in_ch*self.groups, num_feat*self.groups, 3, 1, 1, groups=self.groups, bias=True)
        self.conv0_1 = nn.Conv2d(num_feat*self.groups, num_feat*self.groups, 4, 2, 1, groups=self.groups, bias=False)
        self.bn0_1 = nn.BatchNorm2d(num_feat*self.groups, affine=True)

        self.conv1_0 = nn.Conv2d(num_feat*self.groups, num_feat * 2 * self.groups, 3, 1, 1, groups=self.groups, bias=False)
        self.bn1_0 = nn.BatchNorm2d(num_feat * 2 * self.groups, affine=True)
        self.conv1_1 = nn.Conv2d(num_feat * 2 * self.groups, num_feat * 2 * self.groups, 4, 2, 1, groups=self.groups, bias=False)
        self.bn1_1 = nn.BatchNorm2d(num_feat * 2 * self.groups, affine=True)

        self.conv2_0 = nn.Conv2d(num_feat * 2 * self.groups, num_feat * 4 * self.groups, 3, 1, 1, groups=self.groups, bias=False)
        self.bn2_0 = nn.BatchNorm2d(num_feat * 4 * self.groups, affine=True)
        self.conv2_1 = nn.Conv2d(num_feat * 4 * self.groups, num_feat * 4 * self.groups, 4, 2, 1, groups=self.groups, bias=False)
        self.bn2_1 = nn.BatchNorm2d(num_feat * 4 * self.groups, affine=True)

        self.conv3_0 = nn.Conv2d(num_feat * 4 * self.groups, num_feat * 8 * self.groups, 3, 1, 1, groups=self.groups, bias=False)
        self.bn3_0 = nn.BatchNorm2d(num_feat * 8 * self.groups, affine=True)
        self.conv3_1 = nn.Conv2d(num_feat * 8 * self.groups, num_feat * 8 * self.groups, 4, 2, 1, groups=self.groups, bias=False)
        self.bn3_1 = nn.BatchNorm2d(num_feat * 8 * self.groups, affine=True)

        self.conv4_0 = nn.Conv2d(num_feat * 8 * self.groups, num_feat * 8 * self.groups, 3, 1, 1, groups=self.groups, bias=False)
        self.bn4_0 = nn.BatchNorm2d(num_feat * 8 * self.groups, affine=True)
        self.conv4_1 = nn.Conv2d(num_feat * 8 * self.groups, num_feat * 8 * self.groups, 4, 2, 1, groups=self.groups, bias=False)
        self.bn4_1 = nn.BatchNorm2d(num_feat * 8 * self.groups, affine=True)

        if self.input_size == 256:
            self.conv5_0 = nn.Conv2d(num_feat * 8* self.groups, num_feat * 8* self.groups, 3, 1, 1, groups=self.groups, bias=False)
            self.bn5_0 = nn.BatchNorm2d(num_feat * 8* self.groups, affine=True)
            self.conv5_1 = nn.Conv2d(num_feat * 8* self.groups, num_feat * 8* self.groups, 4, 2, 1, groups=self.groups, bias=False)
            self.bn5_1 = nn.BatchNorm2d(num_feat * 8* self.groups, affine=True)

        self.globalPooling = nn.AdaptiveAvgPool2d((1, 1))

        self.linear1 = nn.ModuleList([nn.Linear(num_feat * 8, 100) for i in range(self.groups)])
        self.linear2 = nn.ModuleList([nn.Linear(100, 1) for i in range(self.groups)])
        
        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        with torch.cuda.amp.autocast():
            # assert x.size(2) == self.input_size, (f'Input size must be identical to input_size, but received {x.size()}.')

            feat = self.lrelu(self.conv0_0(x))
            feat = self.lrelu(self.bn0_1(self.conv0_1(feat)))  # output spatial size: /2

            feat = self.lrelu(self.bn1_0(self.conv1_0(feat)))
            feat = self.lrelu(self.bn1_1(self.conv1_1(feat)))  # output spatial size: /4

            feat = self.lrelu(self.bn2_0(self.conv2_0(feat)))
            feat = self.lrelu(self.bn2_1(self.conv2_1(feat)))  # output spatial size: /8

            feat = self.lrelu(self.bn3_0(self.conv3_0(feat)))
            feat = self.lrelu(self.bn3_1(self.conv3_1(feat)))  # output spatial size: /16

            feat = self.lrelu(self.bn4_0(self.conv4_0(feat)))
            feat = self.lrelu(self.bn4_1(self.conv4_1(feat)))  # output spatial size: /32

            if self.input_size == 256:
                feat = self.lrelu(self.bn5_0(self.conv5_0(feat)))
                feat = self.lrelu(self.bn5_1(self.conv5_1(feat)))  # output spatial size: / 64

            # spatial size: (4, 4)
            feat = self.globalPooling(feat)
            out = []
            for i in range(self.groups):
                feat_sub = feat[:,i*self.num_feat * 8:(i+1)*self.num_feat * 8,...]
                feat_sub = feat_sub.view(feat_sub.size(0), -1)
                feat_sub = self.lrelu(self.linear1[i](feat_sub))
                out.append(self.linear2[i](feat_sub))
        return out