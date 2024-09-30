import collections.abc
import math
import torch
import torchvision
import warnings
from distutils.version import LooseVersion
from itertools import repeat
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm

def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)

def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)

def pixel_unshuffle(x, scale):
    """ Pixel unshuffle.

    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.

    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)
    
class Fusion(nn.Module):
    def __init__(self, args, in_planes):
        super(Fusion, self).__init__()

        in_nc = in_planes
        nf = args.num_feat_fusion
        num_params = args.num_params
        num_networks = args.num_network
        use_bias = args.use_bias

        self.ConvNet = nn.Sequential(*[
            nn.Conv2d(in_nc, nf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, num_params, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.LeakyReLU(0.2, True),
        ])

        self.globalPooling = nn.AdaptiveAvgPool2d((1, 1))

        self.MappingNet = nn.Sequential(*[
            nn.Linear(num_params, 15),
            nn.LeakyReLU(0.2, True),
            nn.Linear(15, num_networks),
            # nn.Sigmoid(),
            nn.Softmax(),
        ])

    def forward(self, input):
        conv = self.ConvNet(input)
        flat = self.globalPooling(conv)
        out_params = flat.view(flat.size()[:2])
        mapped_weights = self.MappingNet(out_params)

        return out_params, mapped_weights

    def forward(self, input):
        conv = self.ConvNet(input)
        flat = self.globalPooling(conv)
        out_params = flat.view(flat.size()[:2])
        mapped_weights = self.MappingNet(out_params)
        # normalization
        return out_params, mapped_weights

class Dynamic_conv2d(nn.Module):
    def __init__(self, args, in_planes, out_planes, kernel_size, stride=1, padding=1, dilation=1, groups=1, if_bias=True, K=5, init_weight=False):
        super(Dynamic_conv2d, self).__init__()
        assert in_planes % groups == 0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.if_bias = if_bias
        self.fusion = Fusion(args, in_planes)
        self.K = K

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=args.is_train_expert)
        if self.if_bias:
            self.bias = nn.Parameter(torch.Tensor(K, out_planes), requires_grad=args.is_train_expert)
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])
            if self.if_bias:
                nn.init.constant_(self.bias[i], 0)

    def forward(self, x):
        _, softmax_attention = self.fusion(x)
        batch_size, in_planes, height, width = x.size()
        x = x.contiguous().view(1, -1, height, width)
        weight = self.weight.view(self.K, -1)

        aggregate_weight = torch.mm(softmax_attention, weight).view(-1, self.in_planes, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output, softmax_attention


