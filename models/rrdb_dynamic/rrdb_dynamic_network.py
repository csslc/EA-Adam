import torch
from torch import nn as nn
from torch.nn import functional as F

from models.rrdb_dynamic.rrdb_dynamic_block import default_init_weights, make_layer, pixel_unshuffle, Dynamic_conv2d

def create_model(args):
    return RRDBNetDynamic(args)

class ResidualDenseBlockDynamic(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, args, num_feat, num_grow_ch):
        super(ResidualDenseBlockDynamic, self).__init__()
        num_models = args.num_network

        self.conv1 = Dynamic_conv2d(args, num_feat, num_grow_ch, 3, groups=1, if_bias=True, K=num_models)
        self.conv2 = Dynamic_conv2d(args, num_feat + num_grow_ch, num_grow_ch, 3, groups=1, if_bias=True, K=num_models)
        self.conv3 = Dynamic_conv2d(args, num_feat + 2 * num_grow_ch, num_grow_ch, 3, groups=1, if_bias=True, K=num_models)
        self.conv4 = Dynamic_conv2d(args, num_feat + 3 * num_grow_ch, num_grow_ch, 3, groups=1, if_bias=True, K=num_models)
        self.conv5 = Dynamic_conv2d(args, num_feat + 4 * num_grow_ch, num_feat, 3, groups=1, if_bias=True, K=num_models)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)


    def forward(self, inputs):
        weights = inputs[1]
        x = inputs[0]
        identity = x.clone()
        out, weight = self.conv1(x)
        weight = weight.detach().cpu()
        weights.append(weight)
        x1 = self.lrelu(out)
        out, weight = self.conv2(torch.cat((identity, x1), 1))
        weight = weight.detach().cpu()
        weights.append(weight)
        x2 = self.lrelu(out)
        out, weight = self.conv3(torch.cat((identity, x1, x2), 1))
        weight = weight.detach().cpu()
        weights.append(weight)
        x3 = self.lrelu(out)
        out, weight = self.conv4(torch.cat((identity, x1, x2, x3), 1))
        weight = weight.detach().cpu()
        weights.append(weight)
        x4 = self.lrelu(out)
        x5, weight = self.conv5(torch.cat((identity, x1, x2, x3, x4), 1))
        weight = weight.detach().cpu()
        weights.append(weight)
        out = x5 * 0.2 + identity
        outputs = [out, weights]
        # Empirically, we use 0.2 to scale the residual for better performance
        return outputs


class RRDBDynamic(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, args, num_feat, num_grow_ch):
        super(RRDBDynamic, self).__init__()

        self.rdb1 = ResidualDenseBlockDynamic(args, num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlockDynamic(args, num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlockDynamic(args, num_feat, num_grow_ch)

    def forward(self, inputs):
        weights = inputs[1]
        x = inputs[0]
        identity = x.clone()
        outputs = self.rdb1([x, weights])
        outputs = self.rdb2(outputs)
        outputs = self.rdb3(outputs)
        out = outputs[0]
        weights = outputs[1]
        # Empirically, we use 0.2 to scale the residual for better performance
        out_x = out * 0.2 + identity
        return [out_x, weights]


class RRDBNetDynamic(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(self, args):
    # num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(RRDBNetDynamic, self).__init__()
        scale = args.scale
        self.scale = args.scale
        num_in_ch = args.colors
        num_out_ch = args.colors
        num_feat = args.num_feat
        num_block = args.num_block
        num_models = args.num_network

        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = Dynamic_conv2d(args, num_in_ch, num_feat, 3, groups=1, if_bias=True, K=num_models)
        self.body = make_layer(RRDBDynamic, num_block, args = args, num_feat = num_feat, num_grow_ch = args.num_grow_ch)
        self.conv_body = Dynamic_conv2d(args, num_feat, num_feat, 3, groups=1, if_bias=True, K=num_models)
        # upsample
        self.conv_up1 = Dynamic_conv2d(args, num_feat, num_feat, 3, groups=1, if_bias=True, K=num_models)
        self.conv_up2 = Dynamic_conv2d(args, num_feat, num_feat, 3, groups=1, if_bias=True, K=num_models)
        self.conv_hr = Dynamic_conv2d(args, num_feat, num_feat, 3, groups=1, if_bias=True, K=num_models)
        self.conv_last = Dynamic_conv2d(args, num_feat, num_out_ch, 3, groups=1, if_bias=True, K=num_models)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, weights):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat, weight = self.conv_first(feat)
        weight = weight.detach().cpu()
        weights.append(weight)

        out_body, weights = self.body([feat, weights]) 
        body_feat, weight = self.conv_body(out_body)
        weight = weight.detach().cpu()
        weights.append(weight)
        feat = feat + body_feat
        # upsample
        out_up1, weight = self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest'))
        weight = weight.detach().cpu()
        weights.append(weight)

        feat = self.lrelu(out_up1)
        out_up2, weight = self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest'))
        weight = weight.detach().cpu()
        weights.append(weight)
        feat = self.lrelu(out_up2)

        out_last, weight = self.conv_hr(feat)
        weight = weight.detach().cpu()
        weights.append(weight)
        feat = self.lrelu(out_last)

        out, weight = self.conv_last(feat)
        weight = weight.detach().cpu()
        weights.append(weight)
        return out, weights
