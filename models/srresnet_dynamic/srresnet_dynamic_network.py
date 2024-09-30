import torch
from torch import nn as nn
from torch.nn import functional as F
from models.srresnet_dynamic.srresnet_dynamic_block import ResidualBlockNoBNDynamic, make_layer, Dynamic_conv2d

def create_model(args):
    return MSRResNetDynamic(args)

class MSRResNetDynamic(nn.Module):

    def __init__(self, args):
        super(MSRResNetDynamic, self).__init__()

        self.upscale = args.scale
        num_in_ch = args.colors
        num_out_ch = args.colors
        num_feat = args.num_feat
        num_block = args.num_block
        num_models = args.num_network

        self.conv_first = Dynamic_conv2d(args, num_in_ch, num_feat, 3, groups=1, if_bias=True, K=num_models)
        self.body = make_layer(ResidualBlockNoBNDynamic, num_block, args = args, num_feat=num_feat, num_models=num_models)

        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = Dynamic_conv2d(args, num_feat, num_feat * self.upscale * self.upscale, 3, groups=1, if_bias=True, K=num_models)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = Dynamic_conv2d(args, num_feat, num_feat * 4, 3, groups=1, if_bias=True, K=num_models)
            self.upconv2 = Dynamic_conv2d(args, num_feat, num_feat * 4, 3, groups=1, if_bias=True, K=num_models)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = Dynamic_conv2d(args, num_feat, num_feat, 3, groups=1, if_bias=True, K=num_models)
        self.conv_last = Dynamic_conv2d(args, num_feat, num_out_ch, 3, groups=1, if_bias=True, K=num_models)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, weights):
        out1, weight = self.conv_first(x)
        weight = weight.detach().cpu()
        weights.append(weight)
        out = self.lrelu(out1)
        out, weights = self.body([out, weights])

        if self.upscale == 4:
            out, weight = self.upconv1(out)
            weight = weight.detach().cpu()
            weights.append(weight)
            out = self.lrelu(self.pixel_shuffle(out))
            out, weight = self.upconv2(out)
            weight = weight.detach().cpu()
            weights.append(weight)
            out = self.lrelu(self.pixel_shuffle(out))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(out))

        out, weight = self.conv_hr(out)
        weight = weight.detach().cpu()
        weights.append(weight)
        out = self.lrelu(out)
        out, weight = self.conv_last(out)
        weight = weight.detach().cpu()
        weights.append(weight)
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base
        return out, weights
            

    # def forward(self, x, weights):
    #     out = self.lrelu(self.conv_first({'x': x, 'weights': weights}))
    #     out = self.body({'x': out, 'weights': weights})['x']

    #     if self.upscale == 4:
    #         out = self.lrelu(self.pixel_shuffle(self.upconv1({'x': out, 'weights': weights})))
    #         out = self.lrelu(self.pixel_shuffle(self.upconv2({'x': out, 'weights': weights})))
    #     elif self.upscale in [2, 3]:
    #         out = self.lrelu(self.pixel_shuffle(self.upconv1({'x': out, 'weights': weights})))

    #     out = self.lrelu(self.conv_hr({'x': out, 'weights': weights}))
    #     out = self.conv_last({'x': out, 'weights': weights})
    #     base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
    #     out += base
    #     return out
