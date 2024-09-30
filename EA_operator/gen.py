# This code is for generating individuals
import numpy as np
import pickle as pkl
import functools
import torch
import os
import copy
from functools import reduce


def gen_cross(net_g, pt1, pt2, pt3, type, mu, type_model):
    if type=='SBX':
        net_g2 = copy.deepcopy(net_g)
        net = copy.deepcopy(net_g)
        new_model_para1, new_model_para2 = {}, {}
        for k,v in pt1.state_dict().items():
            if (k[-1] == 't') or (k[-1] == 's'):
                p1 = v
                p2 = pt2.state_dict().get(k)
                miu = torch.Tensor(1).uniform_(0,1)
                if miu<0.5:
                    siggam = torch.pow((2*miu),(1/21))
                else:
                    siggam = torch.pow((2-2*miu),-1/21)
                siggam = siggam.cuda()
                off1 = 0.5*((1+siggam)*p1 + (1-siggam)*p2)
                off2 = 0.5*((1-siggam)*p1 + (1+siggam)*p2)
                new_model_para1[k] = off1
                new_model_para2[k] = off2
            else:
                new_model_para1[k] = v
                new_model_para2[k] = pt2.state_dict().get(k)
        net.load_state_dict(new_model_para1, strict = True)
        net_g2.load_state_dict(new_model_para2, strict = True)

    return net, net_g2

def gen_mut(net_g, pt1, pt2, pt3, type, mu, type_model, only_conv = True):
    net = copy.deepcopy(net_g)
    new_model_para = {}
    if type=='simple':
        for k,v in pt1.state_dict().items():
            if only_conv:
                if k[-1] == 't':
                    a = torch.rand(v.shape).to(v.device)
                    b= a<mu
                    b=b.int().to(v.device)
                    p_mut = torch.randn(v.shape) * 0.03
                    p_mut = p_mut.to(v.device)
                    off = v + b * p_mut
                    new_model_para[k] = off

            else:
                if k[-1] != 's':
                    total_size= v.shape[0] * v.shape[1] * v.shape[2] * v.shape[3]
                else:
                    total_size= v.shape[0]
                nmu = round(mu*total_size)
                index_mu = np.random.choice(total_size, nmu, replace=False)
                p1 = v.reshape(total_size)
                p_mut = torch.randn(nmu) * 0.01
                p_mut = p_mut.cuda()
                p1[index_mu] = p_mut
                off = p1.reshape(v.shape)
                new_model_para[k] = off

        net.load_state_dict(new_model_para, strict = False)
    return net