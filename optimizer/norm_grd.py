import torch
import copy
from einops import repeat, rearrange
#from torch.optim.optimizer import Optimizer, required



def centralized_gradient(x,use_gc=True,gc_conv_only=False):
    if use_gc:
      if gc_conv_only:
        if len(list(x.size()))>3:
            x.add_(-x.mean(dim = tuple(range(1,len(list(x.size())))), keepdim = True))
      else:
        if len(list(x.size()))>1:
            x.add_(-x.mean(dim = tuple(range(1,len(list(x.size())))), keepdim = True))
    return x    

def normalized_gradient(x,use_norm=True, norm_conv_only=False):
    if use_norm:
      if norm_conv_only:
        if len(list(x.size()))>3:
            x = x/(torch.norm(x,p=2) + 0.0000001)
      else:
        x = x/(torch.norm(x,p=2) + 0.0000001)
    return x

def normalized_gradient_effc(x,groups,use_norm=True, norm_conv_only=False):
    if use_norm:
      if norm_conv_only:
        if len(list(x.size()))>3:
          c = int(x.shape[0]/groups)
          x_sub_all = torch.chunk(x,groups,dim=0)
          x_sub = torch.stack(x_sub_all,0)
          a = torch.norm(x_sub,p=2,dim=[1,2,3,4])
          a = repeat(a, 'h -> h b c k1 k2', b=c, c=x.shape[1],k1=x.shape[2],k2=x.shape[3])
          x_new = x_sub/(a+0.0000001)
          x_new = rearrange(x_new, 'h b c k1 k2 -> (h b) c k1 k2')
      else:
        if len(list(x.size()))>3:
          c = int(x.shape[0]/groups)
          x_sub_all = torch.chunk(x,groups,dim=0)
          x_sub = torch.stack(x_sub_all,0)
          a = torch.norm(x_sub,p=2,dim=[1,2,3,4])
          a = repeat(a, 'h -> h b c k1 k2', b=c, c=x.shape[1],k1=x.shape[2],k2=x.shape[3])
          x_new = x_sub/(a+0.0000001)
          x_new = rearrange(x_new, 'h b c k1 k2 -> (h b) c k1 k2')
        else:
          c = int(x.shape[0]/groups)
          x_sub_all = torch.chunk(x,groups,dim=0)
          x_sub = torch.stack(x_sub_all,0)
          a = torch.norm(x_sub,p=2,dim=[1])
          a = repeat(a, 'h -> h b', b=c)
          x_new = x_sub/(a+0.0000001)
          x_new = rearrange(x_new, 'h b-> (h b)')
    # for i in range(groups):
    #   x_sub = x_sub_all[i]
    #   if use_norm:
    #     if norm_conv_only:
    #       if len(list(x.size()))>3:
    #           x_sub = x_sub/(torch.norm(x_sub,p=2) + 0.0000001)
    #     else:
    #       x_sub = x_sub/(torch.norm(x_sub,p=2) + 0.0000001)
    #   x[c*i:c*(i+1),...] = x_sub
    return x_new

if __name__ == '__main__':
    x1 = torch.randn((1, 1, 64, 128))
    x2 = torch.randn((1, 1, 64, 128))*10
    print(sum(sum(sum(sum(x1**2))))/x1.shape[0]/x1.shape[1]/x1.shape[2]/x1.shape[3])
    print(sum(sum(sum(sum(x2**2))))/x1.shape[0]/x1.shape[1]/x1.shape[2]/x1.shape[3])
    x1_norm = normalized_gradient(x1,use_norm=True,norm_conv_only=False)
    x2_norm = normalized_gradient(x2,use_norm=True,norm_conv_only=False)
    print(sum(sum(sum(sum(x1_norm**2))))/x1.shape[0]/x1.shape[1]/x1.shape[2]/x1.shape[3])
    print(sum(sum(sum(sum(x2_norm**2))))/x1.shape[0]/x1.shape[1]/x1.shape[2]/x1.shape[3])
