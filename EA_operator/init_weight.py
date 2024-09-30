# This code for weight init
from torch.nn import init
import functools


def para_init(model, init_type, init_bn_type, scale=1):
    # init weights with different types:
        # for conv and linear: 0_normal; 1_uniform; 2_xavier_normal; 3_xavier_uniform;
        # 4_kaiming_normal; 5_kaiming_uniform; 6_orthogonal;
        # for BN: 0_uniform; 1_constant.
    def init_fn(m, init_type=2, init_bn_type=0, scale=1):
        classname = m.__class__.__name__

        if classname.find('Conv') != -1 or classname.find('Linear') != -1:
            if init_type == 0:
                init.normal_(m.weight.data, 0, 0.1)
                m.weight.data.clamp_(-1, 1).mul_(scale)

            elif init_type == 1:
                init.uniform_(m.weight.data, -0.2, 0.2)
                m.weight.data.mul_(scale)

            elif init_type == 2:
                init.xavier_normal_(m.weight.data, gain=scale)
                m.weight.data.clamp_(-1, 1)

            elif init_type == 3:
                init.xavier_uniform_(m.weight.data, gain=scale)

            elif init_type == 4:
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.clamp_(-1, 1).mul_(scale)

            elif init_type == 5:
                init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.mul_(scale)

            elif init_type == 6:
                init.orthogonal_(m.weight.data, gain=scale)

            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_type))

            if m.bias is not None:
                m.bias.data.zero_()

        elif classname.find('BatchNorm2d') != -1:

            if init_bn_type == 0:  # preferred
                if m.affine:
                    init.uniform_(m.weight.data, 0.1, 1.0)
                    init.uniform_(m.bias.data, 0.0)
            elif init_bn_type == 1:
                if m.affine:
                    init.constant_(m.weight.data, 1.0)
                    init.constant_(m.bias.data, 0.0)
            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_bn_type))

    if init_type not in ['default', 'none']:
        print('Initialization method [{:.2f} + {:.2f}], scale is [{:.2f}]'.format(init_type, init_bn_type, scale))
        fn = functools.partial(init_fn, init_type=init_type, init_bn_type=init_bn_type, scale=scale)
        model.apply(fn)
    else:
        print('Pass this initialization! Initialization was done during network definition!')

    return model