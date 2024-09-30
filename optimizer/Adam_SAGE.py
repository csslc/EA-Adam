import math
import torch
from torch.optim.optimizer import Optimizer
from .norm_grd import centralized_gradient, normalized_gradient

class Adam_SAGE(Optimizer):
    r"""
    Implements Sensitvity-Guided AdamW (AdamW-SAGE) algorithm with weight decay fix as
    introduced in `No Parameters Left Behind: Sensitivity Guided Adaptive
    Learning Rate for Training Large Transformer Models `__.
    
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """
    def __init__(
        self,
        params,
        lr = 1e-3,
        t_total = -1,
        betas = (0.9, 0.999),
        beta3 = 0.75,
        eps = 1e-8,
        weight_decay = 0.0,
        correct_bias = True,
        max_grad_norm = 0.0
    ):
        betas = betas + (beta3,)
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[2]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, t_total=t_total, \
                        betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias, max_grad_norm=max_grad_norm)
        super(Adam_SAGE, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam_SAGE, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure = None):
        """
        Performs a single optimization step.
        Arguments:
            closure (:obj:`Callable`, `optional`): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta1, beta2, beta3 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    state['exp_avg_ipt'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq, exp_avg_ipt = state['exp_avg'], state['exp_avg_sq'], state['exp_avg_ipt']

                state["step"] += 1
                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                # Update moving average of importance score
                ipt = (p.data * grad).abs()
                exp_avg_ipt.mul_(beta3).add_(ipt, alpha=1.0 - beta3)

                step_size = group["lr"]

                if group['t_total'] != -1:
                    schedule_fct = schedule_func(group['schedule'])
                    step_size = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    step_size = group['lr']

                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    bias_correction3 = 1.0 - beta3 ** state["step"]
                    exp_avg_ipt_bias_corrected = exp_avg_ipt / bias_correction3
                    step_size *= (ipt - exp_avg_ipt_bias_corrected).abs() / (exp_avg_ipt_bias_corrected + group["eps"]**2)
                    step_size *= math.sqrt(bias_correction2) / bias_correction1
                else:
                    step_size *= (ipt - exp_avg_ipt).abs() / (exp_avg_ipt + group["eps"]**2)

                p.data -= exp_avg / denom * step_size

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])
        return loss