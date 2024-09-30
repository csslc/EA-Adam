# from: https://github.com/WANG-Chaoyue/EvolutionaryGAN
from EA_operator.public import NDsort, F_distance, F_EnvironmentSelect
from collections import OrderedDict
import numpy as np

def non_dom_sel(model_pop, loss_pop, args):
    N = len(model_pop)
    Remain_Num = args.popsize
    f1 = loss_pop[0].detach().numpy()
    f2 = loss_pop[1].detach().numpy()
    PopObj = np.vstack((f1,f2)).T
    # 计算拥挤度距离
    new_pop, new_ind, FunctionValue, FrontValue, CrowdDistance = F_EnvironmentSelect.F_EnvironmentSelect(
    model_pop, PopObj, Remain_Num)

    return new_pop, new_ind