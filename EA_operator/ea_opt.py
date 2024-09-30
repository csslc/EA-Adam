# EA optimizer
import numpy as np
import os
import torch
from collections import OrderedDict
from public import NDsort, F_distance, F_EnvironmentSelect
class eaopt():
    '''
    Args:
        params (iterable): iterable of loss groups(dicts) to optimize
    '''
    def __init__(self, params, kwargs):
        self.cro = kwargs['cro']# type of crossover
        self.mut = kwargs['mut']# type of mutation
        self.sel = kwargs['sel']# type of selection
        self.popsize = kwargs['sel']# number of popsize
        # self.indsize = kwargs['indsize']# number of individual
        self.current_iter = params['current_iter']


    def step(self, popsize, model_pop, loss_pop, sel_type):
        # selction
        if sel_type == 'simple':
            opt_ind, opt_loss = get_opt(model_pop, loss_pop, popsize)
            for i in range(len(model_pop)):
                if key[i] not in K: # 判断newpop中是否有该文件，若有：保留；没有：删除。
                    model_pop.pop(key[i])

        elif sel_type == 'non_dom':
            N = len(model_pop)
            Remain_Num = popsize
            l_pixel = loss_pop.get('l_pix')
            l_percep = loss_pop.get('l_percep')
            # the same sort
            model_pop_sort = OrderedDict()
            for key in list(l_percep.keys()):
                model_pop_sort[key] = model_pop.get(key)

            f1 = torch.tensor(list(l_pixel.values())) # pixel loss
            f2 = torch.tensor(list(l_percep.values())) # perceptual loss
            PopObj = np.vstack((f1,f2)).T
            FrontValue = NDsort.NDSort(PopObj, Remain_Num)[0] # 基于dominant排序
            CrowdDistance = F_distance.F_distance(PopObj, FrontValue)
            # 计算拥挤度距离
            new_pop, FunctionValue, FrontValue, CrowdDistance = F_EnvironmentSelect.F_EnvironmentSelect(
            model_pop_sort, PopObj, Remain_Num)
            model_pop = new_pop
            ind_adam = 0
            opt_ind = None
            opt_loss = 0

        elif sel_type == 'melite':
            m = 0.7
            m_pop = int(m * popsize)
            oth_pop = int(popsize - m_pop)
            l_total = loss_pop.get('l_total')
            Fitness = torch.tensor(list(l_total.values()))
            f_values, slind = torch.sort(Fitness)
            index = slind[0:m_pop]
            index = index.cuda()
            key = list(model_pop.keys())
            K = np.array(key)[index.cpu()] # 选择出来的新的 model index
            opt_ind = model_pop.get(K[0])
            opt_loss = Fitness[index[0]]
            # 需要添加的 ind
            otindex = np.random.choice(len(l_total) - m_pop, oth_pop, replace=False)
            indc = 0
            ind_adam = 0
            for i in range(len(model_pop)):
                if key[i] in K:
                    if key[i].__contains__('Adam'):
                        ind_adam = ind_adam + 1 # 子代中选取的 adam 的数目
                else: # 判断 newpop 中是否有该文件，若有：保留；没有：删除且赋给pop_other。
                    if indc not in otindex:
                        model_pop.pop(key[i])
                    indc = indc + 1

        elif sel_type == 'roulette-wheel_selection':
            Fitness = torch.stack(tuple(loss_pop.values()))
            Key = loss_pop.keys()
            f_values, slind = torch.sort(Fitness) # sort from small to large
            Full = Fitness/torch.sum(Fitness) # normalization
            p_sele, _ = torch.sort(Full, descending=True) # sort from large to small
            # p_sele = svalues[slind] # selection probalibity for each inds
            p_sele = torch.cumsum(p_sele, dim=0)
            index = np.random.rand(popsize)
            for i in range(popsize):
                cin = index[0]
                z = torch.zeros(p_sele.shape)
                cin = torch.where(p_sele <= cin, z, Full)


        return opt_ind, opt_loss, model_pop, ind_adam


def get_opt(model_pop, loss_pop, m_pop):

    l_total = loss_pop.get('l_total')
    Fitness = torch.stack(tuple(l_total.values()))
    f_values, slind = Fitness.sort(0, False)
    index = slind[0:m_pop]
    index = index.cuda()
    key = list(model_pop.keys())
    K = np.array(key)[index.cpu()] #选择出来的新的model index
    opt_ind = model_pop.get(K[0])
    opt_loss = Fitness[index[0]]

    return opt_ind, opt_loss




