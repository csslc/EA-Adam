# select new individuals of population
import numpy as np
import os
import torch
from collections import OrderedDict


def Update_pop(model, popsize, train_data, type, path_para):
    # calculate loss values
    cur_path = os.getcwd()
    cur_file = os.listdir(cur_path + '/' + path_para)
    loss_pop = OrderedDict()
    loss = []
    for file in range(len(cur_file)):
        model.feed_data(train_data)
        model.net_g.load_state_dict(torch.load(cur_path + '/' + path_para + '/' + cur_file[file]))
        l = model.cal_loss()
        loss.append(l)
        loss_pop[cur_file[file]] = l
    ind_loss = loss.index(min(loss))
    model.net_g.load_state_dict(torch.load(cur_path + '/' + path_para + '/' + cur_file[file]))
    # model.net_g_ema.load_state_dict(torch.load(cur_path + '/' + path_para + '/' + cur_file[file]))
    print(min(loss))

    if type == 'roulette-wheel_selection':
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
    if type == 'simple':
        Fitness = torch.stack(tuple(loss_pop.values()))
        key = list(loss_pop.getkeys())
        f_values, slind = torch.sort(Fitness)
        index = slind[0:popsize]
        new_pop = np.array(key)[index.cpu()]

        # for i in range(len(cur_file)):
        #     if cur_file[i] not in new_pop: # 判断newpop中是否有该文件，若有：保留；没有：删除。
        #         path = cur_path + '/' + path_para + '/' + cur_file[i]
        #         os.remove(path)
        #         del loss_pop[cur_file[i]]
        # opt_ind, new_pop



















    #      Fitness = reshape(Fitness,1,[]);
    # Fitness = Fitness + min(min(Fitness),0);
    # Fitness = cumsum(1./Fitness);
    # Fitness = Fitness./max(Fitness);
    # index   = arrayfun(@(S)find(rand<=Fitness,1),1:N);