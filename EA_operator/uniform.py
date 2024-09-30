import torch

# generate weight vector uniformly

def uniform_point(popsize, start_point=[1,0], end_point=[0,1]):
    inter_point1 = (end_point[1] - start_point[1])/(popsize-1)
    w1 = torch.tensor([inter_point1*x for x in range(popsize)])
    w2 = torch.tensor([(1 - x) for x in w1])
    w = torch.vstack((w2, w1))
    return w