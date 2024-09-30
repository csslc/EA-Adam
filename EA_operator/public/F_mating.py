import numpy as np
import cupy as cp
def F_mating(Population,FrontValue,CrowdDistance):
    N,D = Population.shape
    MatingPool = cp.zeros((N,D))
    # Temp = MatingPool
    Rank = np.random.permutation(N)
    Pointer=0
    index = []
    for i in range(0,N,2):
        k = [0, 0]
        for j in range(2):
            if Pointer >= N:
                Rank = np.random.permutation(N)
                Pointer = 0

            p = Rank[Pointer]
            q = Rank[Pointer+1]
            if FrontValue[0,p] < FrontValue[0,q]:
                k[j] = p
            elif FrontValue[0,p] > FrontValue[0,q]:
                k[j] = q
            elif CrowdDistance[0,p] > CrowdDistance[0,q]:
                k[j] = p
            else:
                k[j] = q

            Pointer += 2
        MatingPool[i:i+2, :] = Population[k[0:2], :]
        index = np.hstack((index,k[0:2]))
        # MatingPool[i+1,:] = Population[k[1], :]
        # Temp[i:i+2,:] = Population[k[0:2],:]
    return cp.array(MatingPool),np.int64(index)


