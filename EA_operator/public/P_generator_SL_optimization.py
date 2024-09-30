import cupy as np


def P_generator_SL(MatingPool,Pop_Gradient, Boundary, Coding, MaxOffspring):

    N, D = MatingPool.shape
    if MaxOffspring < 1 or MaxOffspring > N:
        MaxOffspring = N
    if Coding == "Real":
        ProC = 1
        ProM = 1/D
        DisC = 20
        DisM = 20

        Out = Pop_Gradient
        Offspring = np.zeros((N, D))

        for i in range(0, N, 2):

            flag = np.random.rand(1) > 0.5  #>1 时

            miu1 = np.random.rand(D,)/2
            miu2 = np.random.rand(D,)/2+0.5
            miu_temp = np.random.random((D,))



            dictor = MatingPool[i,:]>MatingPool[i+1,:]
            MatingPool[i][dictor], MatingPool[i + 1][dictor] = MatingPool[i + 1][dictor], MatingPool[i][dictor]
            Out[i][dictor], Out[i + 1][dictor] = Out[i + 1][dictor], Out[i][dictor]
            G_temp = Out[i:i + 2, :].copy()

            ##

            L = G_temp[0,:].copy()
            P = miu1.copy()
            P[L>0] = miu2[L>0].copy()
            P[ L== 0] = miu_temp[L==0].copy()
            miu = P.copy()

            beta = np.zeros((D,))
            beta[miu <= 0.5] = (2 * miu[miu <= 0.5]) ** (1 / (DisC + 1))
            beta[miu > 0.5] = (2 - 2 * miu[miu > 0.5]) ** (-1 / (DisC + 1))
            beta[np.random.random((D,)) > ProC] = 1

            if flag == True:
                beta[MatingPool[i] == 0] = 1

            Offspring[i, :] = ((MatingPool[i, :] + MatingPool[i + 1, :]) / 2) + (
                np.multiply(beta, (MatingPool[i, :] - MatingPool[i + 1, :]) / 2))

            ##




            L = -G_temp[0,:].copy()
            P = miu1.copy()
            P[L>0] = miu2[L>0].copy()
            P[L == 0] = miu_temp[L == 0].copy()
            miu = P.copy()

            beta = np.zeros((D,))
            beta[miu <= 0.5] = (2 * miu[miu <= 0.5]) ** (1 / (DisC + 1))
            beta[miu > 0.5] = (2 - 2 * miu[miu > 0.5]) ** (-1 / (DisC + 1))
            beta[np.random.random((D,)) > ProC] = 1

            if flag == True:
                beta[MatingPool[i + 1] == 0] = 1

            Offspring[i + 1, :] = ((MatingPool[i, :] + MatingPool[i + 1, :]) / 2) - (
                np.multiply(beta, (MatingPool[i, :] - MatingPool[i + 1, :]) / 2))




            Out[i][dictor], Out[i + 1][dictor] = Out[i + 1][dictor], Out[i][dictor]
             #

            k1 = np.random.rand(D,)>0.5
            L = G_temp[0,:].copy()
            k2 =Offspring[i,:]!=0
            kl1 = np.bitwise_and(k1, L < 0)

            L = -G_temp[1,:].copy()
            k2 = Offspring[i+1, :] != 0
            # kl2 = np.bitwise_and(np.bitwise_and(k1, L < 0), k2)
            kl2 = np.bitwise_and(k1, L < 0)


            Offspring[i][ kl1], Offspring[i+1][kl2] = Offspring[i+1][kl1], Offspring[i][kl2]
            Out[i][kl1], Out[i + 1][kl2] = Out[i + 1][kl1], Out[i][kl2]
            Offspring[i][dictor], Offspring[i + 1][dictor] = Offspring[i + 1][dictor], Offspring[i][dictor]





        Offspring_temp = Offspring[:MaxOffspring, :].copy()
        Offspring = Offspring_temp


        if MaxOffspring == 1:
            MaxValue = Boundary[0, :]
            MinValue = Boundary[1, :]
        else:
            MaxValue = np.tile(Boundary[0, :], (MaxOffspring, 1))
            MinValue = np.tile(Boundary[1, :], (MaxOffspring, 1))






        #
        k = np.random.random((MaxOffspring, D))
        miu = np.random.random((MaxOffspring, D))

        Temp = np.bitwise_and(k <= ProM, miu < 0.5)

        # Offspring[Temp] = Offspring[Temp] + np.multiply((MaxValue[Temp] - MinValue[Temp]),
        #                                                 ((2 * miu[Temp] + np.multiply(
        #                                                     1 - 2 * miu[Temp],
        #                                                     (1 - (Offspring[Temp] - MinValue[Temp]) / (
        #                                                                 MaxValue[Temp] - MinValue[Temp])) ** (
        #                                                                 DisM + 1))) ** (1 / (
        #                                                         DisM + 1)) - 1))
        Offspring[Temp] = 0
        Temp = np.bitwise_and(k <= ProM, miu >= 0.5)
        #
        # Offspring[Temp] = Offspring[Temp] + np.multiply((MaxValue[Temp] - MinValue[Temp]),
        #                                                 (1 - ((2 * (1 - miu[Temp])) + np.multiply(
        #                                                     2 * (miu[Temp] - 0.5),
        #                                                     (1 - (MaxValue[Temp] - Offspring[Temp]) / (
        #                                                                 MaxValue[Temp] - MinValue[Temp])) ** (
        #                                                                 DisM + 1))) ** (1 / (
        #                                                         DisM + 1))))
        Offspring[Temp] = 0

        Offspring[Offspring > MaxValue] = MaxValue[Offspring > MaxValue]
        Offspring[Offspring < MinValue] = MinValue[Offspring < MinValue]


    elif Coding == "Binary":
        Offspring = []

    elif Coding == "DE":
        Offspring = []

    return Offspring














