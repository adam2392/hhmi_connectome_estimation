import numpy as np

N = 3;          # number of neurons
T = 100;        # number of timesteps

X = np.random.rand(N,);

C = np.random.rand(N,N);

indEx = C <= .2;
indIn = C >= .8;
indNo = (C > .2) & (C < .8);

C[indEx] = 1;
C[indIn] = -1;
C[indNo] = 0;

for i in range(T):
    