#含代价实验模型

import random
from datetime import time

import numpy as np

n = 1000
m = 0
for k in range(n):
    X1 = []
    for i in range(1000):
        random_number = random.random()
        X1.append(random_number)
    # print(X1)
    Xsrt = X1[0:]
    X2 = X1[0:368]
    Xsrt.sort()
    # print(X1)
    # print(Xsrt)
    #X2.sort()
    max1 = X2[367]
    min1 = X2[0]
    M1 = max1
    M5 = min1
    M4 = X2[92]
    M3 = X2[184]
    M2 = X2[276]
    maxval=-4
    P = np.zeros((6, 6), dtype=float)
    for i in range(337):
        maxval=max(maxval,X2[i]-i*0.0001)
    mxpos = 0
    cnt = 0
    for i in range(367, 1000):
        mxpos=i
        if  (X1[i]-i*0.0001)>maxval:
            break
    # print(mxpos)
    # print(X1[mxpos])
    max1 = X1[mxpos]
    posmx = 0
    for i in range(1000):
        if Xsrt[i] == max1:
            posmx = i
            print(i)
            break
    if posmx >= 999:
        m += 1
print(float(m / n))