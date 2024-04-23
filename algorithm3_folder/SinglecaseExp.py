import random
from datetime import time
import matplotlib
import numpy as np
from tomlkit import string
import matplotlib.pyplot as plt
import math
import copy
c=0.01
p=0.8
miu=1
REPEAT_COUNT=1
TARGET_COUNT=0
PERCENT1_COUNT=0
RESULT_DIC=[]
X=[]
B=[]
for i in range(REPEAT_COUNT):
    X.append(i+1)
for k in range(REPEAT_COUNT):
    X1=[]
    res = []
    itemss = 100
    xaxis=[]
    re=0
    for i in range(itemss):
        random_number = np.random.exponential(scale=miu)
        random_bonulli= np.random.binomial(1,p)
        X1.append(random_number)
        B.append(random_bonulli)
        xaxis.append(i+1)
        re=(re+X1[i])*B[i]
        res.append(re-(i+1)*c)
    raw_res=copy.deepcopy(res)
    res.sort()
    Stemp=0
    if (p*miu>=c) :
        Stemp = miu / (1 - p) * math.log(p * miu / c)
    else :
        Stemp = 0
    Maxpos=itemss-1
    Maxval=X1[itemss-1]-c*itemss
    current_val=0
    flag=False
    for i in range(itemss):
        current_val=B[i]*(current_val+X1[i])
        if current_val >= Stemp:
            Maxval=current_val-(i+1)*c
            flag=True
            break
    if flag==False:
        Maxval = current_val - itemss * c
    Range=0
    for j in range(itemss):
        if(Maxval==res[j]):
            Range=j
    RESULT_DIC.append(Range+1)
    if Range>=itemss-1:
        TARGET_COUNT+=1
    if Range>=itemss*0.99:
        PERCENT1_COUNT += 1
    plt.scatter(xaxis,X1)
    plt.show()
    plt.bar(xaxis,raw_res)
    plt.show()
    print(Range)


#print("最大值命中率:")
#print(float(TARGET_COUNT/REPEAT_COUNT))
#print("前1%命中率:")
#print(float(PERCENT1_COUNT/REPEAT_COUNT))
#plt.scatter(X,RESULT_DIC)
#plt.show()