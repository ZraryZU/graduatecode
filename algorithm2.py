import random
from datetime import time
import matplotlib
import numpy as np
from tomlkit import string
import matplotlib.pyplot as plt
import math
c=0.1
REPEAT_COUNT=10

TARGET_COUNT=0
PERCENT1_COUNT=0
RESULT_DIC=[]
X=[]
for i in range(REPEAT_COUNT):
    X.append(i)
for k in range(REPEAT_COUNT):
    X1=[]
    res = []
    itemss = 10
    for i in range(itemss):
        random_number = random.random()
        X1.append(random_number)
        res.append(X1[i]-i*c)
    res.sort()
    Vtemp=0
    if (c<=0.5) :
        Vtemp=1-math.sqrt(2*c)
    else :
        Vtemp=-c+1/2
    Maxpos=itemss-1
    Maxval=X1[itemss-1]-c*itemss
    for i in range(itemss):
        if X1[i] >= Vtemp:
            Maxpos=i
            Maxval=X1[i]-i*c
            break
    Range=0
    for j in range(itemss):
        if(Maxval==res[j]):
            Range=j
    RESULT_DIC.append(Range+1)
    if Range>=itemss-1:
        TARGET_COUNT+=1
    if Range>=itemss*0.99:
        PERCENT1_COUNT += 1


print("最大值命中率:")
print(float(TARGET_COUNT/REPEAT_COUNT))
print("前1%命中率:")
print(float(PERCENT1_COUNT/REPEAT_COUNT))
plt.scatter(X,RESULT_DIC)
plt.show()