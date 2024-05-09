import csv
import random
import pandas as pd
from datetime import time
import matplotlib
import numpy as np
from tomlkit import string
import matplotlib.pyplot as plt
import math
REPEAT_COUNT_LOW=50
REPEAT_COUNT_HIGH=10000
REPEAT_COUNT_STEP=50
X=[]
Rate1=[]
Rate1pct=[]
Rate10pct=[]
Selet_Value_high=[]
Select_Value_low=[]
Select_Value_aver=[]
for REPEAT_COUNT in range(REPEAT_COUNT_LOW,REPEAT_COUNT_HIGH,REPEAT_COUNT_STEP):
    PERCENT1_COUNT=0
    PERCENT10_COUNT=0
    TARGET_COUNT=0
    X.append(REPEAT_COUNT)
    S_V_H = float('-inf')
    S_V_L = float('inf')
    S_V_A = 0
    for k in range(REPEAT_COUNT):
        X1=[]
        res = []
        itemss = int(random.random() * 1000)+1000
        for i in range(itemss):
            random_number = random.random()
            X1.append(random_number)
            res.append(X1[i])
        res.sort()
        Vtemp=0
        Maxpos=itemss-1
        Maxval=X1[itemss-1]
        for i in range(itemss):
            if i > itemss/math.e and X1[i]>=Vtemp:
                Maxpos=i
                Maxval=X1[i]
                break
            if i <= itemss/math.e:
                Vtemp=max(Vtemp,X1[i])
        Range=0
        for j in range(itemss):
            if(Maxval==res[j]):
                Range=j
        if Range>=itemss-1:
            TARGET_COUNT += 1
        if Range>=itemss*0.99:
            PERCENT1_COUNT += 1
        if Range>=itemss*0.9:
            PERCENT10_COUNT += 1
        S_V_A += Maxval
        S_V_H = max(S_V_H,Maxval)
        S_V_L = min(S_V_L,Maxval)
    S_V_A = S_V_A/REPEAT_COUNT
    Rate1.append(TARGET_COUNT/REPEAT_COUNT)
    Rate1pct.append(PERCENT1_COUNT/REPEAT_COUNT)
    Rate10pct.append(PERCENT10_COUNT/REPEAT_COUNT)
    Selet_Value_high.append(S_V_H)
    Select_Value_low.append(S_V_L)
    Select_Value_aver.append(S_V_A)

#数据保存
Data=[]
header=['REPEAT_COUNT','Rate1','Rate1pct','Rate10pct','Selet_Value_high','Select_Value_low','Select_Value_aver']
Data.append(X)
Data.append(Rate1)
Data.append(Rate1pct)
Data.append(Rate10pct)
Data.append(Selet_Value_high)
Data.append(Select_Value_low)
Data.append(Select_Value_aver)
Data=np.array(Data)
Data=Data.T
with open('algorithm1_mean_data.csv', mode='w', encoding='utf-8',newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    for row in Data:
        writer.writerow(row)
file.close()

plt.scatter(X,Rate1,color="blue")
plt.scatter(X,Rate1pct,color="green")
plt.scatter(X,Rate10pct,color="red")

plt.show()

plt.plot(X, Selet_Value_high,color="blue")
plt.plot(X, Select_Value_low,color="green")
plt.plot(X, Select_Value_aver,color="red")

plt.show()



