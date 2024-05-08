import csv
import random
from datetime import time
import matplotlib
import numpy as np
from tomlkit import string
import matplotlib.pyplot as plt
import math
c=0.1
REPEAT_COUNT=100
RESULT_DIC=[]
X=[]
Rate1=[]
Rate1pct=[]
Rate10pct=[]
Selet_Value_high=[]
Select_Value_low=[]
Select_Value_aver=[]
for c in range(1,7000,1000):
    c=c/10000
    PERCENT1_COUNT=0
    PERCENT10_COUNT=0
    TARGET_COUNT=0
    X.append(c)
    S_V_H = float('-inf')
    S_V_L = float('inf')
    S_V_A = 0
    for k in range(REPEAT_COUNT):
        X1=[]
        res = []
        itemss = int(random.random() * 90)+10
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
header=['OBSERVE_COST','Rate1','Rate1pct','Rate10pct','Selet_Value_high','Select_Value_low','Select_Value_aver']
Data.append(X)
Data.append(Rate1)
Data.append(Rate1pct)
Data.append(Rate10pct)
Data.append(Selet_Value_high)
Data.append(Select_Value_low)
Data.append(Select_Value_aver)
Data=np.array(Data)
Data=Data.T
with open('algorithm2_mean_data.csv', mode='w', encoding='utf-8',newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    for row in Data:
        writer.writerow(row)
file.close()

plt.scatter(X,Rate1,color="blue")
plt.scatter(X,Rate1pct,color="green")
plt.scatter(X,Rate10pct,color="red")
plt.show()

plt.scatter(X, Selet_Value_high,color="blue")
plt.scatter(X, Select_Value_low,color="green")
plt.scatter(X, Select_Value_aver,color="red")
plt.show()
