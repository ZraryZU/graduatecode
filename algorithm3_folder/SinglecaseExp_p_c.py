import csv
import random
from datetime import time
import matplotlib
import numpy as np
from tomlkit import string
import matplotlib.pyplot as plt
import math
import copy
from scipy.interpolate import make_interp_spline
from mpl_toolkits.mplot3d import Axes3D

c=0.001
p=0.8
miu=1
REPEAT_COUNT=15
TARGET_COUNT=0
PERCENT1_COUNT=0
PERCENT10_COUNT=0
P_LOW=1
P_HIGH=10000
P_STEP=200
C_LOW=1
C_HIGH=10000
C_STEP=200
Rate1=[]
Rate1pct=[]
Rate10pct=[]
Selet_Value_high=[]
Select_Value_low=[]
Select_Value_aver=[]
X=[]
P=[]
Slt_v_aver_p_c=[]
for p in range(P_LOW,P_HIGH,P_STEP):
    p=p/P_HIGH
    slt_v_aver_c=[]
    for c in range(C_LOW,C_HIGH,C_STEP):
        c = c / C_HIGH
        X.append(c)
        P.append(p)
        #p=p/10000
        RESULT_DIC = []
        PERCENT1_COUNT = 0
        PERCENT10_COUNT = 0
        TARGET_COUNT = 0
        S_V_H = float('-inf')
        S_V_L = float('inf')
        S_V_A = 0
        for k in range(REPEAT_COUNT):
            X1 = []
            res = []
            itemss = int(random.random() * 90)+10
            xaxis = []
            re = 0
            B = []
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
                if(Maxval == res[j]):
                    Range = j
            RESULT_DIC.append(Range+1)
            if Range>=itemss-1:
                TARGET_COUNT += 1
            if Range>=itemss*0.99:
                PERCENT1_COUNT += 1
            if Range>=itemss*0.9:
                PERCENT10_COUNT += 1
            S_V_A += Maxval
            S_V_H = max(S_V_H, Maxval)
            S_V_L = min(S_V_L, Maxval)
        S_V_A = S_V_A / REPEAT_COUNT
        Rate1.append(TARGET_COUNT / REPEAT_COUNT)
        Rate1pct.append(PERCENT1_COUNT / REPEAT_COUNT)
        Rate10pct.append(PERCENT10_COUNT / REPEAT_COUNT)
        Selet_Value_high.append(S_V_H)
        Select_Value_low.append(S_V_L)
        Select_Value_aver.append(S_V_A)
        slt_v_aver_c.append(S_V_A)
    Slt_v_aver_p_c.append(slt_v_aver_c)


#数据保存
Data=[]
header=['OBSERVE_COST','BERNOULLI','Rate1','Rate1pct','Rate10pct','Selet_Value_high','Select_Value_low','Select_Value_aver']
Data.append(X)
Data.append(P)
Data.append(Rate1)
Data.append(Rate1pct)
Data.append(Rate10pct)
Data.append(Selet_Value_high)
Data.append(Select_Value_low)
Data.append(Select_Value_aver)
Data=np.array(Data)
Data=Data.T
with open('algorithm3_exp_data.csv', mode='w', encoding='utf-8',newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    for row in Data:
        writer.writerow(row)
file.close()


'''
plt.scatter(X,Rate1,color="blue")
plt.scatter(X,Rate1pct,color="green")
plt.scatter(X,Rate10pct,color="red")

plt.show()

X_np=np.array(X)
Y_np=np.array(Select_Value_aver)
x_new = np.linspace(X_np.min(), X_np.max(),100)
y_smooth = make_interp_spline(X_np, Y_np)(x_new)
 
plt.scatter(X, Selet_Value_high,color="blue")
plt.scatter(X, Select_Value_low,color="green")
plt.scatter(X, Select_Value_aver,color="red")
plt.plot(x_new, y_smooth,color="yellow")
plt.show()
'''

xx = np.arange(P_LOW/P_HIGH,P_HIGH/P_HIGH,P_STEP/P_HIGH)
yy = np.arange(C_LOW/C_HIGH,C_HIGH/C_HIGH,C_STEP/C_HIGH)
X_np, P_np = np.meshgrid(yy, xx)

Select_Value_aver_np=np.array(Select_Value_aver)
Slt_v_aver_p_c_np=np.array(Slt_v_aver_p_c)
ax3 = plt.axes(projection='3d')
ax3.plot_surface(X_np,P_np,Slt_v_aver_p_c_np,cmap='rainbow') 
plt.show()
#print("最大值命中率:")
#print(float(TARGET_COUNT/REPEAT_COUNT))
#print("前1%命中率:")
#print(float(PERCENT1_COUNT/REPEAT_COUNT))
#plt.scatter(X,RESULT_DIC)
#plt.show()