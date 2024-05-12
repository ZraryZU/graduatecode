import csv
import random
import pandas as pd
from datetime import time
import matplotlib
import numpy as np
from scipy.interpolate import make_interp_spline
from tomlkit import string
import matplotlib.pyplot as plt
import math


P_LOW = 1
P_HIGH = 1000
P_STEP = 8
C_LOW = 1
C_HIGH = 1000
C_STEP = 8

X=[]
B=[]
Rate1=[]
Rate1pct=[]
Rate10pct=[]
Select_Value_high=[]
Select_Value_low=[]
Select_Value_aver=[]
X_cp=[]
B_cp=[]
Rate1_cp=[]
Rate1pct_cp=[]
Rate10pct_cp=[]
Select_Value_high_cp=[]
Select_Value_low_cp=[]
Select_Value_aver_cp=[]

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

cnt = 0
for i in range(16):
    with open('algorithm3_exp_data_'+str(i+1)+'.csv', mode='r', encoding='utf-8',newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            cnt+=1
            X.append(float(row['BERNOULLI']))
            B.append(float(row['OBSERVE_COST']))
            Rate1.append(float(row['Rate1']))
            Rate1pct.append(float(row['Rate1pct']))
            Rate10pct.append(float(row['Rate10pct']))
            Select_Value_high.append(float(row['Selet_Value_high']))
            Select_Value_low.append(float(row['Select_Value_low']))
            Select_Value_aver.append(float(row['Select_Value_aver']))
            if cnt==125:
                cnt=0
                X_cp.append(X)
                B_cp.append(B)
                Rate1_cp.append(Rate1)
                Rate1pct_cp.append(Rate1pct)
                Rate10pct_cp.append(Rate10pct)
                Select_Value_high_cp.append(Select_Value_high)
                Select_Value_low_cp.append(Select_Value_low)
                Select_Value_aver_cp.append(Select_Value_aver)
                X = []
                B = []
                Rate1 = []
                Rate1pct = []
                Rate10pct = []
                Select_Value_high = []
                Select_Value_low = []
                Select_Value_aver = []
                #print(Select_Value_aver)
    file.close()


Slt_v_aver_p_c_np=np.array(Select_Value_aver_cp)
Rate1_np=np.array(Rate1_cp)
X_np = np.array(X_cp)
P_np = np.array(B_cp)


#print(Select_Value_aver_cp)

print(Slt_v_aver_p_c_np.shape)
ax3 = plt.axes(projection='3d')
ax3.set_xlabel('伯努利实验成功率p')
ax3.set_ylabel('单次观察成本c')
ax3.set_zlabel('选择奖励均值')
ax3.plot_surface(X_np,P_np,Slt_v_aver_p_c_np,cmap='rainbow')
plt.savefig("选择奖励均值-c-p.png")
plt.show()