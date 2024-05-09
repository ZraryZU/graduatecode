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

X=[]
Rate1=[]
Rate1pct=[]
Rate10pct=[]
Select_Value_high=[]
Select_Value_low=[]
Select_Value_aver=[]

plt.rcParams['font.sans-serif'] = ['SimHei']

with open('algorithm2_exp_data.csv', mode='r', encoding='utf-8',newline='') as file:
    reader = csv.DictReader(file)
    for row in reader:
        X.append(float(row['OBSERVE_COST']))
        Rate1.append(float(row['Rate1']))
        Rate1pct.append(float(row['Rate1pct']))
        Rate10pct.append(float(row['Rate10pct']))
        Select_Value_high.append(float(row['Selet_Value_high']))
        Select_Value_low.append(float(row['Select_Value_low']))
        Select_Value_aver.append(float(row['Select_Value_aver']))
file.close()

plt.xlabel('观察对象数量')
plt.ylabel('命中率')

plt.scatter(X,Rate1,color="blue",s=10,label='最大值命中率')
plt.scatter(X,Rate1pct,color="green",s=10,label='前1%命中率')
plt.scatter(X,Rate10pct,color="red",s=10,label='前10%值命中率')

plt.legend(labels=['最大值命中率','前1%命中率','前10%值命中率'],loc='best')

X_np=np.array(X)
Y_np=np.array(Select_Value_aver)
x_new = np.linspace(X_np.min(), X_np.max(),100)
y_smooth = make_interp_spline(X_np, Y_np)(x_new)
 # 绘制趋势线参数RATE1
m, b = np.polyfit(X_np, Rate1, 1)
plt.plot(X_np, m*X_np + b, '-')
 # 绘制趋势线参数RATE1
m, b = np.polyfit(X_np, Rate1pct, 1)
plt.plot(X_np, m*X_np + b, '-')
 # 绘制趋势线参数RATE1
m, b = np.polyfit(X_np, Rate10pct, 1)
plt.plot(X_np, m*X_np + b, '-')
plt.show()

plt.xlabel('观察对象数量')
plt.ylabel('选中值')
plt.plot(X, Select_Value_high,color="blue",label='选中值上界')
plt.plot(X, Select_Value_low,color="green",label='选中值下界')
plt.plot(X, Select_Value_aver,color="red",label='选中值均值')

plt.legend(labels=['选中值上界','选中值下界','选中值均值'],loc='best')
plt.show()
