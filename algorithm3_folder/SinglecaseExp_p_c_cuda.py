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
from numba import cuda
import numba.cuda.random as cuda_random
from numba.typed import List
from numba import njit
@cuda.jit
def random_binomial(x):
    rd=random.random()
    if(rd<=x):
        return 1
    else:
        return 0


@cuda.jit
def gpu_caculate(rng_states, p, c_list, REPEAT_COUNT, item_random_low,item_random_high,RESULTS,Treadsum):
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if idx < Treadsum:
        c=c_list[idx]
        miu=1
        #RESULT_DIC = []
        PERCENT1_COUNT = 0
        PERCENT10_COUNT = 0
        TARGET_COUNT = 0
        S_V_H = 9999999
        S_V_L = -9999999
        S_V_A = 0
        for k in range(REPEAT_COUNT):
            itemss = int(random.random() *(item_random_high-item_random_low)) + item_random_low
            X1 = np.empty(shape=(itemss))
            print(X1)
            res = List()
            re = 0
            B = List()
            for i in range(itemss):
                random_number = math.log(1/(1-random.random()))
                random_bonulli = random_binomial(p)
                X1.append(random_number)
                #X1[i]=random_number
                B.append(random_bonulli)
                #B[i] = random_bonulli
                re = (re + X1[i]) * B[i]
                res.append(re - (i + 1) * c)
            #raw_res = copy.deepcopy(res)
            res.sort()
            #Stemp = 0
            if (p * miu >= c):
                Stemp = miu / (1 - p) * math.log(p * miu / c)
            else:
                Stemp = 0
            #Maxpos = itemss - 1
            Maxval = X1[itemss - 1] - c * itemss
            current_val = 0
            flag = False
            for i in range(itemss):
                current_val = B[i] * (current_val + X1[i])
                if current_val >= Stemp:
                    Maxval = current_val - (i + 1) * c
                    flag = True
                    break
            if flag == False:
                Maxval = current_val - itemss * c
            Range = 0
            for j in range(itemss):
                if (Maxval == res[j]):
                    Range = j
            #RESULT_DIC.append(Range + 1)
            if Range >= itemss - 1:
                TARGET_COUNT += 1
            if Range >= itemss * 0.99:
                PERCENT1_COUNT += 1
            if Range >= itemss * 0.9:
                PERCENT10_COUNT += 1
            S_V_A += Maxval
            S_V_H = max(S_V_H, Maxval)
            S_V_L = min(S_V_L, Maxval)
        S_V_A = S_V_A / REPEAT_COUNT
        '''
        result=[]
        result.append(c)
        result.append(p)
        result.append(TARGET_COUNT / REPEAT_COUNT)
        result.append(PERCENT1_COUNT / REPEAT_COUNT)
        result.append(PERCENT10_COUNT / REPEAT_COUNT)
        result.append(S_V_H)
        result.append(S_V_L)
        result.append(S_V_A)
        RESULTS.append(result)
        '''
        RESULTS[idx][0] = c
        RESULTS[idx][1] = p
        RESULTS[idx][2] = TARGET_COUNT / REPEAT_COUNT
        RESULTS[idx][3] = PERCENT1_COUNT / REPEAT_COUNT
        RESULTS[idx][4] = PERCENT10_COUNT / REPEAT_COUNT
        RESULTS[idx][5] = S_V_H
        RESULTS[idx][6] = S_V_L
        RESULTS[idx][7] = S_V_A


def main():
    # my Data
    c = 0.004
    p = 0.1
    miu = 1
    P_LOW = 1
    P_HIGH = 10000
    P_STEP = 200
    C_LOW = 1
    C_HIGH = 10000
    C_STEP = 200
    REPEAT_COUNT = 300
    item_random_low=100
    item_random_high=500
    blocks_per_grid=1024


    # 拷贝数据到设备端
    # my 拷贝数据到设备端
    REPEAT_COUNT_device = cuda.to_device(REPEAT_COUNT)
    item_random_low_device = cuda.to_device(item_random_low)
    item_random_high_device = cuda.to_device(item_random_high)

    #my program
    c_list=[]
    for c in range(C_LOW, C_HIGH, C_STEP):
        c = c / C_HIGH
        c_list.append(c)
    #my host-device caculate
    c_list_device=cuda.to_device(c_list)
    header = ['OBSERVE_COST', 'BERNOULLI', 'Rate1', 'Rate1pct', 'Rate10pct', 'Selet_Value_high', 'Select_Value_low',
              'Select_Value_aver']
    start = time()
    print("caculating start")
    with open('algorithm3_exp_data.csv', mode='w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

        for p in range(P_LOW, P_HIGH, P_STEP):
            p = p / P_HIGH
            p_device = cuda.to_device(p)
            threads_per_block = math.ceil(len(c_list) / blocks_per_grid)
            # 在显卡设备上初始化一块用于存放GPU计算结果的空间
            # myRESULTS
            gpu_RESULTS = cuda.device_array(shape=(len(c_list), 8))

            rng_states = cuda_random.create_xoroshiro128p_states(threads_per_block * blocks_per_grid, seed=1)

            gpu_caculate[blocks_per_grid, threads_per_block](rng_states, p, c_list_device, REPEAT_COUNT,
                                                             item_random_low, item_random_high,
                                                             gpu_RESULTS, len(c_list))
            result_c = gpu_RESULTS.copy_to_host()
            writer.writerows(result_c)
            print("caculating has finished "+str(p))
    file.close()
    print("caculating finished. Time used "+ str(time() - start))

if __name__ == "__main__":
    main()



