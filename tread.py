from numba import cuda
import numpy as np
import math
from time import time
import numba.cuda.random as cuda_random
from numba.typed import List
@cuda.jit
def gpu_add(rng_states,a, b, result, n):
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if idx < n :
        result[idx][0] = a[idx] + b[idx]
        result[idx][1] = idx

def main():
    n = 20000000
    x = np.arange(n).astype(np.int32)
    y = 2 * x

    # 拷贝数据到设备端
    x_device = cuda.to_device(x)
    y_device = cuda.to_device(y)
    # 在显卡设备上初始化一块用于存放GPU计算结果的空间
    gpu_result = cuda.device_array(shape=(n,2))
    cpu_result = np.empty(shape=(n))

    threads_per_block = 1024
    blocks_per_grid = math.ceil(n / threads_per_block)
    rng_states = cuda_random.create_xoroshiro128p_states(threads_per_block * blocks_per_grid, seed=1)
    start = time()
    gpu_add[blocks_per_grid, threads_per_block](rng_states,x_device, y_device, gpu_result, n)
    cuda.synchronize()
    print("gpu vector add time " + str(time() - start))
    start = time()
    cpu_result = np.add(x, y)
    print("cpu vector add time " + str(time() - start))
    gpu_result=gpu_result.copy_to_host()
    print(gpu_result.size)

if __name__ == "__main__":
    main()
    print(float('-inf'))
    print(float('inf'))