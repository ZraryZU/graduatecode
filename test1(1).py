import torch.nn as nn
import torch.functional as F
import numpy as np
import torch
from torch.utils.data import Dataset


class Model(nn.Module):
    def __init__(self):
        super.__init__()
        self.conv1 = nn.Conv2d(1,20,5)
        self.conv2 = nn.Conv2d(20,20,5)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))

class ImageDataset(Dataset):
    def __init__(self, raw_data)
        self.raw_data = raw_data

    def __len__(self)
        return len(self.raw_data)

    def __getitem__(self, index)
        image, lable = self.raw_data[index]
        return image, lable

# 标量Tensor求导
# 求 f(x) = a*x**2 + b*x + c 的导数
x = torch.tensor(-2.0, requires_grad=True)
a = torch.tensor(1.0)
b = torch.tensor(2.0)
c = torch.tensor(3.0)
y = a*torch.pow(x,2)+b*x+c
y.backward() # backward求得的梯度会存储在自变量x的grad属性中
dy_dx =x.grad
print(dy_dx)






# f(x) = a*x**2 + b*x + c的最小值
x = torch.tensor(8.0, requires_grad=True)  # x需要被求导
z = torch.tensor(0.0, requires_grad=True)  #

x = torch.tensor([[-2.0,-1.0],[0.0,1.0]], requires_grad=True)
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)
gradient=torch.tensor([[1.0,1.0],[1.0,1.0]])
optimizer = torch.optim.SGD(params=[x], lr=0.01)  # SGD为随机梯度下降
print(optimizer)


def f(x):
    result = a * torch.pow(x, 2) + b * x + c
    return (result)


for i in range(500):
    optimizer.zero_grad()  # 将模型的参数初始化为0
    y = f(x)
    y.backward(gradient=gradient)  # 反向传播计算梯度
    optimizer.step()  # 更新所有的参数
print("y=", y.data, ";", "x=", x.data)
