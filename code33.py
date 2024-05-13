import torch
from torch.autograd import Variable

batch_n = 64  # 一个批次输入数据的数量
hidden_layer = 100
input_data = 1000  # 每个数据的特征为1000
output_data = 10


class Model(torch.nn.Module):  # 完成类继承的操作
    def __init__(self):
        super(Model, self).__init__()  # 类的初始化

    def forward(self, input, w1, w2):
        x = torch.mm(input, w1)
        x = torch.clamp(x, min=0)
        x = torch.mm(x, w2)
        return x

    def backward(self):
        pass


model = Model()
x = Variable(torch.randn(batch_n, input_data), requires_grad=False)
y = Variable(torch.randn(batch_n, output_data), requires_grad=False)
# 用Variable对Tensor数据类型变量进行封装的操作。requires_grad如果是F，表示该变量在进行自动梯度计算的过程中不会保留梯度值。
w1 = Variable(torch.randn(input_data, hidden_layer), requires_grad=True)
w2 = Variable(torch.randn(hidden_layer, output_data), requires_grad=True)

epoch_n = 30

for epoch in range(epoch_n):
    y_pred = model(x, w1, w2)

    loss = (y_pred - y).pow(2).sum()
    print("epoch:{},loss:{:.4f}".format(epoch, loss.data))
    loss.backward()
    w1.data -= lr * w1.grad.data
    w2.data -= lr * w2.grad.data

    w1.grad.data.zero_()
    w2.grad.data.zero_()
