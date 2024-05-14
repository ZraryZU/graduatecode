import torch
from matplotlib import pyplot as plt

from torch.autograd import Variable
batch_n = 64#一个批次输入数据的数量
hidden_layer = 100
input_data = 1000#每个数据的特征为1000
output_data = 10
lr=0.000001
class Model(torch.nn.Module):#完成类继承的操作
    def __init__(self):
        super(Model,self).__init__()#类的初始化
        
    def forward(self,input,w1,w2):
        x = torch.mm(input,w1)
        x = torch.clamp(x,min = 0)
        x = torch.mm(x,w2)
        return x
    
    def backward(self):
        pass
model = Model()
x = Variable(torch.randn(batch_n,input_data),requires_grad=False)
y = Variable(torch.randn(batch_n,output_data),requires_grad=False)
#用Variable对Tensor数据类型变量进行封装的操作。requires_grad如果是F，表示该变量在进行自动梯度计算的过程中不会保留梯度值。
w1 = Variable(torch.randn(input_data,hidden_layer),requires_grad=True)
w2 = Variable(torch.randn(hidden_layer,output_data),requires_grad=True)
 
epoch_n=300
 
for epoch in range(epoch_n):
    y_pred = model(x,w1,w2)
   
    loss = (y_pred-y).pow(2).sum()
    print("epoch:{},loss:{:.4f}".format(epoch,loss.data))
    loss.backward()
    w1.data -= lr*w1.grad.data
    w2.data -= lr*w2.grad.data
 
    w1.grad.data.zero_()
    w2.grad.data.zero_()




 
batch_n = 100#一个批次输入数据的数量
hidden_layer = 100
input_data = 1000#每个数据的特征为1000
output_data = 10
 
x = Variable(torch.randn(batch_n,input_data),requires_grad=False)
y = Variable(torch.randn(batch_n,output_data),requires_grad=False)
#用Variable对Tensor数据类型变量进行封装的操作。requires_grad如果是F，表示该变量在进行自动梯度计算的过程中不会保留梯度值。
 
models = torch.nn.Sequential(
    torch.nn.Linear(input_data,hidden_layer),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_layer,output_data)
)
#torch.nn.Sequential括号内就是我们搭建的神经网络模型的具体结构，Linear完成从隐藏层到输出层的线性变换，再用ReLU激活函数激活
#torch.nn.Sequential类是torch.nn中的一种序列容器，通过在容器中嵌套各种实现神经网络模型的搭建，
#最主要的是，参数会按照我们定义好的序列自动传递下去。
 
# loss_fn = torch.nn.MSELoss()
# x = Variable(torch.randn(100,100))
# y = Variable(torch.randn(100,100))
# loss = loss_fn(x,y)
 
epoch_n=100
lr=1e-4
loss_fn = torch.nn.MSELoss()
 
optimzer = torch.optim.Adam(models.parameters(),lr=lr)
#使用torch.optim.Adam类作为我们模型参数的优化函数，这里输入的是：被优化的参数和学习率的初始值。
#因为我们需要优化的是模型中的全部参数，所以传递的参数是models.parameters()
 
#进行，模型训练的代码如下：
for epoch in range(epoch_n):
    y_pred = models(x)
    loss = loss_fn(y_pred,y)
    print("Epoch:{},Loss:{:.4f}".format(epoch,loss.data))
    optimzer.zero_grad()#将模型参数的梯度归0
    
    loss.backward()
    optimzer.step()#使用计算得到的梯度值对各个节点的参数进行梯度更新。



# Image Classification
import torch
from torchvision.transforms import v2

H, W = 32, 32
img = torch.randint(0, 256, size=(3, H, W), dtype=torch.uint8)

transforms = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
img = transforms(img)


ix = 24150
plt.imshow(tr_images[ix], cmap='gray')
plt.title(fmnist.classes[tr_targets[ix]])
plt.show()
