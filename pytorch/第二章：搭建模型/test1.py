#用最原始的方式搭建，自己实现梯度下降法
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt 
import torch
from torch import nn
def show_data(X,Y):
    plt.figure(figsize = (12,5))
    ax1 = plt.subplot(121)
    ax1.scatter(X[:,0].numpy(),Y[:,0].numpy(), c = "b",label = "samples")
    ax1.legend()
    plt.xlabel("x1")
    plt.ylabel("y",rotation = 0)

    ax2 = plt.subplot(122)
    ax2.scatter(X[:,1].numpy(),Y[:,0].numpy(), c = "g",label = "samples")
    ax2.legend()
    plt.xlabel("x2")
    plt.ylabel("y",rotation = 0)
    plt.show()
# 构建数据管道迭代器
def data_iter(features, labels, batch_size=8):
    num_examples = len(features)
    indices = list(range(num_examples))
    np.random.shuffle(indices)  #样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        indexs = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])
        yield  features.index_select(0, indexs), labels.index_select(0, indexs)

def train_step(model, features, labels):

    predictions = model.forward(features)
    loss = model.loss_func(predictions,labels)

    # 反向传播求梯度
    loss.backward()
    with torch.no_grad():
        # 梯度下降法更新参数
        model.w -= 0.001*model.w.grad
        model.b -= 0.001*model.b.grad
        # 梯度清零
        model.w.grad.zero_()
        model.b.grad.zero_()
    return loss
def train_model(model,X,Y,epochs,batchsize=512):
    for epoch in range(1,epochs+1):
        for features, labels in data_iter(X,Y,batchsize):
            loss = train_step(model,features,labels)
        if epoch%200==0:
            print("epoch =",epoch,"loss = ",loss.item())
            # print("model.w =",model.w.data)
            # print("model.b =",model.b.data)


class LinearRegression: 

    def __init__(self):
        self.w = torch.randn_like(w0,requires_grad=True)
        self.b = torch.zeros_like(b0,requires_grad=True)

    #正向传播
    def forward(self,x): 
        #@内积求和
        return x@self.w + self.b

    # 损失函数
    def loss_func(self,y_pred,y_true):  
        return torch.mean((y_pred - y_true)**2)

#样本数量
n = 400
# 生成测试用数据集
X = 10*torch.rand([n,2])-5.0  #torch.rand是均匀分布 
#W0和b0是产生数据的参数，和模型对应的参数同尺寸
w0 = torch.tensor([[2.0],[-3.0]])
b0 = torch.tensor([[10.0]])
Y = X@w0 + b0 + torch.normal( 0.0,2.0,size = [n,1])  # @表示矩阵乘法,增加正态扰动
#show_data(X,Y)
#X,Y是人造出来的
model = LinearRegression()
train_model(model,X,Y,epochs = 5000)

 