import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,TensorDataset
#最标准的做法，不用tensor作为参数，用Modlue，进一步封装
class DNNModel(nn.Module):
    def __init__(self):
        super(DNNModel, self).__init__()
        #Linear作为基本层，参数是由Parameter构成
        #但是DNN可以用上层作为参数
        #nn.Linear本身就是一个模型，用模型本身当参数
        #直接使用比较经典的层，降低变成难度
        self.fc1 =nn.Linear(2,4)
        self.fc2 = nn.Linear(4,8) 
        self.fc3 = nn.Linear(8,1)

    # 正向传播
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = nn.Sigmoid()(self.fc3(x))
        return y

    # 损失函数
    #损失函数也不使用toroch的基础函数，使用模型层对应的损失函数
    def loss_func(self,y_pred,y_true):
        return nn.BCELoss()(y_pred,y_true)

    # 评估函数(准确率)
    def metric_func(self,y_pred,y_true):
        y_pred = torch.where(y_pred>0.5,torch.ones_like(y_pred,dtype = torch.float32),
                          torch.zeros_like(y_pred,dtype = torch.float32))
        acc = torch.mean(1-torch.abs(y_true-y_pred))
        return acc

    # 优化器
    @property
    def optimizer(self):
        #把常见的梯度下降法给封装好了
        return torch.optim.Adam(self.parameters(),lr = 0.001)
def train_step(model, features, labels):

    # 正向传播求损失
    predictions = model(features)
    loss = model.loss_func(predictions,labels)
    metric = model.metric_func(predictions,labels)

    #下面是完成梯度下降法的流程
    # 反向传播求梯度
    loss.backward()
    # 更新模型参数
    model.optimizer.step()
    model.optimizer.zero_grad()

    return loss.item(),metric.item()
def train_model(model,epochs):
    for epoch in range(1,epochs+1):
        loss_list,metric_list = [],[]
        for features, labels in dl:
            lossi,metrici = train_step(model,features,labels)
            loss_list.append(lossi)
            metric_list.append(metrici)
        loss = np.mean(loss_list)
        metric = np.mean(metric_list)

        if epoch%100==0:
            print("epoch =",epoch,"loss = ",loss,"metric = ",metric)
#正负样本数量
n_positive,n_negative = 2000,2000

#生成正样本, 小圆环分布
r_p = 5.0 + torch.normal(0.0,1.0,size = [n_positive,1]) 
theta_p = 2*np.pi*torch.rand([n_positive,1])
Xp = torch.cat([r_p*torch.cos(theta_p),r_p*torch.sin(theta_p)],axis = 1)
Yp = torch.ones_like(r_p)

#生成负样本, 大圆环分布
r_n = 8.0 + torch.normal(0.0,1.0,size = [n_negative,1]) 
theta_n = 2*np.pi*torch.rand([n_negative,1])
Xn = torch.cat([r_n*torch.cos(theta_n),r_n*torch.sin(theta_n)],axis = 1)
Yn = torch.zeros_like(r_n)

#汇总样本
X = torch.cat([Xp,Xn],axis = 0)
Y = torch.cat([Yp,Yn],axis = 0)
ds = TensorDataset(X,Y)
dl = DataLoader(ds,batch_size = 10,shuffle=True)
model = DNNModel()
train_model(model,epochs = 300)