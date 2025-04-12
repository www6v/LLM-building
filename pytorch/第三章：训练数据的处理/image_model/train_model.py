import torchvision
import torch 
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,datasets 
import torchkeras
import pandas as pd
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import numpy as np
def train_step(model,features,labels):
 

    # 训练模式，dropout层发生作用
    model.train()

    # 梯度清零
    model.optimizer.zero_grad()

    # 正向传播求损失
    predictions = model(features)
 
    loss = model.loss_func(predictions,labels)
 
    metric = model.metric_func(predictions,labels)

    # 反向传播求梯度
    loss.backward()
    model.optimizer.step()

    return loss.item(),metric.item()

def valid_step(model,features,labels):

    # 预测模式，dropout层不发生作用
    model.eval()

    predictions = model(features)
    loss = model.loss_func(predictions,labels)
    metric = model.metric_func(predictions,labels)
    return loss.item(), metric.item()

def train_model(model,epochs,dl_train,dl_valid,log_step_freq):

    metric_name = model.metric_name
    dfhistory = pd.DataFrame(columns = ["epoch","loss",metric_name,"val_loss","val_"+metric_name]) 
    print("Start Training...")
    for epoch in range(1,epochs+1):  

        # 1，训练循环-------------------------------------------------
        loss_sum = 0.0
        metric_sum = 0.0
        step = 1

        for step, (features,labels) in enumerate(dl_train,1):

            loss,metric = train_step(model,features,labels)
 
            # 打印batch级别日志
            loss_sum += loss
            metric_sum += metric
            if step%log_step_freq == 0:   
                print(("[step = %d] loss: %.3f, "+metric_name+": %.3f") %
                      (step, loss_sum/step, metric_sum/step))

        # 2，验证循环-------------------------------------------------
        val_loss_sum = 0.0
        val_metric_sum = 0.0
        val_step = 1

        for val_step, (features,labels) in enumerate(dl_valid,1):

            val_loss,val_metric = valid_step(model,features,labels)

            val_loss_sum += val_loss
            val_metric_sum += val_metric

        # 3，记录日志-------------------------------------------------
        info = (epoch, loss_sum/step, metric_sum/step, 
                val_loss_sum/val_step, val_metric_sum/val_step)
        dfhistory.loc[epoch-1] = info

        # 打印epoch级别日志
        print(("\nEPOCH = %d, loss = %.3f,"+ metric_name + \
              "  = %.3f, val_loss = %.3f, "+"val_"+ metric_name+" = %.3f") 
              %info)

    print('Finished Training...')

    return dfhistory


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size = 3)
        self.pool = nn.MaxPool2d(kernel_size = 2,stride = 2)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size = 5)
        self.dropout = nn.Dropout2d(p = 0.1)
        self.adaptive_pool = nn.AdaptiveMaxPool2d((1,1))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64,32)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(32,10)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

def get_data_method():
    #直接读取图像文件夹，出来的数据格式就是dataset
    #对放图像的文件夹 有特定要求
    #类别id做子文件夹
    #target_transform对目标做转化
    #transform对输入特征做转化
    train_set = datasets.ImageFolder("images/train/",
             transform = transform,target_transform= lambda t:torch.tensor(t))
    test_set = datasets.ImageFolder("images/test/",
            transform = transform,target_transform= lambda t:torch.tensor(t))
 
    return train_set,test_set


#transform=transforms.Compose([transforms.ToTensor()])
#把数值特征转化成张量
t0=transforms.ToTensor()
#归一化大小
t1=transforms.Resize((224, 224))
#对数值进行归一化
t2=transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
#随机水平旋转
t3=torchvision.transforms.RandomHorizontalFlip() 
t4=torchvision.transforms.RandomVerticalFlip()
t5=torchvision.transforms.RandomRotation() #随机旋转+-10°
#类似于管道，按执行顺序放入transformer
transform=transforms.Compose([t0,t1,t2])
train_set,test_set=get_data_method()


model= Net()
# print(model) 
# torchkeras.summary(model,input_shape= (3,32,32))
model.optimizer = torch.optim.SGD(model.parameters(),lr = 0.01)
model.loss_func = F.cross_entropy
 
model.metric_func = lambda y_pred,y_true: accuracy_score(y_true.data.numpy(),np.argmax(y_pred.tolist(),axis=-1))
model.metric_name = "accuracy"




dl_train = DataLoader(train_set,batch_size =1,shuffle = True)
dl_valid = DataLoader(test_set,batch_size = 1,shuffle = True)
epochs = 20
dfhistory = train_model(model,epochs,dl_train,dl_valid,log_step_freq = 50)