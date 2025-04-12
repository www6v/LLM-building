import torch.nn as nn
import torch
from torch.utils.data import Dataset,DataLoader,TensorDataset
nn.BatchNorm1d
class My_loss(nn.Module):
    def __init__(self):
        super().__init__()     
    def forward(self, x, y):
        eps=1e-8     
        pos_num=y.sum().float()
        neg_num=len(y)-pos_num
        pos_weight=neg_num/(pos_num+neg_num)
        neg_weight=pos_weight/(pos_num+neg_num)
        d=-pos_weight*y*torch.log2(x+eps)-neg_weight*(1-y)*torch.log2(1-x+eps)
        d=torch.mean(d)
        return d

# 直接定义函数 ， 不需要维护参数，梯度等信息
# 注意所有的数学操作需要使用tensor完成。
def my_mse_loss(x, y):
    return torch.mean(torch.pow((x - y), 2))
net= nn.Sequential(
    nn.Linear(2,1),nn.Sigmoid()
)
#构建数据
n = 400
# 生成测试用数据集
X = 10*torch.rand([n,2])-5.0  #torch.rand是均匀分布 
w0 = torch.tensor([[2.0],[-3.0]])
b0 = torch.tensor([[10.0]])
Y = X@w0 + b0 + torch.normal( 0.0,2.0,size = [n,1])  # @表示矩阵乘法,增加正态扰动
def f(x,*y): 
    return 1 if x>0 else 0
     
Y.map_(Y,f)
ds = TensorDataset(X,Y)
dl = DataLoader(ds,batch_size = 1000,shuffle=True)
epoch=1000
#my_loss=my_mse_loss
my_loss=My_loss()
optimizer=torch.optim.Adam(net.parameters(),lr = 0.001)
for _ in range(0,epoch):
    for x,target in dl:
        y=net(x)
        loss=my_loss(y,target)
        loss.backward()
        print (loss)
        # 更新模型参数
        optimizer.step()
        optimizer.zero_grad()
    

