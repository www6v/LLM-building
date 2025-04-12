import torch 
from torch import nn
from collections import OrderedDict
#复杂模型，模型套模型
#无法指定层名
# net1 = nn.Sequential(
#     nn.Conv2d(in_channels=3,out_channels=32,kernel_size = 3),
#     nn.MaxPool2d(kernel_size = 2,stride = 2),
#     nn.Conv2d(in_channels=32,out_channels=64,kernel_size = 5),
#     nn.MaxPool2d(kernel_size = 2,stride = 2),
#     nn.Dropout2d(p = 0.1),
#     nn.AdaptiveMaxPool2d((1,1)),
#     nn.Flatten(),
#     nn.Linear(64,32),
#     nn.ReLU(),
#     nn.Linear(32,1),
#     nn.Sigmoid()
# )
# print(net1)
# #可以指定层名
# net2 = nn.Sequential(OrderedDict(
#           [("conv1",nn.Conv2d(in_channels=3,out_channels=32,kernel_size = 3)),
#             ("pool1",nn.MaxPool2d(kernel_size = 2,stride = 2)),
#             ("conv2",nn.Conv2d(in_channels=32,out_channels=64,kernel_size = 5)),
#             ("pool2",nn.MaxPool2d(kernel_size = 2,stride = 2)),
#             ("dropout",nn.Dropout2d(p = 0.1)),
#             ("adaptive_pool",nn.AdaptiveMaxPool2d((1,1))),
#             ("flatten",nn.Flatten()),
#             ("linear1",nn.Linear(64,32)),
#             ("relu",nn.ReLU()),
#             ("linear2",nn.Linear(32,1)),
#             ("sigmoid",nn.Sigmoid())
#           ])
#         )
# print(net2)
#混合使用
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #模型1，模型即参数
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size = 3),
            nn.MaxPool2d(kernel_size = 2,stride = 2),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size = 5),
            nn.MaxPool2d(kernel_size = 2,stride = 2),
            nn.Dropout2d(p = 0.1),
            nn.AdaptiveMaxPool2d((1,1))
        )
        #模型2，模型即参数
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        x = self.conv(x)
        y = self.dense(x)
        return y 
net3 = Net()
#冻结参数
for name,para  in net3.named_parameters():
    if "conv" in name:
        para.requires_grad=False
 


