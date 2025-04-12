### 《pytorch的损失函数》###
### 对比各类损失函数 ###

#损失函数的输入 有两个  1.预测  2.目标
import torch
#CrossEntropyLoss的两种用法
#5分类的数据 3为batchsize，CrossEntropyLoss对于输入，
#必须要给出概率，需要给出logit，未softmax的概率
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
loss1 = torch.nn.CrossEntropyLoss()(input, target)
print (input)
print (target)
print (loss1)
#常用在对标签分类，需要拟合概率分布
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5).softmax(dim=1)
loss2 = torch.nn.CrossEntropyLoss()(input, target)
print (input)
print (target)
print (loss2)

#CrossEntropyLoss 和 NLLLoss 的差别 
input = torch.randn(3, 10, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(10)
loss1 = torch.nn.CrossEntropyLoss()(input, target)
# #NLLLoss和交叉熵的区别，他的输入是需要进行LogSoftmax
loss2 = torch.nn.NLLLoss()(torch.nn.LogSoftmax()(input), target)
print (loss1)
print (loss2)
 

#CrossEntropyLoss 和 BCELoss 的差别 
#BCELoss的特殊之处
input = torch.randn(3, 2, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(2)
loss1 = torch.nn.CrossEntropyLoss()(input, target)
target2=torch.tensor([[1,0] if s==0 else [0,1]  for s in target],dtype=torch.float)
# #输入的必须是概率
loss2 = torch.nn.BCELoss()(torch.nn.Softmax()(input), target2)
print (loss1)
print (loss2)

#BCEWithLogitsLoss 和 BCELoss 的差别 
input = torch.randn(3, 2, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(2)
target=torch.tensor([[1,0] if s==0 else [0,1]  for s in target],dtype=torch.float)
loss1 = torch.nn.BCEWithLogitsLoss()(input, target)
# #对输入不再是softmax 而是sigmoid
loss2 = torch.nn.BCELoss()(torch.nn.Sigmoid()(input), target)
print (loss1)
print (loss2)



