### 《pytorch求导》 ###


#深度理解计算图
#叶子节点和非叶子节点
import torch

print("--------")
#标量求导
x = torch.tensor(-2.0,requires_grad = True) # x需要被求导
b = torch.tensor(-2.0)
c = torch.tensor(1.0)
d=torch.pow(x,2)
e=b*x
y = d + e + c
print ("求导前",x.grad)
y.backward(retain_graph=True)
print ("1次求导后x",x.grad)
x.grad.zero_()
y.backward(retain_graph=True)
print ("2次求导后x",x.grad)
x.grad.zero_()
y.backward(retain_graph=True)
print ("3次求导后x",x.grad)
x.grad.zero_()
# y.backward(retain_graph=True)
# print (x.grad)
# y.backward()
# print (x.grad)

# #向量/矩阵的求导
x = torch.tensor([[3.0,4.0],[1.0,2.0]],requires_grad = True) 
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)
#y也是一个矩阵
y = a*torch.pow(x,2) + b*x + c
#方法1
#把y变成了一个标量
y1=y.mean()
y1.backward(retain_graph=True)
print ("把y变成了标量",x.grad)
x.grad.zero_()
#方法2
gradient = torch.tensor([[1.0,1.0],[1.0,1.0]])/4
#用gradient对各个元素进行加权求和
y.backward(gradient = gradient)
print ("把y加权求和",x.grad)
 








# 需要计算梯度-requires_grad=True
w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([2.])
# 前向传播
a = torch.add(w, x)     
#a.retain_grad()
b = torch.add(w, 1)
y = torch.mul(a, b)
# 反向传播-自动求导
y.backward(retain_graph=True)
print (w._grad)
w.grad.zero_()
y.backward()
print (w._grad)
 