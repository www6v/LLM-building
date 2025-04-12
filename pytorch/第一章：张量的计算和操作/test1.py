### 《常见张量计算》###
### 如何构建张量 ###

import numpy as np
import torch 

print("--------")
#张量的生成，直接从python的list中生成
x=torch.tensor([1,2,3])
print (x)
print (x.dtype)
# #半精度，拿两个字节 16位 表示一个浮点数
x=torch.tensor([1,2,3]).to(torch.float16)
print (x.dtype)
# #全精度，那四个字节 32位表示一个浮点数
x=torch.tensor([1,2,3]).to(torch.float32)
print (x.dtype)

print("--------")
### 存储方式
x=torch.tensor([[1,2,3],[4,5,6]])
print (x)
print (x.storage())

#两种存放方式  gpu和cpu
x=torch.tensor([1,2,3])
y=torch.tensor([4,5,6])
# #把张量放进GPU
# x=x.to("cuda:0")
# y=y.to("cuda:0")
z=x+y
print (z)
# # print (x)
# # x=x.to("cuda")
x=x.to("cpu")
z=x+y
print (z)

print("--------")
#和 numpy的相互转换
#numpy->tensor
arr = np.zeros(3)
tensor = torch.from_numpy(arr)
print("before add 1:")
print(arr)
print(tensor)

print("\nafter add 1:")
np.add(arr,1, out = arr) #给 arr增加1，tensor也随之改变
print(arr)
print(tensor)

#tensor->numpy
# tensor = torch.zeros(3)
# arr = tensor.numpy()
# print("before add 1:")
# print(tensor)
# print(arr)
# print("\nafter add 1:")
# #使用带下划线的方法表示计算结果会返回给调用 张量
# tensor.add_(1) #给 tensor增加1，arr也随之改变 
# #或： torch.add(tensor,1,out = tensor)
# print(tensor)
# print(arr)

print("--------")
# 可以用clone() 方法拷贝张量，中断这种关联
tensor = torch.zeros(3)
# #使用clone方法拷贝张量, 拷贝后的张量和原始张量内存独立
arr = tensor.clone().numpy() # 也可以使用tensor.data.numpy()
print("before add 1:")
print(tensor)
print(arr)
print("\nafter add 1:")
# #使用 带下划线的方法表示计算结果会返回给调用 张量
tensor.add_(1) #给 tensor增加1，arr不再随之改变
print(tensor)
print(arr)

# a = torch.arange(5)  # 初始化张量 a 为 [0, 1, 2, 3, 4]
# b = a[2:]            # 截取张量a的部分值并赋值给b，b其实只是改变了a对数据的索引方式
# print('a:', a)
# print('b:', b)
# print('ptr of storage of a:', a.storage().data_ptr())  # 打印a的存储区地址
# print('ptr of storage of b:', b.storage().data_ptr())  # 打印b的存储区地址,可以发现两者是共用存储区

# # print('==================================================================')

# b[1] = 0    # 修改b中索引为1，即a中索引为3的数据为0
# print('a:', a)
# print('b:', b)
# print('ptr of storage of a:', a.storage().data_ptr())  # 打印a的存储区地址,可以发现a的相应位置的值也跟着改变，说明两者是共用存储区
# print('ptr of storage of b:', b.storage().data_ptr())  # 打印b的存储区地址

print("--------")
#张量的stride
a = torch.tensor([[1,2,3],
                  [4,5,6]])  # 初始化张量 a
b = torch.tensor([[1,2],
                  [3,4],
                  [5,6]])   # 初始化张量 b
print('a:', a)
print('stride of a:', a.stride())  # 打印a的stride
print('b:', b)
print('stride of b:', b.stride())  # 打印b的stride





# item方法和tolist方法可以将张量转换成Python数值和数值列表
# scalar = torch.tensor(1.0)
# s = scalar.item()
# print(s)
# print(type(s))
# tensor = torch.rand(2,2)
# t = tensor.tolist()
# print(t)
# print(type(t))

#生成一些固定数值的张量
# x=torch.zeros(2,3)
# print (x)

# x=torch.ones(2,3)
# print (x)
#划一块空间，不给里面初始化
# y=torch.empty(3,4)
# #生成一个尺寸和y一样，但是全0的张量
# x=torch.zeros_like(y)
# print (x)

print("--------")
#均值为0，方差为1的标准正态分布
#神经网络里面的参数初始化
x = torch.randn(3, 4)#n=normal
print('标准正态分布x:',x)
# 0-1的均匀分布
x = torch.rand(3, 4)
print('均匀分布x:',x)


print("--------")
#生成一些特殊的数组
#从 0~15的数组 view把数据按4*4的格式排列
tensor = torch.arange(0,16).view(4,4)
print('origin tensor:\n{}\n'.format(tensor))

# mask = torch.eye(4,dtype=torch.bool)
# print('mask tensor:\n{}\n'.format(mask))
# tensor = tensor.masked_fill(mask,100)
# print('filled tensor:\n{}'.format(tensor))

print("--------")
#一维mask
x=torch.tensor([1,0,-1])
mask = x.ge(0.5)
print('mask:',mask)
y=torch.masked_select(x, mask)
print (y)


 #二维mask
# a = torch.randn(3,4)
# mask = torch.tensor([[1,1,1,0],[1,0,0,0],[0,0,0,0]],dtype=torch.bool)
# b = torch.masked_select(a,mask)
# print('a:',a)
# print('mask:',mask)
# print('b:',b) 

print("--------")
#跟人工智能比较相关的一些操作
x=torch.randn(2,3)
print (x)
y=torch.sigmoid(x)
print ("sigmoid",y)
y=torch.softmax(x,dim=0)
print ("softmax  dim=0",y)
y=torch.softmax(x,dim=1)
print ("softmax  dim=1",y)
#dim=-1 最后一个维度，在这里就和dim=1等价
y=torch.softmax(x,dim=-1)
print ("softmax  dim=-1",y)
y=torch.relu(x)
print ("relu",y)
y=torch.tanh(x)
print ("tanh",y)




print("--------")
#向量计算
x=torch.tensor([1,2,3,4,5],dtype=torch.float16)
y=torch.tensor([6,7,8,9,10],dtype=torch.float16)
#print (x*y)
def cosine_similar(tensor_1, tensor_2):
    #归一化分母 cosin相似度
    normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
    normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
    return (normalized_tensor_1 * normalized_tensor_2).sum(dim=-1)
print (cosine_similar(x,y))
#是神经网络的一层，只是该层没有参数
# print (torch.nn.CosineSimilarity(dim=-1)(x,y))
#是一个纯函数
# print (torch.nn.functional.cosine_similarity(x,y,dim=-1))

print("--------")
#内积和外积
x=torch.tensor([1,2,3,4,5],dtype=torch.float32)
y=torch.tensor([6,7,8,9,10],dtype=torch.float32)

z=torch.dot(x,y)
print ("内积",z)
z=torch.outer(x,y)
print ("外积",z)

print("--------")
#一些基本的矩阵计算
x=torch.tensor([[1,2],
                [3,4]])
y=torch.tensor([[2,2],
                  [3,3]])
#元素对应相乘
print (x*y)
#矩阵相乘
print (torch.matmul(x,y))


#矩阵计算
x= torch.tensor([[2,1,3],
                 [6,5,4]])
y=torch.sum(x)
print ("相加",y)
y=torch.pow(x,3)
print ("pow",y)
y=torch.argmax(x)
#返回index
print ("argmax,全量",y)
y=torch.argmax(x,dim=0)
print ("argmax dim=0",y)
y=torch.argmax(x,dim=1)
print ("argmax dim=1",y)

#矩阵的筛选
# x= torch.tensor([[2,1,3],
#                  [6,5,4]])
# y=x[x>2]
# print (y)

 



