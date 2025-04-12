### 《张量的几何操作》 ###
### 张量的几何操作 ###

import torch

print("--------")
A=torch.arange(0,16).view(2,8)
B=torch.arange(16,32).view(2,8)
print ("A",A)
print ("B",B)
#2维数组 dim=0,1
#3维数组 dim=0,1,2
#dim=-1 表示最后一个维度
C= torch.cat([A, B], dim=0)
print ("C",C)
D = torch.stack([A, B], dim=0)
print ("D",D)

print("--------")
#现有张量沿着值为1的维度扩展到新的维度n,数据重复n遍
a=torch.tensor([
    [[1,2,3],
     [4,5,6]]
    ])
#[1,2,3]
print (a.size())
print (a)
#仅限于size=1的维度
a=a.expand(2,2,3)
print (a.size())
print (a)

print("--------")
#改变张量维度
a=torch.arange(9).reshape(3, 3)
print ("a",a)
print ("a stride",a.stride())
#把原有的维度0,1 变换成1,0
#用了转置，但是并不改变内存，为了b能正确索引，最终会导致b的stride异常
b=a.permute(1,0)
print("b",b)
#张量的索引方式
print ("stride b",b.stride())
#张量是否连续，视图索引和内存索引是否一致
print (b.is_contiguous())
#强制转化成一致
c=b.contiguous()
print (c.is_contiguous())
print ("stride C",c.stride())
print('ptr of storage of a:', b.storage().data_ptr())
print('ptr of storage of b:', b.storage().data_ptr())
print('ptr of storage of c:', c.storage().data_ptr())

print("--------")
a = torch.arange(9).reshape(3, 3)             # 初始化张量a
b = a.permute(1, 0)
print (b)
# print("b reshape",b.reshape(9))
# print("b view",b.view(9))
c=b.contiguous()#重新开辟一块存储空间，
print (c.storage())
print("b view",c.view(9))

a=torch.tensor([[1,2,3],[4,5,6]])
print (a.view(3,2))
c=a.contiguous()
print (a.storage().data_ptr())
print (a.is_contiguous())
print (c.storage().data_ptr())


#模拟图片
# a=torch.arange(2*3*2*1).reshape(2,3,2,1) 
# print (a)
# b=a.permute(0,3,2,1)
# print (b)








#一些基本的算数运算
# x=torch.tensor(2,dtype=torch.float16)
# y=x*3
# print ("mul",y)
# y=torch.log(x)
# print ("log",y)
# y=torch.exp(x)
# print ("exp",y)
# a=torch.tensor(3)
# print ("a*x",a*x)

# x=torch.tensor([1,2,-3,0])
# y=torch.abs(x, out=None)
# print (y)







