### 《爱因斯坦求和》 ###

import torch
 

a=torch.tensor([[1,2],
                [3,4]])
#B为A的i和j互换
#B为A的转置
b=torch.einsum("ij->ji",a)
print (b)

#  爱因斯坦求和
a=torch.tensor([[1,2],
                [3,4]])
#爱因斯坦求和的规则：把输出消失的字母，全部相加
# b=torch.einsum("ij->i",a)
# print (b)
c=torch.einsum("ij->j",a)
print (c)

a=torch.tensor([[1,2],
                [3,4]])
#ii表示两个坐标位置相同 0,0  和 1,1
d=torch.einsum('ii->i', a)
print (d)
#因为输出的字母消失了 对应的是i求和
e=torch.einsum('ii', a)
print (e)

a=torch.tensor([[1,2],
                 [3,4]])
b=torch.tensor([[2,3],
                [5,6]])
z=torch.einsum("ij,kh->ijkh",a,b)
print (z)

x = torch.tensor([[1,1,1],
                  [2,2,2],
                  [5,5,5]])
y = torch.tensor([[0,1,0],
                  [1,1,0],
                  [1,1,1]])
#输入两个数组，角标相同，表示相乘
z1=torch.einsum('ij,jk->ik', x, y)
print ("z1",z1)
#对应元素相乘
z2=torch.einsum('ij,ij->ij', x, y)
print ("z2",z2)


x = torch.tensor([1,2])
y = torch.tensor([4,5,6])
z=torch.einsum('i,j->ij', x, y)
print (z)
x = torch.tensor([1,2,3])
y = torch.tensor([4,5,6])
#求内积 元素两两相乘
z=torch.einsum("i,i->i",x,y)
print (z)




As = torch.randn(3, 2, 5)
Bs = torch.randn(3, 5, 4)
#先把b忽略，ij,jk->ik 矩阵相乘
z=torch.einsum('bij,bjk->bik', As, Bs)
print ("As",As)
print ("Bs",Bs)
print ("z",z)



# A = torch.randn(2, 3, 4, 5)
# torch.einsum('...ij->...ji', A).shape
