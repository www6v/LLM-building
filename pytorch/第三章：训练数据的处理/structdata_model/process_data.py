import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import torch 
from torch import nn 
from torch.utils.data import Dataset,DataLoader,TensorDataset

def preprocessing(dfdata):

    dfresult= pd.DataFrame()

    #Pclass
    #get_dummies 把离散的特征转成onehot编码
    #可穷举的整数类型
    #print (dfdata['Pclass'])
 
    dfPclass = pd.get_dummies(dfdata['Pclass'],dtype=int) 
    #print (dfPclass)
    #给列名前面加个名字，充分不必要的操作
    dfPclass.columns = ['Pclass_' +str(x) for x in dfPclass.columns ]
 
    dfresult = pd.concat([dfresult,dfPclass],axis = 1)

    #Sex
    dfSex = pd.get_dummies(dfdata['Sex'],dtype=int)
 
    dfresult = pd.concat([dfresult,dfSex],axis = 1)

    #Age
    #fillna 出现None数值 填0
    dfresult['Age'] = dfdata['Age'].fillna(0)
    #pd.isna .astype('int32') 如果缺失了，则用0表示，没有缺失 用1表示
    dfresult['Age_null'] = pd.isna(dfdata['Age']).astype('int32')

    #SibSp,Parch,Fare
    dfresult['SibSp'] = dfdata['SibSp']
    dfresult['Parch'] = dfdata['Parch']
    dfresult['Fare'] = dfdata['Fare']

    #Carbin
    dfresult['Cabin'] = dfdata['Cabin'].fillna(0)
    dfresult['Cabin_null'] =  pd.isna(dfdata['Cabin']).astype('int32')

    #Embarked
    dfEmbarked = pd.get_dummies(dfdata['Embarked'],dummy_na=True,dtype=int)
    dfEmbarked.columns = ['Embarked_' + str(x) for x in dfEmbarked.columns]
    dfresult = pd.concat([dfresult,dfEmbarked],axis = 1)
    return dfresult
def get_data(batchsize=128):
    dftrain_raw = pd.read_csv('./titanic/train.csv')
    dftest_raw = pd.read_csv('./titanic/test.csv')
    #处理完特征
    x_train = preprocessing(dftrain_raw).values
    #target的处理
    y_train = dftrain_raw[['Survived']].values
    x_test = preprocessing(dftest_raw).values
    y_test = dftest_raw[['Survived']].values
    
    #print (type(TensorDataset(torch.tensor(x_train).float(),torch.tensor(y_train).float())))
    #torch.tensor(x_train).float() 把数据格式转换成pytorch tensor
    #TensorDataset 把特征x和目标y 打包  类似于python里面的zip
    train_data=TensorDataset(torch.tensor(x_train).float(),torch.tensor(y_train).float())
    test_data=TensorDataset(torch.tensor(x_test).float(),torch.tensor(y_test).float())
    #DataLoader 非常重要
    #shuffle 随机抽取，每次抽batchsize
    #无回放抽取
    dl_train = DataLoader(train_data,shuffle = True, batch_size = batchsize)
    dl_valid = DataLoader(test_data,shuffle = False, batch_size = batchsize)

  
 
    return dl_train,dl_valid
#get_data()




