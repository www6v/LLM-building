import torch
import string,re
import torchtext
import torchkeras
torch.random.seed()
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
def accuracy(y_pred,y_true):
    y_pred = torch.where(y_pred>0.5,torch.ones_like(y_pred,dtype = torch.float32),
                      torch.zeros_like(y_pred,dtype = torch.float32))
    acc = torch.mean(1-torch.abs(y_true-y_pred))
    return acc
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        #设置padding_idx参数后将在训练过程中将填充的token始终赋值为0向量
        self.embedding = nn.Embedding(num_embeddings = MAX_WORDS,embedding_dim = 3)
        self.conv = nn.Sequential()
        self.conv.add_module("conv_1",nn.Conv1d(in_channels = 3,out_channels = 16,kernel_size = 5))
        self.conv.add_module("pool_1",nn.MaxPool1d(kernel_size = 2))
        self.conv.add_module("relu_1",nn.ReLU())
        self.conv.add_module("conv_2",nn.Conv1d(in_channels = 16,out_channels = 128,kernel_size = 2))
        self.conv.add_module("pool_2",nn.MaxPool1d(kernel_size = 2))
        self.conv.add_module("relu_2",nn.ReLU())

        self.dense = nn.Sequential()
        self.dense.add_module("flatten",nn.Flatten())
        self.dense.add_module("linear",nn.Linear(6144,1))
        self.dense.add_module("sigmoid",nn.Sigmoid())

    def forward(self,x):
        x = self.embedding(x).transpose(1,2)
        x = self.conv(x)
        y = self.dense(x)
        return y


class My_data(Dataset):
    def __init__(self, train_data):
        #初始化，
        #把文本转成id
        #并且忽略出现频率较低的单词
        train_data=list(train_data)
        texts=[s.text for s in train_data]
        labels=[s.label for s in train_data]
        word_count={"__unknow__":0}
        #统计单词的数量
        for doc in texts:
            for word in doc:
                word_count[word]=word_count.get(word,0)+1
        #获取出现次数为前MAX_WORDS的单词
        legal_word=sorted(word_count.items(),key=lambda s:s[1],reverse=True)[0:MAX_WORDS-1]
        legal_word=set([s[0] for  s in legal_word])
        word_count=dict([ [word,i+1] for i,word in enumerate(list(legal_word))])
        #把单词转化成id
        self.texts=[ [ word_count[word] for word in doc if word in legal_word][0:MAX_LEN] for doc in texts]
        #长度都统一成MAX_LEN
        #过长的截断
        #过短的补0
        self.texts=torch.tensor([doc+(MAX_LEN-len(doc))*[0] for doc in self.texts])
        labels=[[float(s)]  for s in labels]
        self.labels=torch.tensor(labels)
    #获取数据长度的方法
    def __len__(self):
        return len(self.labels)
    #根据索引id获取数据的方法
    def __getitem__(self, item):
        return self.texts[item], self.labels[item]


MAX_WORDS = 50000  # 仅考虑最高频的10000个词
MAX_LEN = 200  # 每个样本保留200个词的长度
BATCH_SIZE = 20 


#分词方法
#按空格分词
tokenizer = lambda x:re.sub('[%s]'%string.punctuation,"",x.replace("\\","")).split(" ")

#分词器，按空格分词
TEXT = torchtext.data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=MAX_LEN)
LABEL = torchtext.data.Field(sequential=False, use_vocab=False)

#2,构建表格型dataset
#torchtext.data.TabularDataset可读取csv,tsv,json等格式
train_data, test_data = torchtext.data.TabularDataset.splits(
        path='./data', train='train_data.tsv',test='test_data.tsv', format='tsv',
        fields=[("id",LABEL),('label', LABEL), ('text', TEXT)],skip_header = True)
train_data=My_data(train_data)
#DataLoader除了dataset格式外 ，也可以支持自定义对象My_data
#自定义对象My_data，需要实现特定的方法
dl_train = DataLoader(train_data,shuffle = True, batch_size = 128)
model = Net()
model = torchkeras.Model(model)
model.compile(loss_func = nn.BCELoss(),optimizer= torch.optim.Adagrad(model.parameters(),lr = 0.02),metrics_dict={"accuracy":accuracy})

dfhistory = model.fit(20,dl_train,log_step_freq= 200)
 
