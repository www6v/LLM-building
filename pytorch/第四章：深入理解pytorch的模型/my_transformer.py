import torch
import torch.nn as nn
from collections import OrderedDict
class SelfAttention(nn.Module):
    def __init__(self,embed_size,heads):
        super(SelfAttention,self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.heads_dim = embed_size // heads
        # assert
        assert (self.heads_dim * heads == embed_size), "Embed size must be divided by num of heads"
        # q k v
        self.v = nn.Linear(self.heads_dim,self.heads_dim)
        self.k = nn.Linear(self.heads_dim,self.heads_dim)
        self.q = nn.Linear(self.heads_dim,self.heads_dim)

        self.fnn = nn.Linear(self.embed_size,self.embed_size)

    def forward(self,value,key,query,mask):
        #N为样本数
        N = value.shape[0]
        v_len,k_len,q_len = value.shape[1],key.shape[1],query.shape[1]

        # 维度切断，每个片段作为一个抽头的向量
        value = value.reshape(N,v_len,self.heads,self.heads_dim)
        key = key.reshape(N,k_len,self.heads,self.heads_dim)
        query = query.reshape(N,q_len,self.heads,self.heads_dim)

        # linear 
        value = self.v(value)
        key = self.k(key)
        query = self.q(query)

        # matmul
        # query shape: N, q_len, heads, heads_dim
        # key shape: N, k_len, heads, heads_dim
        # energy shape: N, heads, q_len, k_len
        energy = torch.einsum("nqhd,nkhd->nhqk",[query,key])

        # scale
        energy = energy / self.embed_size ** (1/2)

        # mask
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-1e20'))

        # softmax
        energy = torch.softmax(energy,dim=-1)
        # matmul & concat 
        # energy shape: N, heads, q_len, k_len
        # value shape: N, v_len, heads, heads_dim
        # output shape: N, q_len, heads, heads_dim
        out = torch.einsum('nhqk,nkhd->nqhd',[energy,value]).reshape(N,q_len,self.heads*self.heads_dim)

        # concat & linear
        out = self.fnn(out)

        return out
class TransformerBlock(nn.Module):
    def __init__(self,embed_size,heads,dropout,forward_expansion):
        super(TransformerBlock,self).__init__()
            # 基本参数
        self.embed_size = embed_size
        self.heads = heads
        # multi-head attention（刚才的SelfAttention）
        self.attention = SelfAttention(
                embed_size=embed_size,
                heads=heads
            )
        # 第一个norm层
        self.norm1 = nn.LayerNorm(embed_size)
        # ffn层，Sequence层序列
        self.ffn = nn.Sequential(OrderedDict([
                ('hidden_layer',nn.Linear(embed_size,forward_expansion*embed_size)),
                ('activation',nn.ReLU()),
                ('dropout',nn.Dropout(dropout)),
                ('output_layer',nn.Linear(forward_expansion*embed_size,embed_size))
        ]))
        # 第二个norm层
        self.norm2 = nn.LayerNorm(embed_size)
        # dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,value,key,query,mask):
        # multi-head attention
        att = self.attention(value,key,query,mask)
        # add & norm 1
        att = self.dropout(self.norm1(att + query))
        # feed forward
        out = self.ffn(att)
        # add & norm 2
        out = self.dropout(self.norm2(out + att))
        return out