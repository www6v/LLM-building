# 从modelscope上下载模型
from modelscope.hub.snapshot_download import snapshot_download
from modelscope import MsDataset

###### model

# model_dir = snapshot_download('baichuan-inc/baichuan-7B', cache_dir='./model', revision='master')
# model_dir = snapshot_download('ZhipuAI/chatglm2-6b', cache_dir='./model', revision='master')
# model_dir = snapshot_download('xrunda/m3e-base', cache_dir='./model', revision='master')


model_dir = snapshot_download('ZhipuAI/chatglm2-6b-int4', cache_dir='./model', revision='master')
model_dir = snapshot_download('LLM-Research/Meta-Llama-3-8B-Instruct', cache_dir='./model', revision='master')


model_dir = snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Llama-8B', cache_dir='./model', revision='master')
model_dir = snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', cache_dir='./model', revision='master')


##### dataset 加载数据集
dataset_dir = snapshot_download('', cache_dir='./dataset', revision='master')

### 好像有问题
#  modelscope download --dataset 'FreedomIntelligence/medical-o1-reasoning-SFT'  --local_dir './dataset'
#  modelscope download --dataset 'FreedomIntelligence/medical-o1-reasoning-SFT' --include 'data/train-000*' --cache_dir './cache_dir'

### 这个没问题
# modelscope download --dataset 'FreedomIntelligence/medical-o1-reasoning-SFT'  --cache_dir './cache_dir'
# modelscope download --dataset 'openai-mirror/gsm8k'  --cache_dir './gsm8k'


### 好像有问题
ds = MsDataset.load('FreedomIntelligence/medical-o1-reasoning-SFT', cache_dir='./dataset', split='train')
ds = MsDataset.load('FreedomIntelligence/medical-o1-reasoning-SFT', cache_dir='./dataset')