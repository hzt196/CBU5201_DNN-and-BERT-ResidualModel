import torch
from transformers import BertTokenizer, BertModel
from data import get_dataloaders
from model import ResidualClassifier
from train import train
from test import test

# 配置参数
max_length = 128
batch_size = 16
epochs = 10
learning_rate = 0.0001
model_path = 'trained_modelx.pth'

# 加载预训练的BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('./bert-base-multilingual-cased')
bert_model = BertModel.from_pretrained('./bert-base-multilingual-cased')

# 数据加载
train_loader = get_dataloaders('train.xlsx', tokenizer, bert_model, batch_size, max_length)
test_loader = get_dataloaders('train.xlsx', tokenizer, bert_model, batch_size, max_length)

print("here!!!")

# 获取输入特征的大小
input_size = 768  # 假设 BERT 的输出为 768 维度

# 构建模型
model = ResidualClassifier(input_size)

# 训练并保存模型
model = train(model, train_loader, epochs=epochs, lr=learning_rate, save_path=model_path)

# 加载并测试模型
test(test_loader, input_size=input_size, model_path=model_path)
