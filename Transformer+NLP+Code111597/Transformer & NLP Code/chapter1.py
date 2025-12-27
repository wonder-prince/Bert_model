# 例1-1
import torch
import torch.nn as nn
import torch.nn.functional as F
# 定义多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        # 定义查询、键、值的线性变换矩阵
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]  # 批量大小
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # 将输入分为多个头
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # 计算查询和键的点积
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out

# 测试多头注意力机制的实现
embed_size = 64
heads = 8
seq_length = 10
x = torch.rand((3, seq_length, embed_size))  # 模拟输入数据

mask = None
attention_layer = MultiHeadAttention(embed_size, heads)
output = attention_layer(x, x, x, mask)

# 输出结果
print("多头注意力机制输出形状:", output.shape)
print("多头注意力机制输出:", output)


# 例1-2
import torch
import torch.nn as nn
import math

# 定义位置编码类
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.embed_size = embed_size

        # 创建一个位置编码矩阵，大小为 (max_len, embed_size)
        position_encoding = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))

        # 奇偶维度分别使用sin和cos进行编码
        position_encoding[:, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 1::2] = torch.cos(position * div_term)
        
        # 增加batch维度并设置为不可训练
        self.position_encoding = position_encoding.unsqueeze(0).detach()

    def forward(self, x):
        # 将位置编码添加到输入张量中
        x = x * math.sqrt(self.embed_size)  # 缩放
        seq_len = x.size(1)
        x = x + self.position_encoding[:, :seq_len, :].to(x.device)
        return x

# 测试位置编码实现
embed_size = 64
seq_length = 10
x = torch.zeros((3, seq_length, embed_size))  # 输入为零张量

position_encoding = PositionalEncoding(embed_size)
output = position_encoding(x)

print("位置编码后的输出形状:", output.shape)
print("位置编码后的输出:", output)


# 定义残差连接与层归一化模块
class ResidualConnectionLayerNorm(nn.Module):
    def __init__(self, embed_size, dropout=0.1):
        super(ResidualConnectionLayerNorm, self).__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_output):
        # 残差连接与层归一化
        return self.norm(x + self.dropout(sublayer_output))

# 测试残差连接与层归一化模块
residual_layer = ResidualConnectionLayerNorm(embed_size)
sublayer_output = torch.rand((3, seq_length, embed_size))  # 模拟子层输出
residual_output = residual_layer(output, sublayer_output)

# 输出结果
print("残差连接与层归一化输出形状:", residual_output.shape)
print


# 例1-3
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义LSTM模型
class TextLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, num_classes):
        super(TextLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # 嵌入层
        out, (h_n, c_n) = self.lstm(x)  # LSTM层
        out = out[:, -1, :]  # 取最后一个时间步的输出
        out = self.fc(out)  # 全连接层
        return out

# 测试TextLSTM模型
vocab_size = 5000
embed_size = 128
hidden_size = 256
num_layers = 2
num_classes = 2
seq_length = 100

model = TextLSTM(vocab_size, embed_size, hidden_size, num_layers, num_classes)
sample_input = torch.randint(0, vocab_size, (32, seq_length))  # 模拟输入数据
output = model(sample_input)

# 输出模型结果
print("TextLSTM模型输出形状:", output.shape)
print("TextLSTM模型输出:", output)


# 例1-4
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义LSTM模型
class TextLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, num_classes):
        super(TextLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # 嵌入层
        out, (h_n, c_n) = self.lstm(x)  # LSTM层
        out = out[:, -1, :]  # 取最后一个时间步的输出
        out = self.fc(out)  # 全连接层
        return out

# 测试TextLSTM模型
vocab_size = 5000
embed_size = 128
hidden_size = 256
num_layers = 2
num_classes = 2
seq_length = 100

model = TextLSTM(vocab_size, embed_size, hidden_size, num_layers, num_classes)
sample_input = torch.randint(0, vocab_size, (32, seq_length))  # 模拟输入数据
output = model(sample_input)

# 输出模型结果
print("TextLSTM模型输出形状:", output.shape)
print("TextLSTM模型输出:", output)

# 例1-5
import torch
import torch.nn as nn
import time

# 定义自注意力机制的基本实现
class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # 计算注意力分数，并进行缩放
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / (self.embed_size ** (1 / 2))
        attention_weights = self.softmax(attention_scores)
        output = torch.bmm(attention_weights, V)
        return output

# 定义RNN模型的基本实现
class SimpleRNN(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)

    def forward(self, x):
        output, _ = self.rnn(x)
        return output

# 测试自注意力机制和RNN的计算时间
embed_size = 64
hidden_size = 64
seq_length = 100
batch_size = 32

# 创建输入数据
input_data = torch.rand(batch_size, seq_length, embed_size)

# 初始化模型
self_attention = SelfAttention(embed_size)
rnn_model = SimpleRNN(embed_size, hidden_size)

# 测试自注意力机制的计算时间
start_time = time.time()
self_attention_output = self_attention(input_data)
end_time = time.time()
self_attention_time = end_time - start_time

# 测试RNN的计算时间
start_time = time.time()
rnn_output = rnn_model(input_data)
end_time = time.time()
rnn_time = end_time - start_time

# 输出计算时间和结果
print("自注意力机制输出形状:", self_attention_output.shape)
print("自注意力机制计算时间:", self_attention_time, "秒")
print("RNN输出形状:", rnn_output.shape)
print("RNN计算时间:", rnn_time, "秒")

# 例1-6
import torch
import torch.nn as nn
import time

# 定义自注意力机制的基本实现
class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # 计算注意力分数，并进行缩放
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / (self.embed_size ** (1 / 2))
        attention_weights = self.softmax(attention_scores)
        output = torch.bmm(attention_weights, V)
        return output

# 定义RNN模型的基本实现
class SimpleRNN(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)

    def forward(self, x):
        output, _ = self.rnn(x)
        return output

# 测试自注意力机制和RNN的计算时间
embed_size = 64
hidden_size = 64
seq_length = 100
batch_size = 32

# 创建输入数据
input_data = torch.rand(batch_size, seq_length, embed_size)

# 初始化模型
self_attention = SelfAttention(embed_size)
rnn_model = SimpleRNN(embed_size, hidden_size)

# 测试自注意力机制的计算时间
start_time = time.time()
self_attention_output = self_attention(input_data)
end_time = time.time()
self_attention_time = end_time - start_time

# 测试RNN的计算时间
start_time = time.time()
rnn_output = rnn_model(input_data)
end_time = time.time()
rnn_time = end_time - start_time

# 输出计算时间和结果
print("自注意力机制输出形状:", self_attention_output.shape)
print("自注意力机制计算时间:", self_attention_time, "秒")
print("RNN输出形状:", rnn_output.shape)
print("RNN计算时间:", rnn_time, "秒")

# 例1-7
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

# 定义BERT分类模型
class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask):
        # BERT模型输出
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # 取[CLS]标记对应的隐藏状态
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Dropout和全连接层进行分类
        cls_output = self.dropout(cls_output)
        out = self.fc(cls_output)
        return out

# 加载BERT分词器和模型
bert_model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
model = BERTClassifier(bert_model_name, num_classes=2)

# 输入文本示例
texts = ["This is a positive example.", "This is a negative example."]
encoding = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=32)

# 模型推理
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']
output = model(input_ids, attention_mask)

# 输出结果
print("BERT模型输出形状:", output.shape)
print("BERT模型输出:", output)



# 例1-8
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 加载GPT模型和分词器
gpt_model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_name)
model = GPT2LMHeadModel.from_pretrained(gpt_model_name)

# 输入文本并进行分词
input_text = "Once upon a time in a distant land,"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# 使用GPT生成文本
output_sequences = model.generate(
    input_ids=input_ids,
    max_length=50,         # 设置生成文本的最大长度
    num_return_sequences=1, # 返回的生成序列数
    no_repeat_ngram_size=2, # 防止生成重复的短语
    top_k=50,               # 限制每次生成的单词数量
    top_p=0.95,             # 采用nucleus sampling
    temperature=0.7,        # 控制生成文本的随机性
)

# 解码生成的输出文本
generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

# 输出结果
print("GPT模型生成的文本:")
print(generated_text)

# 例1-9
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

# 定义BERT分类模型
class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.3)

        # 冻结BERT前几层
        for param in list(self.bert.parameters())[:int(len(list(self.bert.parameters())) * 0.7)]:
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        out = self.fc(cls_output)
        return out

# 初始化分词器和模型
bert_model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
model = BERTClassifier(bert_model_name, num_classes=2)

# 定义数据和优化器
texts = ["这是一个积极的例子。", "这是一个消极的例子。"]
encoding = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=32)
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']

# 优化器和学习率调度器
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=100)

# 训练循环中的单次前向传播示例
model.train()
optimizer.zero_grad()
output = model(input_ids, attention_mask)
loss = nn.CrossEntropyLoss()(output, torch.tensor([1, 0]))  # 假设标签为[1, 0]
loss.backward()
optimizer.step()
scheduler.step()

# 输出结果
print("BERT模型输出形状:", output.shape)
print("BERT模型输出:", output)
print("训练损失:", loss.item())

# 例1-10
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, AdamW, get_cosine_schedule_with_warmup

# 定义BERT分类模型
class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

        # 冻结部分层
        for param in list(self.bert.parameters())[:int(len(list(self.bert.parameters())) * 0.5)]:
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        out = self.fc(cls_output)
        return out

# 初始化模型和分词器
bert_model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
model = BERTClassifier(bert_model_name, num_classes=2)

# 定义示例数据
texts = ["这个产品非常好，值得购买。", "产品质量不好，非常失望。"]
encoding = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=32)
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']

# 定义优化器和分层学习率
optimizer = AdamW([
    {'params': model.bert.encoder.layer[-2:].parameters(), 'lr': 1e-5},
    {'params': model.fc.parameters(), 'lr': 2e-5}
], weight_decay=1e-4)

# 采用余弦退火学习率调度器
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=100)

# 模拟训练过程中的单次前向传播
model.train()
optimizer.zero_grad()
output = model(input_ids, attention_mask)
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.6, 0.4]))  # 假设不同类别的损失权重
loss = loss_fn(output, torch.tensor([1, 0]))  # 标签示例
loss.backward()
optimizer.step()
scheduler.step()

# 输出结果
print("BERT模型输出形状:", output.shape)
print("BERT模型输出:", output)
print("训练损失:", loss.item())


# 例1-11
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, AdamW, get_cosine_schedule_with_warmup
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# 定义情感分析数据集类
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=32):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# 定义BERT情感分类模型
class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

        # 冻结部分BERT层
        for param in list(self.bert.parameters())[:int(len(list(self.bert.parameters())) * 0.5)]:
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        out = self.fc(cls_output)
        return out

# 加载预训练模型和分词器
bert_model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
model = BERTClassifier(bert_model_name, num_classes=2)

# 准备数据
texts = ["这个产品真的很好，使用感受非常棒！", "服务态度差，完全不推荐。", "很满意，物流很快。", "质量不行，很失望。"]
labels = [1, 0, 1, 0]  # 1表示积极，0表示消极

# 划分训练集和测试集
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.5, random_state=42)
train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)

# 定义数据加载器
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2)

# 定义优化器、调度器和损失函数
optimizer = AdamW([
    {'params': model.bert.encoder.layer[-2:].parameters(), 'lr': 1e-5},
    {'params': model.fc.parameters(), 'lr': 2e-5}
], weight_decay=1e-4)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=100)
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.6, 0.4]))

# 模拟训练过程中的一个完整epoch
model.train()
for batch in train_loader:
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['label']

    optimizer.zero_grad()
    outputs = model(input_ids, attention_mask)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()
    scheduler.step()
    print("训练损失:", loss.item())

# 模型评估
model.eval()
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        
        outputs = model(input_ids, attention_mask)
        predictions = torch.argmax(outputs, dim=1)
        print("真实标签:", labels)
        print("预测结果:", predictions)



