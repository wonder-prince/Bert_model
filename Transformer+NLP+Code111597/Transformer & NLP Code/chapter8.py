# 例8-1
import torch
import torch.nn.functional as F

# 设置查询、键和值矩阵的大小
batch_size = 2
seq_len = 4
d_k = 64  # 键和查询的维度
d_v = 64  # 值的维度

# 创建查询、键和值矩阵，初始化为随机数
Q = torch.rand(batch_size, seq_len, d_k)
K = torch.rand(batch_size, seq_len, d_k)
V = torch.rand(batch_size, seq_len, d_v)

# 计算查询和键的点积
scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

# 使用softmax归一化，得到注意力权重
attention_weights = F.softmax(scores, dim=-1)

# 使用注意力权重对值矩阵进行加权求和
output = torch.matmul(attention_weights, V)

# 输出结果
print("查询矩阵 Q:\n", Q)
print("键矩阵 K:\n", K)
print("值矩阵 V:\n", V)
print("缩放后的点积 scores:\n", scores)
print("注意力权重 attention_weights:\n", attention_weights)
print("最终输出 output:\n", output)


# 例8-2
import torch
import torch.nn.functional as F

# 定义查询、键和值矩阵的维度
batch_size = 2
seq_len = 5
d_k = 64  # 键和查询的维度
d_v = 64  # 值的维度

# 随机初始化查询、键和值矩阵
Q = torch.rand(batch_size, seq_len, d_k)
K = torch.rand(batch_size, seq_len, d_k)
V = torch.rand(batch_size, seq_len, d_v)

# 计算查询与键的点积，并缩放
scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

# 使用Softmax对点积结果进行归一化，得到注意力权重
attention_weights = F.softmax(scores, dim=-1)

# 计算加权求和值
output = torch.matmul(attention_weights, V)

# 打印结果
print("查询矩阵 Q:\n", Q)
print("键矩阵 K:\n", K)
print("值矩阵 V:\n", V)
print("缩放后的点积 scores:\n", scores)
print("注意力权重 attention_weights:\n", attention_weights)
print("加权后的输出 output:\n", output)


# 例8-3
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        # Define linear layers for query, key, value, and the final dense layer
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        # Split the last dimension into (num_heads, depth)
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(1, 2)

    def scaled_dot_product_attention(self, q, k, v):
        # Compute the attention scores and apply softmax
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        dk = torch.tensor(q.size(-1), dtype=torch.float32)
        scaled_attention_logits = matmul_qk / math.sqrt(dk)
        attention_weights = torch.softmax(scaled_attention_logits, dim=-1)
        output = torch.matmul(attention_weights, v)
        return output, attention_weights

    def forward(self, v, k, q, mask=None):
        batch_size = q.size(0)

        # Linear transformations for q, k, v
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # Split the heads
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Apply scaled dot-product attention
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v)
        scaled_attention = scaled_attention.transpose(1, 2).contiguous()

        # Concatenate heads and put through final linear layer
        concat_attention = scaled_attention.view(batch_size, -1, self.d_model)
        output = self.dense(concat_attention)

        return output, attention_weights

# Initialize parameters
d_model = 512  # Dimension of model
num_heads = 8  # Number of attention heads
batch_size = 2  # Batch size
seq_len = 10    # Sequence length

# Initialize random input tensor with batch size and sequence length
input_tensor = torch.rand(batch_size, seq_len, d_model)

# Initialize multi-head attention
multi_head_attention = MultiHeadAttention(d_model, num_heads)

# Compute attention
output, attention_weights = multi_head_attention(input_tensor, input_tensor, input_tensor)

print("Output Shape:", output.shape)
print("Attention Weights Shape:", attention_weights.shape)


# 例8-4
import torch
import torch.nn as nn
import math

# 多头注意力机制实现，包含初始化和正则化
class MultiHeadAttentionWithRegularization(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate=0.1):
        super(MultiHeadAttentionWithRegularization, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        
        # 定义线性层和最终的线性变换层
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)
        
        # Dropout和LayerNorm层
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(d_model)

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.wq.weight)
        nn.init.xavier_uniform_(self.wk.weight)
        nn.init.xavier_uniform_(self.wv.weight)
        nn.init.xavier_uniform_(self.dense.weight)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(1, 2)

    def scaled_dot_product_attention(self, q, k, v):
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        dk = torch.tensor(q.size(-1), dtype=torch.float32)
        scaled_attention_logits = matmul_qk / math.sqrt(dk)
        attention_weights = torch.softmax(scaled_attention_logits, dim=-1)
        output = torch.matmul(attention_weights, v)
        return output, attention_weights

    def forward(self, v, k, q, mask=None):
        batch_size = q.size(0)

        # Query, Key, Value线性变换
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # 分头操作
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # 注意力计算
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v)
        scaled_attention = scaled_attention.transpose(1, 2).contiguous()

        # 拼接多头并通过最终的线性变换
        concat_attention = scaled_attention.view(batch_size, -1, self.d_model)
        output = self.dense(concat_attention)
        
        # 应用Dropout和LayerNorm
        output = self.dropout(output)
        output = self.layer_norm(output + q.view(batch_size, -1, self.d_model))

        return output, attention_weights

# 设置模型参数
d_model = 512
num_heads = 8
dropout_rate = 0.1
batch_size = 2
seq_len = 10

# 输入张量
input_tensor = torch.rand(batch_size, seq_len, d_model)

# 初始化并测试模型
multi_head_attention = MultiHeadAttentionWithRegularization(d_model, num_heads, dropout_rate)
output, attention_weights = multi_head_attention(input_tensor, input_tensor, input_tensor)

print("Output Shape:", output.shape)
print("Attention Weights Shape:", attention_weights.shape)


# 例8-5
import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个简单的深度神经网络类，包含层归一化
class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)  # 第一层的层归一化
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)  # 第二层的层归一化
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.layer_norm1(x)
        x = torch.relu(x)  # 使用ReLU激活函数
        x = self.fc2(x)
        x = self.layer_norm2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x

# 初始化模型、损失函数和优化器
input_dim = 10
hidden_dim = 20
output_dim = 1
model = SimpleModel(input_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 创建随机输入数据和目标数据用于训练
torch.manual_seed(0)
data = torch.randn(100, input_dim)
target = torch.randn(100, output_dim)

# 训练模型
for epoch in range(100):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")

# 查看模型的层归一化后的输出
with torch.no_grad():
    test_input = torch.randn(1, input_dim)
    test_output = model(test_input)
    print("Test input:", test_input)
    print("Model output after LayerNorm and activation:", test_output)


# 例8-6
import torch
import torch.nn as nn
import torch.optim as optim

# 定义带有残差连接的简单神经网络
class ResidualBlock(nn.Module):
    def __init__(self, input_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        # 输入经过第一层全连接层和ReLU激活
        residual = x
        out = torch.relu(self.fc1(x))
        out = self.fc2(out)
        # 加入残差连接，并通过层归一化
        out = self.layer_norm(out + residual)
        return torch.relu(out)

# 构建多层残差网络
class DeepResidualNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, num_blocks):
        super(DeepResidualNetwork, self).__init__()
        self.blocks = nn.ModuleList([ResidualBlock(input_dim) for _ in range(num_blocks)])
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.fc_out(x)

# 初始化模型、损失函数和优化器
input_dim = 10
output_dim = 1
num_blocks = 5  # 使用5个残差块
model = DeepResidualNetwork(input_dim, output_dim, num_blocks)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 创建随机输入数据和目标数据用于训练
torch.manual_seed(0)
data = torch.randn(100, input_dim)
target = torch.randn(100, output_dim)

# 训练模型
for epoch in range(100):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")

# 测试残差连接对模型输出的影响
with torch.no_grad():
    test_input = torch.randn(1, input_dim)
    test_output = model(test_input)
    print("Test input:", test_input)
    print("Model output with residual connection:", test_output)


# 例8-7
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import BartTokenizer, BartForConditionalGeneration

# 加载Bart模型和分词器，用于机器翻译和摘要生成任务
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

# 示例输入文本
input_text = "The quick brown fox jumps over the lazy dog."
target_text = "Le renard brun rapide saute par-dessus le chien paresseux."

# 编码输入和目标文本
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
target_ids = tokenizer(target_text, return_tensors="pt").input_ids

# 将输入数据传递给模型，获取输出和注意力权重
outputs = model(input_ids=input_ids, labels=target_ids, output_attentions=True)
loss = outputs.loss
logits = outputs.logits
attentions = outputs.attentions  # 注意力权重

# 可视化第一个注意力层的权重
import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention(attention_weights, input_tokens, target_tokens, layer_idx=0, head_idx=0):
    # 提取特定层和头的注意力权重
    attn_weights = attention_weights[layer_idx][0][head_idx].detach().cpu().numpy()
    plt.figure(figsize=(10, 10))
    sns.heatmap(attn_weights, xticklabels=input_tokens, yticklabels=target_tokens, cmap="YlGnBu")
    plt.xlabel("Input Tokens")
    plt.ylabel("Target Tokens")
    plt.title(f"Attention weights - Layer {layer_idx + 1}, Head {head_idx + 1}")
    plt.show()

# 将输入和目标ID转换为可视化的token
input_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
target_tokens = tokenizer.convert_ids_to_tokens(target_ids[0])

# 绘制第一个注意力层的第一个头的注意力分布
plot_attention(attentions, input_tokens, target_tokens, layer_idx=0, head_idx=0)

# 输出模型的损失值
print(f"Loss: {loss.item()}")

# 生成翻译文本
generated_ids = model.generate(input_ids)
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

print(f"Generated Text: {generated_text}")


# 例8-8
import torch
import torch.nn as nn
from transformers import BartTokenizer, BartForConditionalGeneration

# 加载Bart模型和分词器，用于中译英翻译任务
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

# 示例长文本输入（中文）
input_text_cn = (
    "在快速发展的现代社会中，科技创新不仅驱动了经济的增长，也改变了人们的生活方式。"
    "越来越多的智能设备进入人们的日常生活，极大地提高了生活的便利性和效率。"
    "从智能手机到智能家居，科技在不断进步的同时，也带来了诸多新的挑战和机遇。"
)

# 编码输入文本
input_ids = tokenizer(input_text_cn, return_tensors="pt").input_ids

# 翻译文本生成
generated_ids = model.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# 输出生成的英文翻译
print("Generated Translation (CN to EN):")
print(generated_text)

# 获取注意力权重
outputs = model(input_ids=input_ids, output_attentions=True)
attentions = outputs.attentions  # 注意力权重

# 将输入文本ID转换为token以便可视化
input_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

# 可视化注意力权重
import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention(attention_weights, input_tokens, layer_idx=0, head_idx=0):
    # 提取指定层和头的注意力权重
    attn_weights = attention_weights[layer_idx][0][head_idx].detach().cpu().numpy()
    plt.figure(figsize=(10, 10))
    sns.heatmap(attn_weights, xticklabels=input_tokens, yticklabels=input_tokens, cmap="YlGnBu")
    plt.xlabel("Input Tokens")
    plt.ylabel("Attention Weights on Tokens")
    plt.title(f"Attention Weights - Layer {layer_idx + 1}, Head {head_idx + 1}")
    plt.show()

# 可视化第一个注意力层的第一个头的权重分布
plot_attention(attentions, input_tokens, layer_idx=0, head_idx=0)



# 例8-9
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertModel

# 加载BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertModel.from_pretrained("bert-base-chinese", output_attentions=True)

# 从本地文件读取文本数据
with open("input_text.txt", "r", encoding="utf-8") as f:
    input_text = f.read().strip()

# 对输入文本进行编码
inputs = tokenizer(input_text, return_tensors="pt")
input_ids = inputs["input_ids"]

# 获取模型的输出和注意力权重
outputs = model(**inputs)
attentions = outputs.attentions  # 注意力权重

# 将Token ID转换为实际词汇，便于后续的可视化展示
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

# 可视化函数，展示特定层和头的注意力权重
def plot_attention(attention_weights, tokens, layer_idx=0, head_idx=0):
    attn_weights = attention_weights[layer_idx][0][head_idx].detach().cpu().numpy()
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_weights, xticklabels=tokens, yticklabels=tokens, cmap="YlGnBu", annot=True, fmt=".2f")
    plt.xlabel("Tokens (Input)")
    plt.ylabel("Attention Weight on Tokens")
    plt.title(f"Attention Weights - Layer {layer_idx + 1}, Head {head_idx + 1}")
    plt.show()

# 展示第1层，第1个注意力头的注意力权重
plot_attention(attentions, tokens, layer_idx=0, head_idx=0)



# 例8-10
import torch
import torch.nn as nn
import math

# 多头注意力机制实现
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        # 定义线性层用于Q, K, V的变换
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)

        # 输出层
        self.out_linear = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, query, key, value, mask=None):
        # Scaled Dot-Product Attention计算
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, value)
        return output, attention_weights

    def forward(self, query, key, value, mask=None):
        # 多头注意力机制前向传播
        batch_size = query.size(0)

        # 通过线性变换计算Q, K, V
        query = self.query_linear(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.key_linear(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.value_linear(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 计算注意力输出
        attention_output, _ = self.scaled_dot_product_attention(query, key, value, mask)

        # 拼接所有头的输出，并通过输出线性层
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_linear(attention_output)

        return output

# 前馈网络实现
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# Transformer块定义，包含多头注意力和前馈网络
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):


# 例8-11
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_size = embed_size
        assert embed_size % num_heads == 0, "Embedding size should be divisible by heads"
        
        # 定义查询、键、值的线性变换
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        
        # 最终线性层
        self.fc_out = nn.Linear(embed_size, embed_size)
    
    def forward(self, values, keys, queries, mask):
        # 获取批次大小
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]
        
        # 维度变换，使得每个头的维度大小为 embed_size // num_heads
        queries = self.query(queries).view(N, query_len, self.num_heads, self.embed_size // self.num_heads)
        keys = self.key(keys).view(N, key_len, self.num_heads, self.embed_size // self.num_heads)
        values = self.value(values).view(N, value_len, self.num_heads, self.embed_size // self.num_heads)

        # 计算点积注意力
        attention = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) / math.sqrt(self.embed_size // self.num_heads)
        
        # 应用遮罩（可选）
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-1e20"))
        
        # 计算注意力权重
        attention = torch.softmax(attention, dim=-1)
        
        # 对值进行加权求和
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.embed_size)
        
        # 通过线性层输出
        out = self.fc_out(out)
        return out


# 例8-12
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # 初始化一个位置编码矩阵，维度为 (max_len, embed_size)
        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)
        
        # 定义正余弦函数的分母部分，使用 10000 的幂次缩放嵌入维度
        div_term = torch.exp(torch.arange(0, embed_size, 2) * (-math.log(10000.0) / embed_size))
        
        # 初始化一个编码矩阵，shape为 (max_len, embed_size)，用于存储 sin 和 cos 值
        pos_embedding = torch.zeros(max_len, embed_size)
        
        # 将偶数索引位置赋值为 sin 值
        pos_embedding[:, 0::2] = torch.sin(position * div_term)
        
        # 将奇数索引位置赋值为 cos 值
        pos_embedding[:, 1::2] = torch.cos(position * div_term)
        
        # 增加一个维度并注册为模型参数
        pos_embedding = pos_embedding.unsqueeze(0)  # (1, max_len, embed_size)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, x):
        # 在输入的嵌入表示上加上位置编码
        # x.shape -> (batch_size, seq_len, embed_size)
        # pos_embedding[:,:x.size(1),:] -> (1, seq_len, embed_size)
        return x + self.pos_embedding[:, :x.size(1), :]




























