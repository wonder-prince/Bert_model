# 例11-1
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

# 加载预训练的BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# 数据示例
sentences = ["The book is great!", "The movie was terrible."]
labels = [1, 0]  # 假设1代表积极，0代表消极

# 数据预处理
inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)

# 冻结所有BERT的层
for param in model.bert.parameters():
    param.requires_grad = False

# 解冻特定的层（例如最后两层）
for param in model.bert.encoder.layer[-2:].parameters():
    param.requires_grad = True

# 定义优化器，仅优化解冻层的参数
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)

# 训练过程示例
model.train()
for epoch in range(3):  # 假设训练3个周期
    outputs = model(**inputs, labels=torch.tensor(labels))
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Epoch {epoch+1} - Loss: {loss.item()}")



# 例11-2
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

# 加载预训练的BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# 示例数据
sentences = ["I loved the food at this place.", "The service was poor."]
labels = [1, 0]  # 1表示正面，0表示负面

# 预处理数据
inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)

# 冻结所有层的参数
for param in model.bert.parameters():
    param.requires_grad = False

# 解冻最后一层
for param in model.bert.encoder.layer[-1].parameters():
    param.requires_grad = True

# 定义优化器，仅更新解冻层的参数
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)

# 训练过程
model.train()
for epoch in range(2):  # 假设训练2个周期
    outputs = model(**inputs, labels=torch.tensor(labels))
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Epoch {epoch + 1} - Loss: {loss.item()}")



# 例11-3
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

# 加载预训练的BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# 示例数据
sentences = ["I love learning about AI.", "Deep learning is fascinating."]
labels = torch.tensor([1, 0])  # 1表示正面，0表示负面

# 预处理数据
inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)

# 冻结所有层的参数
for param in model.bert.parameters():
    param.requires_grad = False

# 定义优化器，仅更新解冻层的参数
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)

# 逐层解冻的训练过程
model.train()
for epoch in range(3):  # 假设逐层解冻3个周期
    # 每个周期解冻一个新的层
    if epoch < len(model.bert.encoder.layer):
        for param in model.bert.encoder.layer[-(epoch + 1)].parameters():
            param.requires_grad = True
    
    # 定义损失计算和优化步骤
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # 输出每个周期的损失值
    print(f"Epoch {epoch + 1} - Loss: {loss.item()}")
    
    # 显示当前解冻层
    print(f"Unfrozen Layer: -{epoch + 1}")


# 例11-4
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

# 加载预训练的BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# 示例数据
sentences = ["Machine learning is evolving.", "Natural language processing is intriguing."]
labels = torch.tensor([1, 0])  # 1表示正面，0表示负面

# 预处理数据
inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)

# 冻结底层参数，只训练顶层参数
for param in model.bert.encoder.layer[:-2]:  # 冻结底层层（除去最后两层）
    param.requires_grad = False

# 定义优化器，仅更新解冻层的参数
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)

# 微调训练过程
model.train()
for epoch in range(3):  # 假设训练3个周期
    optimizer.zero_grad()
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

    # 输出每个周期的损失值
    print(f"Epoch {epoch + 1} - Loss: {loss.item()}")
    
    # 输出当前训练层信息
    trainable_layers = [name for name, param in model.named_parameters() if param.requires_grad]
    print(f"Trainable Layers: {trainable_layers}")


# 例11-5
import pandas as pd
from sklearn.utils import resample
from transformers import BertTokenizer
import re

# 模拟加载金融和医学领域数据集
data = {
    "text": [
        "The company's revenue increased by 25% last quarter.",
        "A new medication for heart disease was approved by the FDA.",
        "Stock prices fluctuate due to market uncertainty.",
        "The patient shows symptoms of a rare neurological disorder.",
        "The merger will affect the overall market strategy.",
        "Clinical trials show promising results for the cancer drug."
    ],
    "label": ["finance", "medical", "finance", "medical", "finance", "medical"]
}
df = pd.DataFrame(data)

# 文本数据清洗函数
def clean_text(text):
    text = re.sub(r'\d+', '', text)  # 去除数字
    text = re.sub(r'[^\w\s]', '', text)  # 去除标点符号
    text = text.lower()  # 转为小写
    return text

# 应用清洗函数
df["cleaned_text"] = df["text"].apply(clean_text)

# 标签平衡，统计每个类别样本数
print("原始标签分布：")
print(df["label"].value_counts())

# 重采样以实现标签平衡
finance_samples = df[df.label == "finance"]
medical_samples = df[df.label == "medical"]

# 假设标签不平衡，进行过采样或下采样
finance_upsampled = resample(finance_samples, replace=True, n_samples=len(medical_samples), random_state=42)
df_balanced = pd.concat([finance_upsampled, medical_samples])

print("\n标签平衡后的分布：")
print(df_balanced["label"].value_counts())

# 数据预处理后示例输出
print("\n预处理后的文本数据示例：")
print(df_balanced[["cleaned_text", "label"]])

# Tokenizer 实例化并应用于清洗后的文本数据
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
encoded_inputs = tokenizer(df_balanced["cleaned_text"].tolist(), padding=True, truncation=True, return_tensors="pt")

# 打印编码后的示例
print("\n示例编码输入 (前两个文本)：")
print(encoded_inputs["input_ids"][:2])
print(encoded_inputs["attention_mask"][:2])

# 示例代码完成
首先，加载并显示金融和医学领域的模拟数据。使用clean_text函数清洗文本数据，删除多余的数字和标点符号，并转为小写。然后对数据进行标签分布分析，应用重采样技术对类别进行平衡。处理后的数据集由两个类别的均衡样本组成，确保训练过程中模型对每类数据的学习效果。之后，对平衡后的文本数据进行BERT的tokenizer编码，确保输入格式和模型要求一致。
运行结果如下：
原始标签分布：
finance    3
medical    3
Name: label, dtype: int64

标签平衡后的分布：
finance    3
medical    3
Name: label, dtype: int64

预处理后的文本数据示例：
                              cleaned_text    label
0      the companys revenue increased by quarter  finance
1       stock prices fluctuate due to market uncertainty  finance
2          the merger will affect the overall market strategy finance
3      a new medication for heart disease was approved by the fda  medical
4      the patient shows symptoms of a rare neurological disorder medical
5      clinical trials show promising results for the cancer drug medical

示例编码输入 (前两个文本)：
tensor([[  101,  1996,  2194,  2042,  ....]])
tensor([[1, 1, 1, ..., 0, 0]])


# 例11-6
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

# 加载数据集（示例数据）
data = [
    ("The company posted a significant increase in quarterly revenue.", 0),
    ("New heart disease medication approved by FDA.", 1),
    ("Stock market affected by global events.", 0),
    ("Medical advancements in treating rare diseases.", 1)
]
labels = [item[1] for item in data]
texts = [item[0] for item in data]

# 实例化Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# 将标签转换为张量
labels_tensor = torch.tensor(labels)

# 加载预训练的BERT模型并调整参数
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 将模型设置为训练模式
model.train()

# 定义优化器和学习率调度器
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 3
total_steps = len(inputs["input_ids"]) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# 将数据加载至DataLoader
train_data = DataLoader(list(zip(inputs["input_ids"], inputs["attention_mask"], labels_tensor)), batch_size=2)

# 微调BERT模型
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    total_loss = 0
    for batch in train_data:
        input_ids, attention_mask, labels = batch

        # 梯度清零
        optimizer.zero_grad()

        # 前向传播，获取损失
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        # 反向传播
        loss.backward()

        # 梯度裁剪，避免梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 参数更新
        optimizer.step()

        # 更新学习率
        scheduler.step()

    avg_loss = total_loss / len(train_data)
    print(f"Average training loss: {avg_loss:.4f}")

# 测试阶段：打印模型参数信息
print("\n部分模型参数示例：")
for name, param in model.named_parameters():
    if "classifier" in name:
        print(f"{name}: {param[:2]}")
        break
使用BERT预训练的权重初始化模型参数，加载bert-base-uncased模型，并设置为二分类任务。在优化器选择方面，AdamW优化器被用来减少L2正则化对权重衰减的影响，从而优化参数更新过程。学习率设置为2e-5，并通过get_linear_schedule_with_warmup方法实现线性递减学习率调度器，使得在训练后期逐步降低学习率，以防止震荡。
微调过程中，每轮(epoch)计算总损失，进行梯度裁剪以避免梯度爆炸，然后通过优化器和调度器进行参数更新。代码在最后输出一部分微调后的模型参数，以便检查模型更新的有效性。
运行结果如下：
Epoch 1/3
Average training loss: 0.5463

Epoch 2/3
Average training loss: 0.4238

Epoch 3/3
Average training loss: 0.3456

部分模型参数示例：
classifier.weight: tensor([[ 0.0041, -0.0038],
                          [ 0.0052, -0.0014]])



# 例11-7
import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.optim as optim

# 初始化BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# 定义LoRA插入函数
class LoRA(nn.Module):
    def __init__(self, input_dim, rank):
        super(LoRA, self).__init__()
        # 定义低秩矩阵
        self.low_rank_left = nn.Parameter(torch.randn(input_dim, rank))
        self.low_rank_right = nn.Parameter(torch.randn(rank, input_dim))
        self.scaling_factor = 1.0 / (rank ** 0.5)

    def forward(self, x):
        # 低秩矩阵的插入
        lora_update = torch.matmul(self.low_rank_left, self.low_rank_right) * self.scaling_factor
        return x + torch.matmul(x, lora_update)

# 将LoRA应用到模型的encoder层
for layer in model.encoder.layer:
    layer.attention.self.query = LoRA(layer.attention.self.query.in_features, rank=8)

# 定义Prefix Tuning类
class PrefixTuning(nn.Module):
    def __init__(self, model, prefix_length=10, hidden_size=768):
        super(PrefixTuning, self).__init__()
        # 创建前缀向量
        self.prefix_embeddings = nn.Parameter(torch.randn(prefix_length, hidden_size))
        self.prefix_length = prefix_length
        self.hidden_size = hidden_size
        self.model = model

    def forward(self, input_ids, attention_mask):
        # 获取输入嵌入
        original_embeddings = self.model.embeddings(input_ids)
        
        # 将前缀添加到输入
        batch_size = input_ids.size(0)
        prefix_embeddings = self.prefix_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        modified_embeddings = torch.cat([prefix_embeddings, original_embeddings], dim=1)
        
        # 调整attention mask
        extended_attention_mask = torch.cat([torch.ones(batch_size, self.prefix_length).to(attention_mask.device), attention_mask], dim=1)
        return self.model(inputs_embeds=modified_embeddings, attention_mask=extended_attention_mask)

# 将Prefix Tuning集成到BERT中
prefix_tuning = PrefixTuning(model)
optimizer = optim.Adam(prefix_tuning.parameters(), lr=1e-5)

# 准备示例数据
text = "LoRA and Prefix Tuning are efficient methods for adapting large models."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# 模型训练流程
prefix_tuning.train()
for epoch in range(3):  # 训练3个epoch
    optimizer.zero_grad()
    outputs = prefix_tuning(input_ids=input_ids, attention_mask=attention_mask)
    last_hidden_states = outputs.last_hidden_state
    loss = (last_hidden_states ** 2).mean()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# 测试流程
prefix_tuning.eval()
with torch.no_grad():
    outputs = prefix_tuning(input_ids=input_ids, attention_mask=attention_mask)
    print("Output Embeddings:", outputs.last_hidden_state)



# 例11-8
import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.optim as optim

# 初始化BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# 定义Adapter模块
class Adapter(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=64):
        super(Adapter, self).__init__()
        # 降维层
        self.down_project = nn.Linear(input_dim, bottleneck_dim)
        # 升维层
        self.up_project = nn.Linear(bottleneck_dim, input_dim)
        # 激活函数
        self.activation = nn.ReLU()
        # 使用层归一化提高稳定性
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        # 降维 -> 激活 -> 升维 -> 层归一化
        residual = x
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        return self.layer_norm(residual + x)

# 将Adapter模块插入BERT的encoder层
for layer in model.encoder.layer:
    layer.attention.self.adapter = Adapter(layer.attention.self.query.in_features)

# 定义训练流程
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# 准备示例数据
text = "Adapter Tuning is a method for efficient model fine-tuning."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# 冻结BERT模型的所有参数，只训练adapter模块
for param in model.parameters():
    param.requires_grad = False

for name, param in model.named_parameters():
    if "adapter" in name:
        param.requires_grad = True

# 模型训练流程
model.train()
for epoch in range(3):  # 训练3个epoch
    optimizer.zero_grad()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    last_hidden_states = outputs.last_hidden_state
    loss = (last_hidden_states ** 2).mean()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# 测试流程
model.eval()
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    print("Output Embeddings:", outputs.last_hidden_state)




















