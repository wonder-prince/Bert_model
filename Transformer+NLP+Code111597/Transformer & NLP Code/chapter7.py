# 例7-1
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Step 2: 假设加载了预训练的英语和法语词嵌入
# 模拟一些数据作为示例（实际项目中应替换为真实嵌入）
source_embeddings = np.random.rand(5000, 300)  # 假设有5000个英文词，每个词300维度
target_embeddings = np.random.rand(5000, 300)  # 假设有5000个法文词，大小相同

# 将numpy数组转换为torch张量
source_embeddings = torch.FloatTensor(source_embeddings)
target_embeddings = torch.FloatTensor(target_embeddings)

# Step 3: 定义生成器（映射矩阵）和判别器
class Generator(nn.Module):
    def __init__(self, input_dim):
        super(Generator, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim, bias=False)
    
    def forward(self, x):
        return self.linear(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# 初始化生成器和判别器
input_dim = 300
generator = Generator(input_dim)
discriminator = Discriminator(input_dim)

# 定义优化器
g_optimizer = optim.Adam(generator.parameters(), lr=0.001)
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)

# Step 4: 对抗训练过程设置
epochs = 1000  # 训练轮数
for epoch in range(epochs):
    # 随机采样一批源语言和目标语言词嵌入
    idx = np.random.choice(5000, 128, replace=False)
    source_batch = source_embeddings[idx]
    target_batch = target_embeddings[idx]

    # 训练判别器
    d_optimizer.zero_grad()
    
    # 将源语言词嵌入映射到目标语言空间
    fake_target = generator(source_batch)
    
    # 判别器对真实和伪目标嵌入的判断输出
    real_output = discriminator(target_batch)
    fake_output = discriminator(fake_target.detach())
    
    # 判别器损失计算，目标是区分真实和伪造
    d_loss = -torch.mean(torch.log(real_output) + torch.log(1 - fake_output))
    d_loss.backward()
    d_optimizer.step()

    # 训练生成器
    g_optimizer.zero_grad()
    
    # 生成映射后的伪目标嵌入
    fake_target = generator(source_batch)
    fake_output = discriminator(fake_target)
    
    # 生成器损失，目标是使判别器认为伪目标嵌入为真
    g_loss = -torch.mean(torch.log(fake_output))
    g_loss.backward()
    g_optimizer.step()
    
    # 每隔100轮输出损失信息
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

# Step 5: 生成对齐后的词嵌入
aligned_source_embeddings = generator(source_embeddings).detach().numpy()

# 计算对齐后英语词嵌入与法语词嵌入的余弦相似度
similarities = cosine_similarity(aligned_source_embeddings, target_embeddings)
print("对齐后的部分词对相似度：")
print(similarities[:10, :10])  # 展示前10个词对的相似度



# 例7-2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

# 模拟源语言和目标语言的词嵌入
np.random.seed(42)
source_embeddings = np.random.rand(100, 300)  # 源语言词嵌入
target_embeddings = np.random.rand(100, 300)  # 目标语言词嵌入

# 对齐前的余弦相似度
initial_similarity = cosine_similarity(source_embeddings, target_embeddings).mean()
print(f"Initial Similarity (Before Alignment): {initial_similarity:.4f}")

# 构建投影矩阵
def train_projection_matrix(src_emb, tgt_emb, learning_rate=0.01, epochs=100):
    projection_matrix = np.random.rand(src_emb.shape[1], tgt_emb.shape[1])
    for epoch in range(epochs):
        # 计算投影后的源语言嵌入
        projected_src = src_emb.dot(projection_matrix)
        
        # 计算当前的余弦相似度
        similarity = cosine_similarity(projected_src, tgt_emb).mean()
        
        # 损失函数：负的余弦相似度
        loss = -similarity
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f} - Similarity: {similarity:.4f}")
        
        # 计算梯度
        gradient = -2 * src_emb.T.dot(tgt_emb - projected_src) / src_emb.shape[0]
        
        # 更新投影矩阵
        projection_matrix -= learning_rate * gradient
    return projection_matrix

# 训练投影矩阵
projection_matrix = train_projection_matrix(source_embeddings, target_embeddings, epochs=10)

# 使用投影矩阵对齐源语言嵌入
projected_source_embeddings = source_embeddings.dot(projection_matrix)

# 对齐后的余弦相似度
aligned_similarity = cosine_similarity(projected_source_embeddings, target_embeddings).mean()
print(f"Aligned Similarity (After Alignment): {aligned_similarity:.4f}")

# 测试跨语言相似度计算
test_src_sentence = np.random.rand(1, 300)  # 模拟测试句子（源语言）
test_tgt_sentence = np.random.rand(1, 300)  # 模拟测试句子（目标语言）

# 使用投影矩阵将源语言测试句子映射到目标语言空间
projected_test_src = test_src_sentence.dot(projection_matrix)
similarity_score = cosine_similarity(projected_test_src, test_tgt_sentence)[0][0]

print(f"Similarity between test source and target sentence: {similarity_score:.4f}")


# 例7-3
import torch

# 初始化XLM-RoBERTa的预训练模型和分词器
model_name = "xlm-roberta-base"
tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
model = XLMRobertaModel.from_pretrained(model_name)

# 定义输入文本，包含多语言文本示例
texts = ["Hello, how are you?", "Bonjour, comment ça va?", "Hola, ¿cómo estás?"]
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

# 使用模型生成多语言文本的嵌入向量
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state  # 获取最后一层隐藏状态

# 打印多语言嵌入向量的维度信息
print("Embedding Shape:", embeddings.shape)

# 显示示例嵌入向量，逐个词汇输出
for idx, text in enumerate(texts):
    print(f"Text: {text}")
    print("Embedding (first 5 tokens):", embeddings[idx, :5].numpy())

# 导入所需库
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, XLMRobertaModel
import torch
import torch.nn.functional as F

# 初始化预训练模型和分词器，用于分类任务
model_name = "xlm-roberta-base"
tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
classification_model = XLMRobertaForSequenceClassification.from_pretrained(model_name, num_labels=3)

# 定义输入文本，包括不同语言的文本
texts = ["Hello, how are you?", "Bonjour, comment ça va?", "Hola, ¿cómo estás?"]
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

# 使用分类模型生成类别预测
with torch.no_grad():
    outputs = classification_model(**inputs)
    predictions = F.softmax(outputs.logits, dim=-1)

# 打印每个文本的分类预测结果
for idx, text in enumerate(texts):
    print(f"Text: {text}")
    print("Predicted Class Probabilities:", predictions[idx].numpy())

# 使用模型进行翻译模拟（基于上下文的向量表示生成）
# 模拟多语言翻译：获取XLM-RoBERTa的嵌入
embedding_model = XLMRobertaModel.from_pretrained(model_name)

# 生成每个输入文本的嵌入向量，作为翻译特征
with torch.no_grad():
    embedded_outputs = embedding_model(**inputs)
    embeddings = embedded_outputs.last_hidden_state.mean(dim=1)  # 使用句子平均嵌入表示翻译

# 打印嵌入表示，模拟多语言翻译的向量生成
for idx, text in enumerate(texts):
    print(f"Text: {text}")
    print("Translation Vector (first 5 dimensions):", embeddings[idx, :5].numpy())

# 导入所需库
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, XLMRobertaModel
import torch
import torch.nn.functional as F

# 初始化预训练模型和分词器，用于分类任务
model_name = "xlm-roberta-base"
tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
classification_model = XLMRobertaForSequenceClassification.from_pretrained(model_name, num_labels=3)

# 定义输入文本，包括不同语言的文本
texts = ["Hello, how are you?", "Bonjour, comment ça va?", "Hola, ¿cómo estás?"]
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

# 使用分类模型生成类别预测
with torch.no_grad():
    outputs = classification_model(**inputs)
    predictions = F.softmax(outputs.logits, dim=-1)

# 打印每个文本的分类预测结果
for idx, text in enumerate(texts):
    print(f"Text: {text}")
    print("Predicted Class Probabilities:", predictions[idx].numpy())

# 使用模型进行翻译模拟（基于上下文的向量表示生成）
# 模拟多语言翻译：获取XLM-RoBERTa的嵌入
embedding_model = XLMRobertaModel.from_pretrained(model_name)

# 生成每个输入文本的嵌入向量，作为翻译特征
with torch.no_grad():
    embedded_outputs = embedding_model(**inputs)
    embeddings = embedded_outputs.last_hidden_state.mean(dim=1)  # 使用句子平均嵌入表示翻译

# 打印嵌入表示，模拟多语言翻译的向量生成
for idx, text in enumerate(texts):
    print(f"Text: {text}")
    print("Translation Vector (first 5 dimensions):", embeddings[idx, :5].numpy())


# 例7-4
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, load_metric
import numpy as np
import torch

# 加载XLM-RoBERTa模型和分词器
model_name = "xlm-roberta-base"
model = XLMRobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)

# 加载示例数据集
dataset = load_dataset("amazon_reviews_multi", "en")
train_data = dataset['train']
test_data = dataset['test']

# 数据预处理：将文本转换为模型所需的输入格式
def preprocess_function(examples):
    return tokenizer(examples["review_body"], truncation=True, padding="max_length", max_length=128)

encoded_train_data = train_data.map(preprocess_function, batched=True)
encoded_test_data = test_data.map(preprocess_function, batched=True)

# 定义评估指标
accuracy_metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs"
)

# 定义Trainer对象
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_train_data,
    eval_dataset=encoded_test_data,
    compute_metrics=compute_metrics
)

# 开始训练与评估
trainer.train()

# 评估模型
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)



import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# 自定义数据集类
class TranslationDataset(Dataset):
    def __init__(self, sentences, tokenizer, max_len=128):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        encoded = self.tokenizer.encode_plus(
            self.sentences[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].squeeze()
        attention_mask = encoded['attention_mask'].squeeze()
        return input_ids, attention_mask

# 初始化模型与分词器
model_name = "xlm-roberta-base"
tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
model = XLMRobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 句子示例
sentences = ["这是一个中文句子。", "Esto es una oración en español."]

# 构建数据集和数据加载器
dataset = TranslationDataset(sentences, tokenizer)
dataloader = DataLoader(dataset, batch_size=2)

# 模型推理
model.eval()
translations = []
with torch.no_grad():
    for input_ids, attention_mask in dataloader:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_tokens = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
        for token_ids in predicted_tokens:
            translation = tokenizer.decode(token_ids, skip_special_tokens=True)
            translations.append(translation)

# 输出结果
print("Translations:")
for i, translation in enumerate(translations):
    print(f"Sentence {i + 1}: {translation}")
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, load_metric
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.nn import CrossEntropyLoss

# 加载XLM-RoBERTa模型和分词器
model_name = "xlm-roberta-base"
model = XLMRobertaForSequenceClassification.from_pretrained(model_name, num_labels=3)  # 假设三分类任务
tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)

# 加载多语言数据集
dataset = load_dataset("amazon_reviews_multi", "all_languages")
train_data = dataset['train']
test_data = dataset['test']

# 数据预处理：将文本转换为模型所需的输入格式
def preprocess_function(examples):
    return tokenizer(examples["review_body"], truncation=True, padding="max_length", max_length=128)

encoded_train_data = train_data.map(preprocess_function, batched=True)
encoded_test_data = test_data.map(preprocess_function, batched=True)

# 计算标签权重（基于类别不均衡性）
label_counts = train_data['stars'].value_counts()
total_labels = sum(label_counts.values())
class_weights = {label: total_labels / count for label, count in label_counts.items()}
weights = [class_weights[label] for label in train_data['stars']]
sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

# 创建数据加载器
train_dataloader = DataLoader(encoded_train_data, sampler=sampler, batch_size=8)
test_dataloader = DataLoader(encoded_test_data, batch_size=8)

# 定义加权损失函数
class_weights_tensor = torch.tensor(list(class_weights.values()), dtype=torch.float)
loss_fn = CrossEntropyLoss(weight=class_weights_tensor)

# 自定义训练步骤以应用加权损失
def train(model, dataloader, optimizer, device):
    model.train()
    for batch in dataloader:
        optimizer.zero_grad()
        inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
        labels = batch["labels"].to(device)
        outputs = model(**inputs)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()

# 定义评估指标
accuracy_metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs"
)

# 使用Trainer进行训练和评估
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_train_data,
    eval_dataset=encoded_test_data,
    compute_metrics=compute_metrics
)

# 进行训练
trainer.train()

# 评估模型
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)


# 例7-5
import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# 自定义数据集类
class TranslationDataset(Dataset):
    def __init__(self, sentences, tokenizer, max_len=128):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        encoded = self.tokenizer.encode_plus(
            self.sentences[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].squeeze()
        attention_mask = encoded['attention_mask'].squeeze()
        return input_ids, attention_mask

# 初始化模型与分词器
model_name = "xlm-roberta-base"
tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
model = XLMRobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 句子示例
sentences = ["这是一个中文句子。", "Esto es una oración en español."]

# 构建数据集和数据加载器
dataset = TranslationDataset(sentences, tokenizer)
dataloader = DataLoader(dataset, batch_size=2)

# 模型推理
model.eval()
translations = []
with torch.no_grad():
    for input_ids, attention_mask in dataloader:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_tokens = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
        for token_ids in predicted_tokens:
            translation = tokenizer.decode(token_ids, skip_special_tokens=True)
            translations.append(translation)

# 输出结果
print("Translations:")
for i, translation in enumerate(translations):
    print(f"Sentence {i + 1}: {translation}")


# 例7-6
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from tqdm import tqdm

# 自定义翻译数据集
class TranslationDataset(Dataset):
    def __init__(self, sentences, targets, tokenizer, max_len=128):
        self.sentences = sentences
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        encoded = self.tokenizer.encode_plus(
            self.sentences[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].squeeze()
        attention_mask = encoded['attention_mask'].squeeze()
        target = torch.tensor(self.targets[idx], dtype=torch.long)
        return input_ids, attention_mask, target

# 初始化模型与分词器
model_name = "xlm-roberta-base"
tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
model = XLMRobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 示例数据
sentences = ["这是一个句子。", "Este es un enunciado."]
translations = ["This is a sentence.", "This is a sentence."]  # 目标翻译
labels = [1, 1]  # 1 表示正确翻译

# 数据集划分
train_sentences, val_sentences, train_labels, val_labels = train_test_split(sentences, labels, test_size=0.2, random_state=42)

# 构建数据集和数据加载器
train_dataset = TranslationDataset(train_sentences, train_labels, tokenizer)
val_dataset = TranslationDataset(val_sentences, val_labels, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=2)

# 优化器与学习率调度器
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
total_steps = len(train_dataloader) * 3  # 训练3个epoch
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# 微调训练循环
model.train()
for epoch in range(3):
    print(f"Epoch {epoch + 1} / 3")
    total_loss = 0
    for batch in tqdm(train_dataloader, desc="Training"):
        input_ids, attention_mask, targets = batch
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=targets)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
    avg_loss = total_loss / len(train_dataloader)
    print(f"Training loss: {avg_loss:.4f}")

# 验证
model.eval()
predictions, true_labels = [], []
with torch.no_grad():
    for batch in val_dataloader:
        input_ids, attention_mask, targets = batch
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
        predictions.extend(preds)
        true_labels.extend(targets)

# 计算准确率
accuracy = accuracy_score(true_labels, predictions)
print(f"Validation Accuracy: {accuracy:.4f}")


# 例7-7

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)

# 训练参数设置
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# 定义训练过程
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(logits, dim=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# 训练模型
trainer.train()

# 评估模型
results = trainer.evaluate()
print("Evaluation Results:", results)


# 例7-8
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import load_metric, load_dataset
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 加载模型和分词器
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
model = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=3)

# 加载多语言数据集
dataset = load_dataset("amazon_reviews_multi", "en")
metric_bleu = load_metric("bleu")
metric_f1 = load_metric("f1")

# 数据预处理
def preprocess_data(examples):
    inputs = tokenizer(examples["review_body"], padding="max_length", truncation=True, max_length=128)
    inputs["labels"] = examples["stars"] - 1  # 调整标签范围从0开始
    return inputs

encoded_dataset = dataset["train"].map(preprocess_data, batched=True)

# 分割数据集
train_dataset = encoded_dataset.select(range(200))
eval_dataset = encoded_dataset.select(range(200, 300))

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=8)

# 定义评估函数
def compute_metrics(predictions, labels):
    # F1 Score计算
    f1 = f1_score(labels, predictions, average="weighted")
    
    # BLEU分数计算
    references = [[tokenizer.decode(ref, skip_special_tokens=True)] for ref in labels]
    candidates = [tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
    bleu_score = sentence_bleu(references, candidates, smoothing_function=SmoothingFunction().method1)

    return {"f1_score": f1, "bleu_score": bleu_score}

# 训练参数设置
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
)

# 定义Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=lambda p: compute_metrics(torch.argmax(p.predictions, axis=1).cpu(), p.label_ids)
)

# 训练模型
trainer.train()

# 评估模型
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)



# 例7-9
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 加载XLM-RoBERTa模型和分词器
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# 测试分词器的效果
sample_text = "This is an example sentence."
inputs = tokenizer(sample_text, return_tensors="pt")
print(inputs)
运行结果如下：
{
  'input_ids': tensor([[0, 83, 101, 31, 113, 6030, 5284, 5, 2]]), 
  'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]])
}
3.微调模型
使用加载的数据集对模型进行微调。此处展示一个简单的训练过程，通常需要更复杂的训练配置。
from transformers import Trainer, TrainingArguments
import torch

# 准备训练数据
def preprocess_data(examples):
    return tokenizer(examples['premise'], examples['hypothesis'], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(preprocess_data, batched=True)
train_data = tokenized_datasets['train'].shuffle(seed=42).select(range(1000))  # 示例：仅选择部分数据进行微调
# 定义训练参数
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=1,
)
# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
)
# 开始训练
trainer.train()
运行结果如下：
***** Running training *****
  Num examples = 1000
  Num Epochs = 1
  Total optimization steps = 63
  ...
{'loss': 0.693, 'learning_rate': 1.8e-05, 'epoch': 1.0}





















