# 例3-1
import re
from typing import List, Dict

class RuleBasedClassifier:
    def __init__(self):
        self.rules = []

    def add_rule(self, label: str, pattern: str, sub_rules=None):
        compiled_pattern = re.compile(pattern)
        self.rules.append({"label": label, "pattern": compiled_pattern, "sub_rules": sub_rules or []})

    def classify(self, text: str) -> List[str]:
        labels = []
        for rule in self.rules:
            if rule["pattern"].search(text):
                labels.append(rule["label"])
                for sub_rule in rule["sub_rules"]:
                    if sub_rule["pattern"].search(text):
                        labels.append(sub_rule["label"])
        return labels

# 定义关键词规则
classifier = RuleBasedClassifier()
classifier.add_rule("科技", r"\b(技术|人工智能|大数据|机器学习)\b")
classifier.add_rule("财经", r"\b(股票|市场|投资|金融)\b", [
    {"label": "股市", "pattern": re.compile(r"\b(股票|证券|股市)\b")},
    {"label": "外汇", "pattern": re.compile(r"\b(汇率|外汇|美元)\b")}
])
classifier.add_rule("体育", r"\b(足球|篮球|网球|比赛|运动会)\b", [
    {"label": "足球", "pattern": re.compile(r"\b(足球|射门|进球)\b")},
    {"label": "篮球", "pattern": re.compile(r"\b(篮球|扣篮|三分球)\b")}
])

# 测试文本数据
texts = [
    "今天的科技进步让人工智能和大数据迅速发展",
    "最新的股票市场分析表明，投资机会很大",
    "昨天的足球比赛非常激烈，进球数很多",
    "汇率波动影响了外汇投资者的决策",
    "篮球比赛中扣篮得分引发观众热烈欢呼"
]

# 执行分类
for text in texts:
    labels = classifier.classify(text)
    print(f"文本: '{text}' -> 分类: {labels}")




from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import gensim.downloader as api

# 数据准备
texts = [
    "人工智能在金融市场中的应用日益广泛",
    "足球和篮球是全球最受欢迎的运动",
    "股市波动对投资者有很大影响",
    "机器学习和数据科学的结合推动了科技创新",
    "外汇市场的变化使得投资回报更具挑战性",
    "网球比赛十分激烈，场上选手表现出色",
    "数据分析在金融行业中非常重要",
    "篮球赛事中三分球表现出色"
]
labels = ["科技", "体育", "财经", "科技", "财经", "体育", "财经", "体育"]

# 划分训练和测试数据集
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.25, random_state=42)

# 定义TF-IDF矢量化器和SVM分类器的管道
tfidf_vectorizer = TfidfVectorizer()
svm_model = SVC(kernel='linear', C=1.0)
pipeline = Pipeline([('tfidf', tfidf_vectorizer), ('svm', svm_model)])

# 训练并评估TF-IDF + SVM
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print("TF-IDF + SVM分类结果")
print("准确率:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 定义Word2Vec嵌入生成函数
def get_word2vec_embeddings(texts, embedding_model):
    embeddings = []
    for text in texts:
        words = text.split()
        word_vectors = [embedding_model[word] for word in words if word in embedding_model]
        if word_vectors:
            text_embedding = np.mean(word_vectors, axis=0)  # 取平均作为文本的嵌入表示
        else:
            text_embedding = np.zeros(embedding_model.vector_size)
        embeddings.append(text_embedding)
    return np.array(embeddings)

# 下载预训练的Word2Vec模型
embedding_model = api.load("glove-wiki-gigaword-50")  # 使用GloVe词向量

# 生成Word2Vec嵌入
X_train_w2v = get_word2vec_embeddings(X_train, embedding_model)
X_test_w2v = get_word2vec_embeddings(X_test, embedding_model)

# 训练逻辑回归模型并评估
lr_model = LogisticRegression()
lr_model.fit(X_train_w2v, y_train)
y_pred_w2v = lr_model.predict(X_test_w2v)
print("\nWord2Vec + 逻辑回归分类结果")
print("准确率:", accuracy_score(y_test, y_pred_w2v))
print(classification_report(y_test, y_pred_w2v))




# 例3-2
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 数据准备
texts = [
    "人工智能在金融市场中的应用日益广泛",
    "足球和篮球是全球最受欢迎的运动",
    "股市波动对投资者有很大影响",
    "机器学习和数据科学的结合推动了科技创新",
    "外汇市场的变化使得投资回报更具挑战性",
    "网球比赛十分激烈，场上选手表现出色",
    "数据分析在金融行业中非常重要",
    "篮球赛事中三分球表现出色"
]
labels = [1, 0, 1, 1, 1, 0, 1, 0]  # 1表示财经，0表示体育

# 分割训练和测试集
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.25, random_state=42)

# 自定义数据集
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return encoding["input_ids"].squeeze(), encoding["attention_mask"].squeeze(), label

# 加载预训练的BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 数据加载器
train_dataset = TextDataset(train_texts, train_labels, tokenizer)
test_dataset = TextDataset(test_texts, test_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2)

# 训练设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# 训练模型
def train_model(model, data_loader, optimizer, criterion, device):
    model.train()
    for input_ids, attention_mask, labels in data_loader:
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"训练损失: {loss.item()}")

# 测试模型
def evaluate_model(model, data_loader, device):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for input_ids, attention_mask, labels in data_loader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    accuracy = np.mean(np.array(predictions) == np.array(true_labels))
    print(f"测试准确率: {accuracy}")

# 执行训练和评估
train_model(model, train_loader, optimizer, criterion, device)
evaluate_model(model, test_loader, device)



from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 示例数据（包含二分类和多分类的标签）
texts = [
    "金融市场有很大波动", "科技公司股价上涨", "体育赛事热度高", 
    "旅游行业恢复", "股市持续下跌", "新的电影上映吸引大量观众", 
    "天气变化较大", "电影票房破纪录", "新科技成果引发关注", 
    "环保问题引起重视"
]
binary_labels = [1, 1, 0, 0, 1, 0, 0, 0, 1, 0]  # 1表示财经类，0表示非财经类
multi_labels = [0, 0, 1, 2, 0, 1, 2, 1, 0, 2]   # 0:财经，1:娱乐，2:其他

# 分割数据集
train_texts, test_texts, train_bin_labels, test_bin_labels, train_multi_labels, test_multi_labels = train_test_split(
    texts, binary_labels, multi_labels, test_size=0.25, random_state=42
)

# 自定义数据集
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return encoding["input_ids"].squeeze(), encoding["attention_mask"].squeeze(), label

# 加载BERT分词器和模型
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 二分类模型设置
binary_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
multi_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

# 数据加载器
binary_train_dataset = TextDataset(train_texts, train_bin_labels, tokenizer)
binary_test_dataset = TextDataset(test_texts, test_bin_labels, tokenizer)
binary_train_loader = DataLoader(binary_train_dataset, batch_size=2, shuffle=True)
binary_test_loader = DataLoader(binary_test_dataset, batch_size=2)

multi_train_dataset = TextDataset(train_texts, train_multi_labels, tokenizer)
multi_test_dataset = TextDataset(test_texts, test_multi_labels, tokenizer)
multi_train_loader = DataLoader(multi_train_dataset, batch_size=2, shuffle=True)
multi_test_loader = DataLoader(multi_test_dataset, batch_size=2)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
binary_model, multi_model = binary_model.to(device), multi_model.to(device)
optimizer = optim.AdamW(binary_model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# 训练函数
def train_model(model, data_loader, optimizer, criterion, device):
    model.train()
    for input_ids, attention_mask, labels in data_loader:
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"训练损失: {loss.item()}")

# 评估函数
def evaluate_model(model, data_loader, device):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for input_ids, attention_mask, labels in data_loader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(true_labels, predictions)
    print(f"测试准确率: {accuracy}")

# 二分类任务训练和评估
print("二分类任务：")
train_model(binary_model, binary_train_loader, optimizer, criterion, device)
evaluate_model(binary_model, binary_test_loader, device)

# 多分类任务训练和评估
print("\n多分类任务：")
optimizer = optim.AdamW(multi_model.parameters(), lr=2e-5)
train_model(multi_model, multi_train_loader, optimizer, criterion, device)
evaluate_model(multi_model, multi_test_loader, device)


# 例3-3
import torch
import re

# 定义自定义数据集类
class CustomTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        # 数据清洗
        text = re.sub(r"[^\w\s]", "", text).lower()
        # 将文本转换为 BERT 输入格式
        encoding = self.tokenizer(
            text, 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {"input_ids": encoding["input_ids"].flatten(), 
                "attention_mask": encoding["attention_mask"].flatten(), 
                "label": torch.tensor(label, dtype=torch.long)}

# 加载 SST-2 数据集并提取训练集
dataset = load_dataset("glue", "sst2")
train_texts = dataset["train"]["sentence"]
train_labels = dataset["train"]["label"]

# 初始化 BERT 分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 创建自定义数据集
train_dataset = CustomTextDataset(train_texts, train_labels, tokenizer)

# 定义 DataLoader
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 检查 DataLoader 中的一个批次
for batch in train_loader:
    print("Input IDs:\n", batch["input_ids"])
    print("\nAttention Mask:\n", batch["attention_mask"])
    print("\nLabels:\n", batch["label"])
    break

# 例3-4
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
import re

# 数据集类定义
class SampleDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = re.sub(r"[^\w\s]", "", self.texts[idx]).lower()
        encoding = self.tokenizer(
            text, 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# 加载预训练的BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 冻结前几层参数
for name, param in model.named_parameters():
    if "encoder.layer.0" in name or "encoder.layer.1" in name or "encoder.layer.2" in name:
        param.requires_grad = False

# 模拟数据
texts = ["This is a positive example", "This is a negative example"] * 100
labels = [1, 0] * 100

# 数据加载
train_dataset = SampleDataset(texts, labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 优化器和学习率调度器
optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = StepLR(optimizer, step_size=1, gamma=0.9)

# 训练代码
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

for epoch in range(3):  # 假设训练3个epoch
    for batch in train_loader:
        optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # 前向传播
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        
        # 更新权重
        optimizer.step()
    
    # 每个epoch结束时更新学习率
    scheduler.step()
    print(f"Epoch {epoch + 1} - Loss: {loss.item():.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

# 例3-4
import torch
from transformers import BertForSequenceClassification, BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import re

# 自定义数据集
class SampleDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = re.sub(r"[^\w\s]", "", self.texts[idx]).lower()
        encoding = self.tokenizer(
            text, 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# 加载BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 数据准备
texts = ["This is a positive example", "This is a negative example"] * 100
labels = [1, 0] * 100

# 加载数据集
train_dataset = SampleDataset(texts, labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 设置优化器和调度器
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * 3  # 假设3个epoch
warmup_steps = int(0.1 * total_steps)  # 10% 的step用于warmup
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

# 训练过程
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

for epoch in range(3):  # 假设训练3个epoch
    for batch in train_loader:
        optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # 前向传播
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        
        # 更新权重
        optimizer.step()
        
        # 更新学习率
        scheduler.step()
    
    print(f"Epoch {epoch + 1} - Loss: {loss.item():.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")







