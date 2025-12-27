# 例5-1
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# 定义数据集
class SequenceDataset(Dataset):
    def __init__(self, sequences, labels, label_map):
        self.sequences = sequences
        self.labels = labels
        self.label_map = label_map

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = [self.label_map[l] for l in self.labels[idx]]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# 样本数据
sequences = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
labels = [["O", "B-ORG", "I-ORG", "O", "O"], ["B-PER", "I-PER", "O", "B-LOC", "I-LOC"]]
label_map = {"O": 0, "B-ORG": 1, "I-ORG": 2, "B-PER": 3, "I-PER": 4, "B-LOC": 5, "I-LOC": 6}

# 加载数据
dataset = SequenceDataset(sequences, labels, label_map)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 定义模型
class SimpleSequenceModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SimpleSequenceModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

# 初始化模型与参数
vocab_size = 11
embedding_dim = 8
hidden_dim = 16
output_dim = len(label_map)

model = SimpleSequenceModel(vocab_size, embedding_dim, hidden_dim, output_dim)

# 标签平滑
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (pred.size(-1) - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

# 定义损失和优化器
criterion = LabelSmoothingLoss(smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 3
for epoch in range(num_epochs):
    total_loss = 0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.view(-1, output_dim)
        targets = targets.view(-1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

# 测试模型输出
with torch.no_grad():
    for inputs, targets in dataloader:
        outputs = model(inputs)
        predicted_labels = torch.argmax(outputs, dim=-1)
        print("Predictions:", predicted_labels)
        print("Targets:", targets)


# 例5-2
import torch
import torch.nn as nn
from torchcrf import CRF

class LSTMCRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=128, hidden_dim=256):
        super(LSTMCRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, sentences, tags=None):
        embeddings = self.embedding(sentences)
        lstm_out, _ = self.lstm(embeddings)
        emissions = self.hidden2tag(lstm_out)

        if tags is not None:
            loss = -self.crf(emissions, tags, reduction='mean')
            return loss
        else:
            prediction = self.crf.decode(emissions)
            return prediction

# 模拟数据生成
vocab_size = 100  # 假设词汇表大小
tagset_size = 5   # 假设标签集大小，例如B、I、O等
model = LSTMCRF(vocab_size, tagset_size)

# 创建输入数据
sentences = torch.randint(0, vocab_size, (4, 10), dtype=torch.long)  # 4个句子，每个句子长度为10
tags = torch.randint(0, tagset_size, (4, 10), dtype=torch.long)       # 真实标签

# 训练模式下计算损失
loss = model(sentences, tags)
print("训练模式下的损失:", loss.item())

# 预测模式下生成序列标签
with torch.no_grad():
    predictions = model(sentences)
print("预测标签序列:", predictions)

# 例5-3
import torch
import torch.nn as nn

# 定义双向LSTM模型
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 双向LSTM，设置bidirectional=True
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        # 全连接层
        self.fc = nn.Linear(hidden_size * 2, output_size)  # 双向，因此维度为2倍hidden_size

    def forward(self, x):
        # 初始化LSTM的隐层状态和记忆单元状态，双向则num_layers需*2
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)
        # LSTM的输出
        out, _ = self.lstm(x, (h0, c0))
        # 仅保留最后时间步的输出
        out = self.fc(out[:, -1, :])
        return out

# 参数设置
input_size = 10  # 输入特征维度
hidden_size = 20  # 隐藏层维度
num_layers = 2  # LSTM层数
output_size = 5  # 输出类别数
model = BiLSTMModel(input_size, hidden_size, num_layers, output_size)

# 随机生成输入数据
batch_size = 3
seq_length = 7  # 序列长度
x = torch.randn(batch_size, seq_length, input_size)

# 模型输出
output = model(x)
print("模型输出：")
print(output)



# 例5-4
import torch
import torch.nn as nn

# 定义ELMo模型
class ELMo(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(ELMo, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 多层双向LSTM
        self.lstm_layers = nn.ModuleList(
            [nn.LSTM(input_size if i == 0 else hidden_size * 2, hidden_size, batch_first=True, bidirectional=True) 
             for i in range(num_layers)]
        )
        # 最后线性层，用于序列标注
        self.classifier = nn.Linear(hidden_size * 2 * num_layers, 3)  # 假设3类序列标签

    def forward(self, x):
        hidden_states = []
        for lstm in self.lstm_layers:
            h0 = torch.zeros(2, x.size(0), self.hidden_size)  # 双向LSTM需2倍的hidden_size
            c0 = torch.zeros(2, x.size(0), self.hidden_size)
            x, _ = lstm(x, (h0, c0))
            hidden_states.append(x)

        # 拼接每层LSTM的输出，形成最终的ELMo嵌入
        elmo_embedding = torch.cat(hidden_states, dim=2)
        # 通过分类层进行标签预测
        output = self.classifier(elmo_embedding)
        return output

# 参数设置
input_size = 10  # 输入特征维度
hidden_size = 20  # LSTM隐藏层维度
num_layers = 2  # LSTM层数
elmo_model = ELMo(input_size, hidden_size, num_layers)

# 生成模拟输入数据
batch_size = 2
seq_length = 5  # 序列长度
x = torch.randn(batch_size, seq_length, input_size)

# 模型输出
output = elmo_model(x)
print("模型输出：")
print(output)




# 例5-5
import torch
from transformers import BertTokenizer, BertModel

# 初始化BERT的预训练模型和分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# 输入文本，进行分词和编码
text = "John Doe lives in New York and works at OpenAI."
encoded_input = tokenizer(text, return_tensors="pt")
outputs = model(**encoded_input)

# 提取CLS标记的输出向量
cls_output = outputs.last_hidden_state[0, 0]

# 提取每个Token的输出向量（包含句子的上下文信息）
token_outputs = outputs.last_hidden_state[0]

# 输出各部分向量的尺寸以确保正确的形状
print("CLS Output Shape:", cls_output.shape)
print("Token Outputs Shape:", token_outputs.shape)

# 进一步分析每个Token的嵌入向量
token_embeddings = {}
for i, token_id in enumerate(encoded_input["input_ids"][0]):
    token = tokenizer.convert_ids_to_tokens(token_id)
    token_embeddings[token] = token_outputs[i].detach().numpy()
    print(f"Token: {token}, Embedding: {token_outputs[i][:5]}...")

# 输出示例，验证各Token向量与CLS向量的差异
print("CLS Vector (first 5 elements):", cls_output[:5])
print("Sample Token Vector - 'John' (first 5 elements):", token_embeddings[



# 例5-6

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset, load_metric

# 加载数据集和分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
dataset = load_dataset("conll2003")  # 经典的NER数据集

# 标签字典
label_list = dataset["train"].features["ner_tags"].feature.names

# 数据预处理：将文本编码为BERT输入格式
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = [-100 if word_id is None else label[word_id] for word_id in word_ids]
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# 加载预训练的BERT模型，设置分类层输出大小为NER标签数目
model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(label_list))

# 训练设置
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# 评估方法
metric = load_metric("seqeval")

# 计算F1、精确度和召回率
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [[label_list[l] for l in label if l != -100] for label in labels]
    true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                        for prediction, label in zip(predictions, labels)]
    
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# 使用Trainer进行训练和评估
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
)

# 开始训练
trainer.train()

# 评估模型
eval_results = trainer.evaluate()

# 打印最终结果
print("Evaluation results:", eval_results)


# 例5-7
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# 模拟NER标签
true_labels = [
    ["O", "B-PER", "I-PER", "O", "B-LOC", "O"],
    ["O", "B-ORG", "I-ORG", "O", "B-LOC", "O"],
    ["O", "O", "B-PER", "I-PER", "O", "O"]
]

pred_labels = [
    ["O", "B-PER", "I-PER", "O", "B-LOC", "O"],
    ["O", "B-ORG", "O", "O", "B-LOC", "O"],
    ["O", "O", "B-PER", "O", "O", "O"]
]

# 将标签展平为一维数组，便于计算
true_flat = [label for seq in true_labels for label in seq]
pred_flat = [label for seq in pred_labels for label in seq]

# 计算每个类别的精确度、召回率、F1得分和准确率
precision, recall, f1, _ = precision_recall_fscore_support(true_flat, pred_flat, average='macro')
accuracy = accuracy_score(true_flat, pred_flat)

print("准确率:", accuracy)
print("精确度:", precision)
print("召回率:", recall)
print("F1得分:", f1)

# 例5-8
import numpy as np
from sklearn.metrics import classification_report

# 模拟的真实标签和预测标签数据集
true_labels = [
    ["O", "B-PER", "I-PER", "O", "B-LOC", "O"],
    ["O", "B-ORG", "I-ORG", "O", "B-LOC", "O"],
    ["O", "O", "B-PER", "I-PER", "O", "O"]
]

pred_labels = [
    ["O", "B-PER", "I-PER", "O", "O", "O"],
    ["O", "B-ORG", "O", "O", "B-LOC", "O"],
    ["O", "O", "B-PER", "O", "O", "O"]
]

# 将嵌套列表展平成一维列表
true_flat = [label for seq in true_labels for label in seq]
pred_flat = [label for seq in pred_labels for label in seq]

# 使用classification_report计算每个实体类别的评估指标
report = classification_report(true_flat, pred_flat, labels=["B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC"], zero_division=0)

print("各类实体的性能评估:")
print(report)



import pandas as pd
import re

# 构建领域特定词典
# 假设构建一个包含地名和人名的词典
gazetteers = {
    "Location": ["New York", "California", "Texas", "Beijing", "Shanghai", "Paris", "Tokyo"],
    "Person": ["Alice", "Bob", "Charlie", "David", "Eve"]
}

# 将词典数据转换为DataFrame便于管理和查找
gazetteer_df = pd.DataFrame([(key, value) for key in gazetteers for value in gazetteers[key]], columns=["Entity_Type", "Entity"])

# 显示词典内容
print("构建的词典内容：")
print(gazetteer_df)

# 定义待处理的文本
text = "Alice and Bob traveled from New York to California last week, while Charlie stayed in Paris."

# 通过词典匹配找到实体
def find_entities(text, gazetteer_df):
    entities = []
    for _, row in gazetteer_df.iterrows():
        # 使用正则表达式查找词典中每个实体
        matches = re.finditer(rf"\b{re.escape(row['Entity'])}\b", text)
        for match in matches:
            entity_info = {
                "Entity": match.group(0),
                "Type": row["Entity_Type"],
                "Start": match.start(),
                "End": match.end()
            }
            entities.append(entity_info)
    return entities

# 查找并输出文本中的实体
entities_found = find_entities(text, gazetteer_df)
print("\n在文本中找到的实体：")
for entity in entities_found:
    print(f"实体: {entity['Entity']} | 类型: {entity['Type']} | 开始位置: {entity['Start']} | 结束位置: {entity['End']}")

。
# 示例NER预测结果 (假设)
ner_predictions = [
    {"Entity": "Alice", "Type": "Person", "Start": 0, "End": 5},
    {"Entity": "California", "Type": "Location", "Start": 38, "End": 48}
]

# 合并词典匹配结果与NER预测结果
def integrate_gazetteer_with_ner(ner_predictions, gazetteer_entities):
    combined_results = {tuple((item['Start'], item['End'])): item for item in ner_predictions}
    
    # 合并词典匹配结果
    for entity in gazetteer_entities:
        key = (entity["Start"], entity["End"])
        if key not in combined_results:
            combined_results[key] = entity

    # 返回合并后的结果
    return list(combined_results.values())

# 结合词典和模型结果
combined_results = integrate_gazetteer_with_ner(ner_predictions, entities_found)

# 显示最终的NER和词典匹配结果
print("\n结合词典和NER模型的实体识别结果：")
for result in combined_results:
    print(f"实体: {result['Entity']} | 类型: {result['Type']} | 开始位置: {result['Start']} | 结束位置: {result['End']}")





# 例5-10
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset
import numpy as np

# 定义领域词典
gazetteer = {
    "location": ["Paris", "New York", "Beijing", "London"],
    "organization": ["Google", "Microsoft", "OpenAI", "Tesla"],
    "person": ["John", "Alice", "Bob", "Charlie"]
}

# 数据集示例
class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=64)
        return encoding, text

# 模型定义
class NERModel(nn.Module):
    def __init__(self):
        super(NERModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Linear(self.bert.config.hidden_size, 3)  # 假设3类：location, organization, person
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output)
        return logits

# 将词典信息结合到NER模型中
def integrate_gazetteer(tokens, gazetteer):
    gazetteer_mask = torch.zeros(len(tokens))
    for idx, token in enumerate(tokens):
        for entity_type, words in gazetteer.items():
            if token in words:
                gazetteer_mask[idx] = list(gazetteer.keys()).index(entity_type) + 1
    return gazetteer_mask

# 创建数据集
texts = ["John works at Google in New York.", "Alice visited Paris last year."]
dataset = TextDataset(texts)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# 初始化模型
model = NERModel()
model.eval()

# 处理每个批次
for batch in dataloader:
    encoding, text = batch
    input_ids = encoding["input_ids"].squeeze(1)
    attention_mask = encoding["attention_mask"].squeeze(1)
    tokens = [model.bert.config.id2token[idx] for idx in input_ids[0].tolist()]

    # 将词典信息整合到输入中
    gazetteer_mask = integrate_gazetteer(tokens, gazetteer)
    logits = model(input_ids, attention_mask)

    # 输出类别预测结果
    predictions = torch.argmax(logits, dim=-1).squeeze().tolist()
    
    # 输出最终预测结果
    result = []
    for token, pred, gaz_mask in zip(tokens, predictions, gazetteer_mask):
        if pred != 0 or gaz_mask != 0:
            result.append((token, "Entity" if pred != 0 else "Dictionary match"))

    print(f"Text: {text[0]}")
    print("Predictions:", result)





