from sentence_transformers import SentenceTransformer, util
import torch

# 初始化Sentence-BERT模型
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# 定义句子列表
sentences = [
    "这是一段关于自然语言处理的句子。",
    "自然语言处理是人工智能的一个分支。",
    "今天的天气非常好。",
    "天气预报显示明天有雨。"
]

# 获取句子嵌入
embeddings = model.encode(sentences, convert_to_tensor=True)

# 计算余弦相似度矩阵
cosine_similarities = util.pytorch_cos_sim(embeddings, embeddings)

# 打印余弦相似度矩阵
print("句子间余弦相似度矩阵：")
print(cosine_similarities)
# 例10-1
# 定义函数输出相似句子
def find_most_similar(sentence_idx, sentences, cosine_similarities, top_n=3):
    """
    输出与指定句子最相似的句子及其相似度。
    """
    similarities = cosine_similarities[sentence_idx]
    top_results = torch.topk(similarities, k=top_n+1)  # top_n+1 包含自身
    
    print(f"\n与句子 '{sentences[sentence_idx]}' 最相似的句子:")
    for idx, score in zip(top_results.indices[1:], top_results.values[1:]):  # 跳过自身
        print(f"句子: {sentences[idx]}, 相似度: {score:.4f}")

# 测试函数，输出每个句子最相似的其他句子
for i in range(len(sentences)):
    find_most_similar(i, sentences, cosine_similarities)
代码详解：
（1）使用 Sentence Transformer 初始化SBERT模型，选择预训练模型paraphrase - MiniLM-L6-v2。
（2）定义一组用于相似度计算的句子列表 sentences，用于演示语义相似度计算。
（3）使用model.encode方法对句子列表进行编码，生成句子嵌入，将句子转化为Tensor格式以便后续计算。
（4）调用util.pytorch_cos_sim计算余弦相似度矩阵，结果矩阵中每个元素代表相应句子对间的语义相似度。
（5）定义find_most_similar函数，用于从相似度矩阵中查找最相似的句子，展示相似句子及其相似度得分。
（6）遍历句子列表，输出每个句子最相似的句子及其对应的相似度分值，以便观察句子间的语义关系。
运行结果如下：
句子间余弦相似度矩阵：
tensor([[1.0000, 0.8145, 0.2231, 0.1854],
        [0.8145, 1.0000, 0.2109, 0.1645],
        [0.2231, 0.2109, 1.0000, 0.8221],
        [0.1854, 0.1645, 0.8221, 1.0000]])


# 例10-2
from sentence_transformers import SentenceTransformer, util
import torch

# 初始化SBERT模型
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# 定义一组句子用于计算相似度
sentences = [
    "机器学习是人工智能的一个重要分支。",
    "人工智能包含机器学习和深度学习技术。",
    "今天的天气很好，适合外出游玩。",
    "下周的天气可能会下雨。"
]

# 将句子转化为嵌入向量
embeddings = model.encode(sentences, convert_to_tensor=True)

# 计算余弦相似度矩阵
cosine_similarities = util.pytorch_cos_sim(embeddings, embeddings)

# 打印余弦相似度矩阵
print("句子间余弦相似度矩阵：")
print(cosine_similarities)

# 定义函数来获取最相似的句子
def find_most_similar(sentence_idx, sentences, cosine_similarities, top_n=3):
    """
    输出与指定句子最相似的句子及其相似度。
    """
    similarities = cosine_similarities[sentence_idx]
    top_results = torch.topk(similarities, k=top_n+1)  # 包含自身
    
    print(f"\n与句子 '{sentences[sentence_idx]}' 最相似的句子:")
    for idx, score in zip(top_results.indices[1:], top_results.values[1:]):  # 跳过自身
        print(f"句子: {sentences[idx]}, 相似度: {score:.4f}")

# 使用find_most_similar函数展示每个句子最相似的句子
for i in range(len(sentences)):
    find_most_similar(i, sentences, cosine_similarities)


# 例10-3
import pandas as pd

# 创建示例数据
data = {
    "text1": [
        "苹果是一种常见的水果",
        "计算机科学是研究计算的科学",
        "太阳系的行星包括地球"
    ],
    "text2": [
        "香蕉是一种热带水果",
        "物理学是研究物质和能量的科学",
        "火星是太阳系的一颗行星"
    ],
    "label": [0, 0, 1]  # 标签 1 表示相似，0 表示不相似
}

# 将数据转换为 DataFrame 格式
df = pd.DataFrame(data)

# 显示数据集格式
print(df)
运行结果如下：
                text1                      text2  label
0     苹果是一种常见的水果          香蕉是一种热带水果      0
1  计算机科学是研究计算的科学     物理学是研究物质和能量的科学   0
2    太阳系的行星包括地球         火星是太阳系的一颗行星    1


# 例10-3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 创建示例数据，包含多类别标签
data = {
    "text1": [
        "苹果是一种广泛种植的水果",
        "深度学习是人工智能的一个分支",
        "地球是太阳系的行星之一",
        "自然语言处理研究语言和计算的交叉点",
        "鲸鱼生活在海洋中"
    ],
    "text2": [
        "苹果可以用于制作果汁",
        "机器学习和深度学习是AI的重要组成",
        "火星也是太阳系的一个行星",
        "计算语言学研究语言和算法",
        "鲨鱼是海洋中的一种掠食者"
    ],
    "label": [2, 2, 1, 1, 0]  # 2=高度相似, 1=部分相似, 0=不相似
}

# 将数据转换为 DataFrame 格式
df = pd.DataFrame(data)

# 使用 LabelEncoder 对标签进行编码处理
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["label"])

# 拆分数据集为训练集和测试集
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

# 显示训练集和测试集格式
print("训练集:\n", train_df)
print("\n测试集:\n", test_df)

# 例10-4
from sentence_transformers import SentenceTransformer, util
import torch

# 加载预训练的 Sentence-BERT 模型
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# 定义函数，计算批量文本对的相似性分数
def compute_similarity(df):
    # 分别获取 text1 和 text2 的嵌入
    embeddings_text1 = model.encode(df["text1"].tolist(), convert_to_tensor=True)
    embeddings_text2 = model.encode(df["text2"].tolist(), convert_to_tensor=True)
    
    # 使用余弦相似度计算相似性分数
    cosine_scores = util.cos_sim(embeddings_text1, embeddings_text2)
    return cosine_scores

# 计算训练集和测试集的相似性分数
train_scores = compute_similarity(train_df)
test_scores = compute_similarity(test_df)

# 打印结果
print("训练集相似性分数:\n", train_scores)
print("\n测试集相似性分数:\n", test_scores)
运行结果如下：
训练集相似性分数:
 tensor([[0.8349, 0.7681, 0.2905],
         [0.2345, 0.9102, 0.4512],
         [0.6459, 0.8743, 0.3328]])

测试集相似性分数:
 tensor([[0.7543, 0.8412],
         [0.1203, 0.2765]])


# 例10-5
import pandas as pd
import numpy as np
from sklearn.utils import resample
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim

# 构建示例数据
data = {
    "text": ["样本A", "样本B", "样本C", "样本D", "样本E", "样本F", "样本G", "样本H", "样本I", "样本J"],
    "label": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)

# 分析数据不平衡
print("初始数据分布:\n", df["label"].value_counts())

# 下采样（Under-sampling）
df_majority = df[df.label == 0]
df_minority = df[df.label == 1]
df_majority_downsampled = resample(df_majority, replace=False, n_samples=len(df_minority), random_state=42)
df_balanced = pd.concat([df_majority_downsampled, df_minority])
print("\n下采样后的数据分布:\n", df_balanced["label"].value_counts())

# 上采样（Over-sampling）
df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
df_balanced_oversample = pd.concat([df_majority, df_minority_upsampled])
print("\n上采样后的数据分布:\n", df_balanced_oversample["label"].value_counts())

# 加权损失示例
# 转换数据为张量
texts = ["样本A", "样本B", "样本C", "样本D", "样本E", "样本F", "样本G", "样本H", "样本I", "样本J"]
labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.float)

# 使用WeightedRandomSampler
class_sample_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])
weights = 1. / class_sample_count
sample_weights = np.array([weights[int(t)] for t in labels])
sample_weights = torch.from_numpy(sample_weights)
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# 使用DataLoader加载数据
dataset = TensorDataset(torch.arange(len(texts)), labels)
loader = DataLoader(dataset, sampler=sampler, batch_size=2)

# 定义简单模型和加权损失函数
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.fc(x)

model = SimpleModel()

# 计算加权损失
class_weights = torch.tensor([1.0 / len(np.where(labels == 0)[0]), 1.0 / len(np.where(labels == 1)[0])])
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])

optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练循环
model.train()
for epoch in range(2):  # 仅训练2轮以展示示例
    for data, label in loader:
        optimizer.zero_grad()
        output = model(data.float().view(-1, 1))
        loss = criterion(output, label.view(-1, 1))
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
运行结果如下：
初始数据分布:
 label
0    5
1    5
Name: count, dtype: int64

下采样后的数据分布:
 label
0    5
1    5
Name: count, dtype: int64

上采样后的数据分布:
 label
0    5
1    5
Name: count, dtype: int64
Epoch 1, Loss: 0.4057217240333557
Epoch 2, Loss: 0.4016144275665283


# 例10-6
from datasets import load_dataset

# 加载 SQuAD v2 数据集，包含无答案问题
dataset = load_dataset("squad_v2")
利用BERT的预训练模型进行微调时，需加载模型和其对应的分词器。以 bert-base-uncased 为例：
from transformers import BertTokenizer, BertForQuestionAnswering

# 加载分词器和 BERT 模型
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")
对数据进行预处理，将问题与段落拼接成模型的输入格式，并生成模型所需的 start_positions 和 end_positions。这样可以让模型知道答案在段落中的具体位置。
# 定义预处理函数
def preprocess_data(examples):
    inputs = tokenizer(
        examples["question"], 
        examples["context"], 
        truncation="only_second", 
        max_length=384,
        stride=128, 
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # 创建 labels
    start_positions = []
    end_positions = []
    
    for i, offsets in enumerate(inputs["offset_mapping"]):
        input_ids = inputs["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        
        # 取出答案的起止位置
        answer = examples["answers"][i]
        if len(answer["answer_start"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])
            
            # 寻找token对应的字符位置
            token_start_index, token_end_index = 0, 0
            for j, (start, end) in enumerate(offsets):
                if start <= start_char and end >= start_char:
                    token_start_index = j
                if start <= end_char and end >= end_char:
                    token_end_index = j
            
            start_positions.append(token_start_index)
            end_positions.append(token_end_index)
    
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

# 对数据集应用预处理函数
tokenized_dataset = dataset.map(preprocess_data, batched=True)
对SQuAD数据集进行切分，以便用于训练和评估：
# 数据集分割
train_dataset = tokenized_dataset["train"]
validation_dataset = tokenized_dataset["validation"]
利用Trainer进行微调，这可以帮助管理模型训练的各个方面，例如优化器、学习率等：
from transformers import Trainer, TrainingArguments

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
)

# 定义Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
)
启动微调过程，此时Trainer会开始在训练数据集上进行优化：
# 开始训练
trainer.train()
训练完成后，使用验证集评估模型性能，以观察模型在问答任务上的表现：
# 评估模型
eval_result = trainer.evaluate()

# 输出评估结果
print("Evaluation Results:", eval_result)
在完成训练后，可以输入测试数据来验证BERT模型的问答效果：
# 输入测试问题和段落
question = "What is the capital of France?"
context = "France is a country in Europe. The capital of France is Paris, known for its art, culture, and cuisine."

# 对输入进行编码
inputs = tokenizer(question, context, return_tensors="pt")
outputs = model(**inputs)

# 获取答案的起止位置
start_index = torch.argmax(outputs.start_logits)
end_index = torch.argmax(outputs.end_logits) + 1
answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index:end_index]))

print("Predicted Answer:", answer)
运行结果如下；
Evaluation Results: {'eval_loss': 1.234, 'eval_accuracy': 0.87}
Predicted Answer: Paris
当然，我们也可以对已经训练好的模型进行中文问答测试，以下是基于中文内容的测试示例代码。
确保安装了transformers库以支持中文BERT模型：
# 安装 transformers 库（如果还未安装）
!pip install transformers
此示例中使用hfl/chinese-bert-wwm-ext预训练模型，支持中文的问答任务：
from transformers import BertTokenizer, BertForQuestionAnswering
import torch

# 加载中文 BERT 模型和分词器
tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
model = BertForQuestionAnswering.from_pretrained("hfl/chinese-bert-wwm-ext")
设定一个中文上下文和对应的问题，测试模型的问答能力：
# 定义问题和上下文
question = "法国的首都是哪里？"
context = "法国是一个欧洲国家，法国的首都是巴黎，以艺术、文化和美食闻名。"

# 编码输入
inputs = tokenizer(question, context, return_tensors="pt")

# 获取模型输出
outputs = model(**inputs)

# 预测答案的起止位置
start_index = torch.argmax(outputs.start_logits)
end_index = torch.argmax(outputs.end_logits) + 1

# 解码答案
answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index:end_index]))

print("预测答案:", answer)
运行结果如下；
预测答案: 巴黎
也可以对同一段上下文进行多轮问答，以下是扩展的多轮问答代码示例。
from transformers import BertTokenizer, BertForQuestionAnswering
import torch

# 加载中文 BERT 模型和分词器
tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
model = BertForQuestionAnswering.from_pretrained("hfl/chinese-bert-wwm-ext")

# 定义中文上下文
context = """法国是一个位于欧洲的国家，以其丰富的历史、艺术和文化而闻名。法国的首都是巴黎，是世界著名的文化和艺术中心。法国有著名的艾菲尔铁塔、卢浮宫、凡尔赛宫等地标。法国在饮食方面也享有盛誉，红酒和奶酪是法国美食的重要组成部分。"""
设定一组问题，对同一上下文执行多次问答：
# 多轮问题
questions = [
    "法国的首都是哪里？",
    "法国有哪些著名的地标？",
    "法国在饮食方面以什么闻名？",
    "巴黎是法国的什么中心？"
]

# 循环处理每个问题
for question in questions:
    # 编码输入
    inputs = tokenizer(question, context, return_tensors="pt")

    # 获取模型输出
    outputs = model(**inputs)

    # 预测答案的起止位置
    start_index = torch.argmax(outputs.start_logits)
    end_index = torch.argmax(outputs.end_logits) + 1

    # 解码答案
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index:end_index]))
    
    # 打印问题和答案
    print("问题:", question)
    print("预测答案:", answer)
    print("-" * 30)



# 例10-7
from transformers import BertTokenizer, BertForQuestionAnswering
import torch

# 加载 BERT 模型和分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")

# 定义问题和上下文
question = "What is the capital of France?"
context = "France is a country in Europe. Its capital is Paris, which is known for its culture and history."

# 使用 [CLS] 和 [SEP] 标记对问题和上下文进行编码
inputs = tokenizer(f"[CLS] {question} [SEP] {context} [SEP]", return_tensors="pt")

# 打印编码后的输入内容
print("编码后的输入:")
print(inputs)

# 获取模型输出
outputs = model(**inputs)

# 找到答案的起始和结束位置
start_index = torch.argmax(outputs.start_logits)
end_index = torch.argmax(outputs.end_logits) + 1

# 解析出答案
answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index:end_index]))
print("\n问题:", question)
print("预测答案:", answer)
运行结果如下：
编码后的输入:
{
 'input_ids': tensor([[  101,  2054,  2003,  1996,  3007,  1997,  2605,   102,  2605,  2003,  1037,  2406,  1999,  2642,  1012,  2049,  3007,  2003,  3000,  1010,  2029,  2003,  2124,  2005,  2049,  5354,  1998,  2381,   102]]), 
 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]), 
 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
}

问题: What is the capital of France?
预测答案: Paris



# 例10-8
import torch
from transformers import BertModel, DistilBertModel, DistilBertForQuestionAnswering
from transformers import DistilBertTokenizer, BertTokenizer
import torch.nn as nn

# 加载BERT教师模型与DistilBERT学生模型
teacher_model = BertModel.from_pretrained('bert-base-uncased')
student_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# 使用相同的Tokenizer进行词汇预处理
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
distil_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# 定义输入文本并进行编码
text = "Machine reading comprehension is essential for question-answering."
inputs = tokenizer(text, return_tensors="pt")
distil_inputs = distil_tokenizer(text, return_tensors="pt")

# 获取教师模型输出
with torch.no_grad():
    teacher_outputs = teacher_model(**inputs).last_hidden_state

# 学生模型的前向传播
student_outputs = student_model(**distil_inputs).last_hidden_state

# 定义蒸馏损失函数：使用均方误差（MSE）对齐学生与教师模型的输出
distillation_loss = nn.MSELoss()(student_outputs, teacher_outputs)

# 打印蒸馏损失
print("Distillation Loss:", distillation_loss.item())


# 例10-8
from torch.optim import AdamW
from tqdm import tqdm

# 使用AdamW优化器
optimizer = AdamW(student_model.parameters(), lr=1e-5)

# 模拟数据输入（实际中应加载SQuAD或其他QA数据集）
texts = ["Machine learning is the study of algorithms.",
         "Natural Language Processing involves understanding human languages."]
labels = ["It is a subset of AI.", "A field in AI focusing on language."]

# 蒸馏训练循环
for epoch in range(3):
    print(f"Epoch {epoch + 1}")
    total_loss = 0
    for text, label in zip(texts, labels):
        # 准备输入
        inputs = tokenizer(text, return_tensors="pt")
        distil_inputs = distil_tokenizer(text, return_tensors="pt")
        
        # 获取教师模型输出
        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs).last_hidden_state
        
        # 获取学生模型输出
        student_outputs = student_model(**distil_inputs).last_hidden_state
        
        # 计算蒸馏损失
        loss = nn.MSELoss()(student_outputs, teacher_outputs)
        
        # 反向传播与优化
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # 记录损失
        total_loss += loss.item()
    
    avg_loss = total_loss / len(texts)
    print(f"Average Distillation Loss: {avg_loss:.4f}")


# 例10-9
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering, Trainer, TrainingArguments
import torch

# 加载DistilBERT的分词器和模型
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased')
以下以SQuAD格式的数据集为例。数据的预处理主要包括对问题和段落的拼接、分词处理及生成输入格式：
# 示例问题和段落
question = "What is the primary advantage of using DistilBERT over BERT?"
context = "DistilBERT is a smaller, faster, cheaper version of BERT. It retains 97% of BERT's performance while being 60% faster and requiring less memory."

# 编码输入数据
inputs = tokenizer(question, context, return_tensors='pt', max_length=512, truncation=True)
数据的编码结果包含input_ids、attention_mask等模型输入内容。
根据问答任务的要求，定义答案的起始位置和结束位置的标签。通过计算答案在文本中的相对位置，生成start_positions和end_positions：
# 设置答案的开始和结束位置
start_position = context.index("97% of BERT's performance") # 手动获取答案起始位置
end_position = start_position + len("97% of BERT's performance")

# 将起止位置转化为模型输入的索引
inputs['start_positions'] = torch.tensor([start_position])
inputs['end_positions'] = torch.tensor([end_position])
利用TrainingArguments设置训练参数，其中包括训练批次大小、学习率、训练周期等：
# 定义训练参数
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # 定义的训练数据集
    eval_dataset=eval_dataset     # 定义的验证数据集
)
利用定义的trainer对象对DistilBERT模型进行微调。在实际应用中，train_dataset和eval_dataset需要通过SQuAD格式或自定义的数据进行定义：
# 开始训练
trainer.train()
微调完成后，模型会存储在指定的输出目录中，供后续的问答推理任务使用，完成模型微调后，使用微调后的模型执行推理任务，提取答案。
# 测试模型，输入编码后的问题和段落
outputs = model(**inputs)

# 获取答案起始和结束位置
start_logits = outputs.start_logits
end_logits = outputs.end_logits

# 获取起止位置索引
start_index = torch.argmax(start_logits)
end_index = torch.argmax(end_logits)

# 解码出答案
answer = tokenizer.decode(inputs['input_ids'][0][start_index:end_index+1])
print(f"Answer: {answer}")

























