# 例6-1
import random
from collections import defaultdict

# 输入文本
text = "机器学习是一种通过统计方法让计算机拥有学习能力的技术。这种技术可以帮助计算机从数据中提取特征、分析模式，从而完成预测任务。"

# 将文本分词
tokens = text.split()

# 定义n-gram模型的n值
n = 3

# 构建n-gram模型
n_grams = defaultdict(list)

# 生成n-grams字典
for i in range(len(tokens) - n):
    key = tuple(tokens[i:i + n - 1])  # 使用前n-1个词作为键
    next_word = tokens[i + n - 1]     # 使用第n个词作为值
    n_grams[key].append(next_word)

# 定义生成文本函数
def generate_text(n_grams, num_words=15):
    # 随机选择n-gram的起始点
    start = random.choice(list(n_grams.keys()))
    result = list(start)
    for _ in range(num_words - len(start)):
        state = tuple(result[-(n - 1):])  # 获取最后n-1个词
        next_word_choices = n_grams.get(state, [])
        if not next_word_choices:
            break
        next_word = random.choice(next_word_choices)
        result.append(next_word)
    return ' '.join(result)

# 生成并输出文本
generated_text = generate_text(n_grams)
print("生成的文本:", generated_text)


# 例6-2
import random
from collections import defaultdict

# 输入示例文本
text = "深度学习模型通过多层非线性变换学习数据的特征。自然语言处理包括语言生成、机器翻译、文本分类等任务。机器学习方法不断演化，以适应更复杂的应用场景。"

# 分词
tokens = text.split()

# 定义n-gram模型的n值
n = 3

# 构建n-gram模型
n_grams = defaultdict(list)
for i in range(len(tokens) - n):
    key = tuple(tokens[i:i + n - 1])
    next_word = tokens[i + n - 1]
    n_grams[key].append(next_word)

# 定义生成文本的函数
def generate_text(n_grams, num_words=30):
    # 随机选择n-gram的起始词
    start = random.choice(list(n_grams.keys()))
    result = list(start)
    for _ in range(num_words - len(start)):
        state = tuple(result[-(n - 1):])  # 获取最后n-1个词作为状态
        next_word_choices = n_grams.get(state, [])
        if not next_word_choices:
            break
        next_word = random.choice(next_word_choices)
        result.append(next_word)
    return ' '.join(result)

# 生成文本
generated_text = generate_text(n_grams)
print("生成的长文本:", generated_text)

# 例6-3
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 初始化GPT-2模型和分词器
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

# Greedy Search实现
def greedy_search(input_text, max_len=50):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    generated = input_ids
    
    for _ in range(max_len):
        outputs = model(generated)
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)
        generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
        
        if next_token == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated[0], skip_special_tokens=True)

# Beam Search实现
def beam_search(input_text, max_len=50, beam_width=3):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    generated_sequences = [(input_ids, 0)]  # (tokens tensor, score)

    for _ in range(max_len):
        all_candidates = []
        for seq, score in generated_sequences:
            outputs = model(seq)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_probs = F.log_softmax(next_token_logits, dim=-1)
            top_tokens = torch.topk(next_token_probs, beam_width, dim=-1)

            for i in range(beam_width):
                next_token = top_tokens.indices[0, i].unsqueeze(0)
                new_seq = torch.cat((seq, next_token.unsqueeze(0)), dim=1)
                new_score = score + top_tokens.values[0, i].item()
                all_candidates.append((new_seq, new_score))
        
        # 按分数排序并选出beam_width个最佳序列
        ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        generated_sequences = ordered[:beam_width]

        # 结束标记检查
        if any(seq[0][-1] == tokenizer.eos_token_id for seq in generated_sequences):
            break
    
    # 返回得分最高的序列
    best_sequence = generated_sequences[0][0]
    return tokenizer.decode(best_sequence[0], skip_special_tokens=True)

# 测试Greedy Search和Beam Search
input_text = "Once upon a time"
greedy_result = greedy_search(input_text)
beam_result = beam_search(input_text)

print("Greedy Search Result:")
print(greedy_result)
print("\nBeam Search Result:")
print(beam_result)


# 例6-4
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 初始化模型和分词器
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# 输入文本
input_text = "在城市的绿色和乡村的绿色之外，还有一块心灵的绿色，它"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Greedy Search
greedy_output = model.generate(input_ids, max_length=80, do_sample=False)
greedy_text = tokenizer.decode(greedy_output[0], skip_special_tokens=True)

# Beam Search
beam_output = model.generate(input_ids, max_length=80, num_beams=5, early_stopping=True)
beam_text = tokenizer.decode(beam_output[0], skip_special_tokens=True)

# 打印生成的文本
print("Greedy Search Result:\n", greedy_text)
print("\nBeam Search Result:\n", beam_text)

# 例6-5
import torch
import torch.nn.functional as F

# 模拟的词汇表及其对应的概率分布
vocab = ['the', 'cat', 'sat', 'on', 'a', 'mat', 'dog', 'barked']
logits = torch.tensor([1.5, 2.3, 1.2, 0.8, 1.4, 0.6, 1.8, 1.0])  # 模拟的模型输出概率分布
probabilities = F.softmax(logits, dim=-1)  # 转换为概率分布

# Top-K采样实现
def top_k_sampling(probabilities, k):
    values, indices = torch.topk(probabilities, k)
    chosen_index = torch.multinomial(values, 1)  # 从Top-K词中随机选择一个
    return indices[chosen_index.item()]

# Top-P采样实现
def top_p_sampling(probabilities, p=0.9):
    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # 获取最小的词集合，使其累积概率大于p
    idx = torch.where(cumulative_probs >= p)[0][0]
    
    # 从这些词中随机选择
    chosen_index = torch.multinomial(sorted_probs[:idx+1], 1)
    return sorted_indices[chosen_index.item()]

# 执行Top-K采样（选择前3个词）
k = 3
top_k_word = top_k_sampling(probabilities, k)
print(f"Top-K采样结果: {vocab[top_k_word]}")

# 执行Top-P采样（设置累积概率阈值为0.9）
top_p_word = top_p_sampling(probabilities, 0.9)
print(f"Top-P采样结果: {vocab[top_p_word]}")

# 例6-5
import torch
import torch.nn.functional as F

# 模拟的中文词汇表及其对应的概率分布
vocab = ['我', '是', '学', '生', '在', '学', '习', '编', '程', '中', '信', '息', '技', '术', '方', '法']
logits = torch.tensor([1.2, 2.3, 1.5, 1.0, 2.5, 1.1, 2.0, 0.8, 1.9, 2.1, 0.9, 1.3, 1.7, 2.2, 1.4, 2.6])  # 模拟的模型输出概率分布
probabilities = F.softmax(logits, dim=-1)  # 转换为概率分布

# Top-K采样实现
def top_k_sampling(probabilities, k):
    values, indices = torch.topk(probabilities, k)  # 获取Top-K概率和索引
    chosen_index = torch.multinomial(values, 1)  # 从Top-K词中随机选择一个
    return indices[chosen_index.item()]

# Top-P采样实现
def top_p_sampling(probabilities, p=0.9):
    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)  # 按概率降序排序
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)  # 计算累积概率
    
    # 获取最小的词集合，使其累积概率大于p
    idx = torch.where(cumulative_probs >= p)[0][0]
    
    # 从这些词中随机选择
    chosen_index = torch.multinomial(sorted_probs[:idx+1], 1)
    return sorted_indices[chosen_index.item()]

# 文本生成函数
def generate_text(probabilities, method='top_k', k=3, p=0.9, num_words=5):
    text = ''
    for _ in range(num_words):
        if method == 'top_k':
            word_idx = top_k_sampling(probabilities, k)
        elif method == 'top_p':
            word_idx = top_p_sampling(probabilities, p)
        
        text += vocab[word_idx.item()]  # 生成下一个字符
    return text

# 执行Top-K采样（选择前3个词）
generated_text_k = generate_text(probabilities, method='top_k', k=3, num_words=5)
print(f"Top-K采样生成文本: {generated_text_k}")

# 执行Top-P采样（设置累积概率阈值为0.9）
generated_text_p = generate_text(probabilities, method='top_p', p=0.9, num_words=5)
print(f"Top-P采样生成文本: {generated_text_p}")


# 例6-6
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset

# 加载T5预训练模型和Tokenizer
model_name = "t5-small"  # 可以根据需求选择更大或者更小的版本
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# 加载新闻摘要数据集
dataset = load_dataset("cnn_dailymail", "3.0.0")

# 数据预处理：将输入的新闻文章和摘要分开
def preprocess_function(examples):
    inputs = [article for article in examples["article"]]
    targets = [summary for summary in examples["highlights"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=150, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 预处理训练和验证数据集
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 使用DataLoader进行批处理
from torch.utils.data import DataLoader

train_dataset = tokenized_datasets["train"]
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# 设置训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 设置优化器
from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)

# 训练模型
epochs = 3
for epoch in range(epochs):
    model.train()
    for batch in train_dataloader:
        # 移动数据到GPU
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# 生成摘要
def generate_summary(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length").to(device)
    summary_ids = model.generate(inputs["input_ids"], max_length=150, num_beams=4, length_penalty=2.0, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# 测试生成摘要
test_text = dataset["test"][0]["article"]
print("Original Article:")
print(test_text)
print("\nGenerated Summary:")
print(generate_summary(test_text))

# 例6-6
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader

# 加载T5模型和Tokenizer
model_name = "t5-small"  # 可以选择更大的模型
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# 加载数据集
dataset = load_dataset("cnn_dailymail", "3.0.0")

# 数据预处理：在文本中添加任务描述
def preprocess_function(examples):
    inputs = ["summarize: " + article for article in examples["article"]]  # 在输入前加上任务指令
    targets = examples["highlights"]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=150, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 预处理数据
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 使用DataLoader进行批处理
train_dataset = tokenized_datasets["train"]
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# 设置训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 设置优化器
optimizer = AdamW(model.parameters(), lr=5e-5)

# 训练模型
epochs = 3
for epoch in range(epochs):
    model.train()
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# 生成摘要
def generate_summary(input_text):
    input_text = "summarize: " + input_text  # 使用任务描述
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length").to(device)
    summary_ids = model.generate(inputs["input_ids"], max_length=150, num_beams=4, length_penalty=2.0, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# 测试生成摘要
test_text = dataset["test"][0]["article"]
print("Original Article:")
print(test_text)
print("\nGenerated Summary:")
print(generate_summary(test_text))

# 例6-7
from transformers import GPT2LMHeadModel, T5ForConditionalGeneration, BartForConditionalGeneration, AutoTokenizer
import torch

# 定义模型和分词器
device = "cuda" if torch.cuda.is_available() else "cpu"
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")

t5_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
t5_tokenizer = AutoTokenizer.from_pretrained("t5-small")

bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-small").to(device)
bart_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-small")

# 输入文本示例
input_text = "In a world where technology advances rapidly"

# GPT-2生成
gpt2_input = gpt2_tokenizer.encode(input_text, return_tensors="pt").to(device)
gpt2_output = gpt2_model.generate(gpt2_input, max_length=50, num_return_sequences=1)
gpt2_generated_text = gpt2_tokenizer.decode(gpt2_output[0], skip_special_tokens=True)

# T5生成
t5_input = t5_tokenizer("summarize: " + input_text, return_tensors="pt").to(device)
t5_output = t5_model.generate(t5_input.input_ids, max_length=50, num_return_sequences=1)
t5_generated_text = t5_tokenizer.decode(t5_output[0], skip_special_tokens=True)

# BART生成
bart_input = bart_tokenizer(input_text, return_tensors="pt").to(device)
bart_output = bart_model.generate(bart_input.input_ids, max_length=50, num_return_sequences=1)
bart_generated_text = bart_tokenizer.decode(bart_output[0], skip_special_tokens=True)

# 打印生成结果
print("GPT-2 Generated Text:\n", gpt2_generated_text)
print("\nT5 Generated Summary:\n", t5_generated_text)
print("\nBART Generated Text:\n", bart_generated_text)



# 例6-8
from transformers import GPT2LMHeadModel, T5ForConditionalGeneration, BartForConditionalGeneration, AutoTokenizer
import torch

# 定义模型和分词器
device = "cuda" if torch.cuda.is_available() else "cpu"
gpt2_model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall").to(device)
gpt2_tokenizer = AutoTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")

t5_model = T5ForConditionalGeneration.from_pretrained("imxly/t5-base-chinese").to(device)
t5_tokenizer = AutoTokenizer.from_pretrained("imxly/t5-base-chinese")

bart_model = BartForConditionalGeneration.from_pretrained("fnlp/bart-base-chinese").to(device)
bart_tokenizer = AutoTokenizer.from_pretrained("fnlp/bart-base-chinese")

# 输入中文文本示例
input_text = "在一个科技快速发展的世界中，人们面临着不断变化的生活方式"

# GPT-2生成
gpt2_input = gpt2_tokenizer.encode(input_text, return_tensors="pt").to(device)
gpt2_output = gpt2_model.generate(gpt2_input, max_length=50, num_return_sequences=1)
gpt2_generated_text = gpt2_tokenizer.decode(gpt2_output[0], skip_special_tokens=True)

# T5生成
t5_input = t5_tokenizer("summarize: " + input_text, return_tensors="pt").to(device)
t5_output = t5_model.generate(t5_input.input_ids, max_length=50, num_return_sequences=1)
t5_generated_text = t5_tokenizer.decode(t5_output[0], skip_special_tokens=True)

# BART生成
bart_input = bart_tokenizer(input_text, return_tensors="pt").to(device)
bart_output = bart_model.generate(bart_input.input_ids, max_length=50, num_return_sequences=1)
bart_generated_text = bart_tokenizer.decode(bart_output[0], skip_special_tokens=True)

# 打印生成结果
print("GPT-2生成文本:\n", gpt2_generated_text)
print("\nT5生成摘要:\n", t5_generated_text)
print("\nBART生成文本:\n", bart_generated_text)


# 例6-9
from transformers import T5ForConditionalGeneration, GPT2LMHeadModel, BartForConditionalGeneration, AutoTokenizer
import torch

# 配置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型与分词器
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
t5_tokenizer = AutoTokenizer.from_pretrained("t5-small")

gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")

bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(device)
bart_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

# 输入文本
input_text = "科技快速发展的今天，人们的生活方式发生了巨大的改变，关注健康和环境成为新趋势。"

# T5模型用于文本摘要生成
t5_input = t5_tokenizer("summarize: " + input_text, return_tensors="pt").to(device)
t5_output = t5_model.generate(t5_input.input_ids, max_length=50, num_beams=4, early_stopping=True)
t5_summary = t5_tokenizer.decode(t5_output[0], skip_special_tokens=True)

# GPT-2模型用于对话生成
gpt2_input = gpt2_tokenizer.encode(input_text, return_tensors="pt").to(device)
gpt2_output = gpt2_model.generate(gpt2_input, max_length=100, do_sample=True, top_k=50)
gpt2_dialogue = gpt2_tokenizer.decode(gpt2_output[0], skip_special_tokens=True)

# BART模型用于文本摘要生成
bart_input = bart_tokenizer(input_text, return_tensors="pt").to(device)
bart_output = bart_model.generate(bart_input.input_ids, max_length=50, num_beams=4, early_stopping=True)
bart_summary = bart_tokenizer.decode(bart_output[0], skip_special_tokens=True)

# 打印生成结果
print("T5模型生成的文本摘要:")
print(t5_summary)
print("\nGPT-2模型生成的对话内容:")
print(gpt2_dialogue)
print("\nBART模型生成的文本摘要:")
print(bart_summary)

# 例6-10
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 初始化模型与分词器
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# 对话上下文定义
context = [
    "用户: 你好，请介绍一下自己。",
    "助手: 你好，我是一个对话生成模型，可以回答你的问题。",
    "用户: 那你都能做些什么？"
]

# 定义函数：生成对话回复并保持上下文
def generate_response(context, max_length=50):
    # 拼接上下文文本
    input_text = "\n".join(context) + "\n助手:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # 生成响应
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=len(input_ids[0]) + max_length, pad_token_id=tokenizer.eos_token_id)
    
    # 解码生成的文本并去掉输入部分
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    response = output_text[len(input_text):].strip()
    
    # 将生成的回复添加到上下文
    context.append(f"助手: {response}")
    return response

# 生成对话
response = generate_response(context)
print(response)

# 再次生成，保持对话的上下文一致性
response = generate_response(context)
print(response)

# 例6-11
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 初始化模型与分词器
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# 定义对话上下文与主题标记
context = [
    {"text": "用户: 你好，可以介绍一下你是什么吗？", "topic": "general"},
    {"text": "助手: 你好，我是一个AI助手，可以回答各种问题。", "topic": "general"},
    {"text": "用户: 那你可以告诉我如何学习编程吗？", "topic": "programming"}
]

# 定义函数：生成对话回复并保持上下文和主题一致
def generate_response(context, topic, max_length=100):
    # 拼接上下文文本，并根据主题进行特殊标记
    input_text = "\n".join([entry["text"] for entry in context]) + f"\n助手({topic}):"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # 生成响应
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=len(input_ids[0]) + max_length, pad_token_id=tokenizer.eos_token_id)
    
    # 解码生成的文本并去掉输入部分
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    response = output_text[len(input_text):].strip()
    
    # 将生成的回复添加到上下文，标记主题
    context.append({"text": f"助手({topic}): {response}", "topic": topic})
    return response

# 生成对话回复，根据用户提问的主题调整内容
response = generate_response(context, "programming")
print(response)

# 用户再提问，保持对话的编程主题上下文
context.append({"text": "用户: 那如何提高编程水平？", "topic": "programming"})
response = generate_response(context, "programming")
print(response)

# 用户切换主题，问与健康相关的问题，生成健康主题的对话
context.append({"text": "用户: 顺便问一下，有关保持健康的建议吗？", "topic": "health"})
response = generate_response(context, "health")
print(response)

# 例6-12
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 初始化DialoGPT模型和分词器
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 对话历史缓存
chat_history_ids = None

# 定义对话函数
def generate_response(input_text, chat_history_ids=None, max_length=100):
    # 将输入文本编码
    new_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')

    # 合并对话历史与新输入，并生成回复
    bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids
    chat_history_ids = model.generate(bot_input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id)

    # 解码生成的回复
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, chat_history_ids

# 示例对话轮次
inputs = ["你好，可以介绍一下自己吗？", "如何提高编程水平？", "有哪些健康建议？"]

# 执行多轮对话
for user_input in inputs:
    response, chat_history_ids = generate_response(user_input, chat_history_ids)
    print(f"用户: {user_input}")
    print(f"助手: {response}\n")

import torch

# 加载T5模型和分词器
model_name = "t5-base"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# 输入文本准备
news_text = """
全球各大公司正积极投入资源开发人工智能，希望在医疗、交通和工业自动化等领域取得突破。
"""

# 加入任务前缀
input_text = "summarize: " + news_text
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# 配置生成策略的参数
max_len = 50

# 1. Greedy Search
greedy_output = model.generate(input_ids, max_length=max_len, num_beams=1)
greedy_summary = tokenizer.decode(greedy_output[0], skip_special_tokens=True)

# 2. Beam Search
beam_output = model.generate(input_ids, max_length=max_len, num_beams=4, early_stopping=True)
beam_summary = tokenizer.decode(beam_output[0], skip_special_tokens=True)

# 3. Top-K 采样
top_k_output = model.generate(input_ids, max_length=max_len, do_sample=True, top_k=50)
top_k_summary = tokenizer.decode(top_k_output[0], skip_special_tokens=True)

# 4. Top-P 采样
top_p_output = model.generate(input_ids, max_length=max_len, do_sample=True, top_p=0.92)
top_p_summary = tokenizer.decode(top_p_output[0], skip_special_tokens=True)

# 5. 结合Beam Search与Top-K采样
combined_output = model.generate(input_ids, max_length=max_len, num_beams=4, do_sample=True, top_k=50, early_stopping=True)
combined_summary = tokenizer.decode(combined_output[0], skip_special_tokens=True)

# 打印结果
print("Greedy Search生成的摘要:")
print(greedy_summary)
print("\nBeam Search生成的摘要:")
print(beam_summary)
print("\nTop-K采样生成的摘要:")
print(top_k_summary)
print("\nTop-P采样生成的摘要:")
print(top_p_summary)
print("\n结合Beam Search与Top-K采样生成的摘要:")
print(combined_summary)



import torch

# 加载T5模型和分词器
model_name = "t5-base"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# 输入文本准备
news_text = """
全球各大公司正积极投入资源开发人工智能，希望在医疗、交通和工业自动化等领域取得突破。
"""

# 加入任务前缀
input_text = "summarize: " + news_text
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# 配置生成策略的参数
max_len = 50

# 1. Greedy Search
greedy_output = model.generate(input_ids, max_length=max_len, num_beams=1)
greedy_summary = tokenizer.decode(greedy_output[0], skip_special_tokens=True)

# 2. Beam Search
beam_output = model.generate(input_ids, max_length=max_len, num_beams=4, early_stopping=True)
beam_summary = tokenizer.decode(beam_output[0], skip_special_tokens=True)

# 3. Top-K 采样
top_k_output = model.generate(input_ids, max_length=max_len, do_sample=True, top_k=50)
top_k_summary = tokenizer.decode(top_k_output[0], skip_special_tokens=True)

# 4. Top-P 采样
top_p_output = model.generate(input_ids, max_length=max_len, do_sample=True, top_p=0.92)
top_p_summary = tokenizer.decode(top_p_output[0], skip_special_tokens=True)

# 5. 结合Beam Search与Top-K采样
combined_output = model.generate(input_ids, max_length=max_len, num_beams=4, do_sample=True, top_k=50, early_stopping=True)
combined_summary = tokenizer.decode(combined_output[0], skip_special_tokens=True)

# 打印结果
print("Greedy Search生成的摘要:")
print(greedy_summary)
print("\nBeam Search生成的摘要:")
print(beam_summary)
print("\nTop-K采样生成的摘要:")
print(top_k_summary)
print("\nTop-P采样生成的摘要:")
print(top_p_summary)
print("\n结合Beam Search与Top-K采样生成的摘要:")
print(combined_summary)

import torch

# 加载T5模型和分词器
model_name = "t5-base"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# 输入文本准备
news_text = """
全球各大公司正积极投入资源开发人工智能，希望在医疗、交通和工业自动化等领域取得突破。
"""

# 加入任务前缀
input_text = "summarize: " + news_text
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# 配置生成策略的参数
max_len = 50

# 1. Greedy Search
greedy_output = model.generate(input_ids, max_length=max_len, num_beams=1)
greedy_summary = tokenizer.decode(greedy_output[0], skip_special_tokens=True)

# 2. Beam Search
beam_output = model.generate(input_ids, max_length=max_len, num_beams=4, early_stopping=True)
beam_summary = tokenizer.decode(beam_output[0], skip_special_tokens=True)

# 3. Top-K 采样
top_k_output = model.generate(input_ids, max_length=max_len, do_sample=True, top_k=50)
top_k_summary = tokenizer.decode(top_k_output[0], skip_special_tokens=True)

# 4. Top-P 采样
top_p_output = model.generate(input_ids, max_length=max_len, do_sample=True, top_p=0.92)
top_p_summary = tokenizer.decode(top_p_output[0], skip_special_tokens=True)

# 5. 结合Beam Search与Top-K采样
combined_output = model.generate(input_ids, max_length=max_len, num_beams=4, do_sample=True, top_k=50, early_stopping=True)
combined_summary = tokenizer.decode(combined_output[0], skip_special_tokens=True)

# 打印结果
print("Greedy Search生成的摘要:")
print(greedy_summary)
print("\nBeam Search生成的摘要:")
print(beam_summary)
print("\nTop-K采样生成的摘要:")
print(top_k_summary)
print("\nTop-P采样生成的摘要:")
print(top_p_summary)
print("\n结合Beam Search与Top-K采样生成的摘要:")
print(combined_summary)















