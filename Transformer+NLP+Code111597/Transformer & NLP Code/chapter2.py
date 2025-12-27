# 例2-1
import nltk
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.corpus import wordnet

# 下载WordNet词典资源
nltk.download('wordnet')
nltk.download('omw-1.4')

# 初始化词干提取器和词形还原器
stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()

# 示例文本数据
text = ["running", "happier", "boxes", "studies", "easily", "flying"]

# 定义词干提取与词形还原函数
def stem_and_lemmatize(words):
    stemmed_words = []
    lemmatized_words = []
    for word in words:
        # 词干提取
        stemmed_word = stemmer.stem(word)
        stemmed_words.append(stemmed_word)
        
        # 词形还原，指定词性为动词以提高准确性
        lemmatized_word = lemmatizer.lemmatize(word, pos=wordnet.VERB)
        lemmatized_words.append(lemmatized_word)
    
    return stemmed_words, lemmatized_words

# 清洗后的词干和词形还原结果
stemmed, lemmatized = stem_and_lemmatize(text)

# 输出结果
print("原始文本:", text)
print("词干提取结果:", stemmed)
print("词形还原结果:", lemmatized)

# 例2-2
import re
from nltk import ngrams
from transformers import BertTokenizer

# 示例文本
text = "自然语言处理是人工智能的重要领域。"

# 定义n-gram分词函数
def generate_ngrams(text, n):
    # 去除标点符号
    text = re.sub(r'[^\w\s]', '', text)
    # 切分为单词列表
    words = text.split()
    # 生成n-grams
    n_grams = list(ngrams(words, n))
    # 将n-gram转为字符串形式
    ngram_strings = [' '.join(gram) for gram in n_grams]
    return ngram_strings

# 二元和三元分词
bigrams = generate_ngrams(text, 2)
trigrams = generate_ngrams(text, 3)

# BERT分词
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
bert_tokens = bert_tokenizer.tokenize(text)

# 输出结果
print("原始文本:", text)
print("二元分词结果:", bigrams)
print("三元分词结果:", trigrams)
print("BERT分词结果:", bert_tokens)

# 例2-3
import torch

# 示例文本数据
sentences = [
    ["自然", "语言", "处理", "是", "人工智能", "的", "重要", "组成"],
    ["机器", "学习", "是", "人工智能", "的", "核心", "领域"]
]

# 训练Word2Vec模型
word2vec_model = Word2Vec(sentences, vector_size=50, window=2, min_count=1, sg=1)  # 使用Skip-gram模型

# 获取Word2Vec词嵌入
word2vec_nlp = word2vec_model.wv['自然']
word2vec_ai = word2vec_model.wv['人工智能']

# 初始化BERT分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
bert_model = BertModel.from_pretrained('bert-base-chinese')

# 定义BERT词嵌入提取函数
def get_bert_embedding(text):
    # 分词并转为ID
    inputs = tokenizer(text, return_tensors="pt")
    # 获取BERT嵌入
    outputs = bert_model(**inputs)
    # 获取[CLS]标记的嵌入表示
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding

# 获取BERT嵌入
text = "自然语言处理是人工智能的重要组成"
bert_embedding = get_bert_embedding(text)

# 输出结果
print("Word2Vec '自然' 词嵌入:", word2vec_nlp)
print("Word2Vec '人工智能' 词嵌入:", word2vec_ai)
print("BERT句子嵌入:", bert_embedding)


# 例2-4
import torch
from gensim.models import Word2Vec
from transformers import BertTokenizer, BertModel

# 定义示例文本
sentences = [
    ["自然", "语言", "处理", "是", "人工智能", "的", "重要", "组成"],
    ["机器", "学习", "是", "人工智能", "的", "核心", "领域"],
    ["深度", "学习", "广泛", "应用", "于", "图像", "处理", "和", "语音", "识别"]
]

# --------------------------- Word2Vec 嵌入生成 ---------------------------

# 使用Skip-gram训练Word2Vec模型，设定词向量维度为50
word2vec_model = Word2Vec(sentences, vector_size=50, window=2, min_count=1, sg=1, epochs=100)

# 获取Word2Vec词嵌入
word2vec_nlp = word2vec_model.wv['自然']
word2vec_ai = word2vec_model.wv['人工智能']

# 显示 Word2Vec 模型的词汇表
print("Word2Vec 词汇表:", list(word2vec_model.wv.index_to_key))

# 显示 '自然' 和 '人工智能' 的词向量
print("Word2Vec '自然' 词向量:", word2vec_nlp)
print("Word2Vec '人工智能' 词向量:", word2vec_ai)

# 使用 Word2Vec 模型计算相似度
similarity = word2vec_model.wv.similarity("自然", "语言")
print("词语 '自然' 与 '语言' 的相似度:", similarity)

# 使用 Word2Vec 查找与 '人工智能' 相似的词
similar_words = word2vec_model.wv.most_similar("人工智能", topn=3)
print("与 '人工智能' 最相似的词:", similar_words)

# --------------------------- BERT 嵌入生成 ---------------------------

# 初始化 BERT 分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
bert_model = BertModel.from_pretrained('bert-base-chinese')

# 定义 BERT 嵌入提取函数
def get_bert_embedding(text):
    # 将文本分词并转换为ID
    inputs = tokenizer(text, return_tensors="pt")
    # 通过 BERT 模型获得嵌入
    outputs = bert_model(**inputs)
    # 获取句子[CLS]标记的嵌入表示
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding

# 定义示例文本
texts = [
    "自然语言处理是人工智能的重要组成",
    "深度学习在图像识别中的应用",
    "机器学习是人工智能的核心领域"
]

# 获取每个文本的 BERT 嵌入
bert_embeddings = [get_bert_embedding(text) for text in texts]

# 输出每个文本的嵌入结果
for idx, embedding in enumerate(bert_embeddings):
    print(f"文本 {idx + 1} 的 BERT 嵌入向量:", embedding)

# --------------------------- 词嵌入对比分析 ---------------------------

# 分析 Word2Vec 嵌入与 BERT 嵌入的区别
def compare_embeddings(word2vec_model, bert_model, tokenizer, words):
    print("词嵌入对比分析结果：")
    for word in words:
        # Word2Vec 嵌入
        word2vec_embedding = word2vec_model.wv[word]
        
        # BERT 嵌入
        inputs = tokenizer(word, return_tensors="pt")
        outputs = bert_model(**inputs)
        bert_embedding = outputs.last_hidden_state[:, 0, :].squeeze().detach().numpy()
        
        # 输出 Word2Vec 嵌入与 BERT 嵌入
        print(f"\n词汇 '{word}' 的嵌入对比:")
        print("Word2Vec 嵌入:", word2vec_embedding)
        print("BERT 嵌入:", bert_embedding)

# 对比分析词汇嵌入效果
compare_words = ["自然", "学习", "人工智能"]
compare_embeddings(word2vec_model, bert_model, tokenizer, compare_words)

# --------------------------- 综合分析 ---------------------------

# 定义一个函数，计算文本的平均 Word2Vec 嵌入
def average_word2vec_embedding(text, model):
    # 计算每个词的嵌入并求平均
    embedding = sum([model.wv[word] for word in text if word in model.wv]) / len(text)
    return embedding

# 获取平均 Word2Vec 嵌入
text1 = ["自然", "语言", "处理", "是", "人工智能", "的", "重要", "组成"]
avg_embedding = average_word2vec_embedding(text1, word2vec_model)
print("文本的平均 Word2Vec 嵌入:", avg_embedding)

# 定义函数计算BERT嵌入的欧氏距离
def euclidean_distance(embedding1, embedding2):
    return torch.dist(embedding1, embedding2, p=2).item()

# 计算两个句子的 BERT 嵌入距离
bert_emb1 = get_bert_embedding("自然语言处理是人工智能的重要组成")
bert_emb2 = get_bert_embedding("深度学习在图像识别中的应用")
distance = euclidean_distance(bert_emb1, bert_emb2)
print("两个句子的 BERT 嵌入欧氏距离:", distance)

# --------------------------- 综合比较结果 ---------------------------

# 使用 BERT 和 Word2Vec 模型查找最相似句子
text_corpus = [
    "自然语言处理是人工智能的重要组成",
    "机器学习是人工智能的核心",
    "深度学习在图像识别中的应用",
    "数据科学与机器学习"
]

# 获取每个文本的 BERT 嵌入和平均 Word2Vec 嵌入
bert_embeddings = [get_bert_embedding(text) for text in text_corpus]
word2vec_embeddings = [average_word2vec_embedding(text.split(), word2vec_model) for text in text_corpus]

# 展示每个文本的 BERT 和 Word2Vec 嵌入
for idx, (bert_embed, word2vec_embed) in enumerate(zip(bert_embeddings, word2vec_embeddings)):
    print(f"\n文本 {idx + 1} 的嵌入对比:")
    print("BERT 嵌入:", bert_embed)
    print("平均 Word2Vec 嵌入:", word2vec_embed)

# 例2-5
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# 示例文本数据，包含OOV词汇
sentences = [
    "深度学习是一种机器学习技术",
    "语言处理在人工智能中扮演重要角色",
    "自然语言处理是一门重要学科",
    "新兴技术包括量子计算和深度神经网络"
]

# 创建字符级别词汇表
def build_char_vocab(sentences):
    char_vocab = set()
    for sentence in sentences:
        for char in sentence:
            char_vocab.add(char)
    char2idx = {char: idx + 1 for idx, char in enumerate(char_vocab)}
    char2idx['<PAD>'] = 0  # 添加填充标记
    return char2idx

char2idx = build_char_vocab(sentences)
vocab_size = len(char2idx)

# 定义字符级嵌入模型
class CharLevelEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(CharLevelEmbedding, self).__init__()
        self.char_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
    
    def forward(self, x):
        embedded_chars = self.char_embedding(x)
        # 计算字符嵌入的平均表示
        return embedded_chars.mean(dim=1)

# 生成字符ID序列
def encode_sentence(sentence, char2idx, max_len=20):
    char_ids = [char2idx.get(char, 0) for char in sentence]  # 0作为OOV字符
    if len(char_ids) < max_len:
        char_ids += [0] * (max_len - len(char_ids))  # 填充字符
    else:
        char_ids = char_ids[:max_len]
    return torch.tensor(char_ids)

encoded_sentences = [encode_sentence(sentence, char2idx) for sentence in sentences]
encoded_tensor = torch.stack(encoded_sentences)

# 初始化模型
embed_dim = 16  # 嵌入维度
char_embed_model = CharLevelEmbedding(vocab_size, embed_dim)

# 生成字符级嵌入
char_embeddings = char_embed_model(encoded_tensor)

# 输出结果
print("字符级词汇表:", char2idx)
print("编码后的字符ID序列:\n", encoded_tensor)
print("字符级嵌入:\n", char_embeddings)

# 例2-6
import json
import pandas as pd

# ------------------------ JSON文件读取与处理 ------------------------

# JSON文件内容模拟
json_data = [
    {
        "id": 1,
        "text": "自然语言处理是一门重要的学科",
        "tags": ["NLP", "人工智能", "计算机科学"]
    },
    {
        "id": 2,
        "text": "深度学习在图像识别领域表现优异",
        "tags": ["深度学习", "计算机视觉", "人工智能"]
    },
    {
        "id": 3,
        "text": "机器学习可以用于预测股票价格",
        "tags": ["机器学习", "金融", "预测"]
    }
]

# 将JSON数据写入文件
with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=4)

# 读取JSON文件
def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 提取特定字段
    texts = [entry['text'] for entry in data]
    tags = [entry['tags'] for entry in data]
    return texts, tags

# 读取JSON文件并展示
texts, tags = read_json_file('data.json')
print("JSON文件中的文本内容:", texts)
print("JSON文件中的标签:", tags)

# ------------------------ CSV文件读取与处理 ------------------------

# 创建模拟CSV数据
csv_data = {
    "id": [1, 2, 3],
    "text": ["自然语言处理是一门重要的学科", "深度学习在图像识别领域表现优异", "机器学习可以用于预测股票价格"],
    "tags": ["NLP|人工智能|计算机科学", "深度学习|计算机视觉|人工智能", "机器学习|金融|预测"]
}

# 将数据写入CSV文件
csv_df = pd.DataFrame(csv_data)
csv_df.to_csv('data.csv', index=False, encoding='utf-8')

# 读取CSV文件
def read_csv_file(file_path):
    data = pd.read_csv(file_path, encoding='utf-8')
    # 分割标签字段，处理为列表形式
    data['tags'] = data['tags'].apply(lambda x: x.split('|'))
    return data

# 读取CSV文件并展示
csv_data = read_csv_file('data.csv')
print("\nCSV文件内容:")
print(csv_data)

# ------------------------ 优化读取大文件的速度 ------------------------

# 分块读取CSV文件
def read_large_csv_in_chunks(file_path, chunk_size=2):
    chunk_data = pd.DataFrame()
    for chunk in pd.read_csv(file_path, chunksize=chunk_size, encoding='utf-8'):
        chunk['tags'] = chunk['tags'].apply(lambda x: x.split('|'))
        chunk_data = pd.concat([chunk_data, chunk], ignore_index=True)
    return chunk_data

# 模拟大文件读取
large_csv_data = read_large_csv_in_chunks('data.csv', chunk_size=1)
print("\n分块读取CSV文件内容:")
print(large_csv_data)

# ------------------------ 转换与保存JSON和CSV格式 ------------------------

# 将读取的数据保存为新的JSON文件
def save_to_json(data, file_path):
    data_dict = data.to_dict(orient='records')
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=4)

save_to_json(csv_data, 'converted_data.json')
print("\n已将CSV数据转换并保存为JSON格式文件: converted_data.json")

# 将JSON数据保存为新的CSV文件
def save_to_csv(data, file_path):
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False, encoding='utf-8')

save_to_csv(json_data, 'converted_data.csv')
print("\n已将JSON数据转换并保存为CSV格式文件: converted_data.csv")

# 例2-6
import json
import pandas as pd

# 示例：逐行读取JSON文件并处理错误数据
def process_large_json(file_path, required_fields):
    valid_data = []
    error_log = []

    with open(file_path, 'r', encoding='utf-8') as f:
        line_num = 1
        for line in f:
            try:
                # 尝试加载当前行的JSON数据
                record = json.loads(line)
                
                # 检查是否包含必要字段
                if all(field in record for field in required_fields):
                    valid_data.append(record)
                else:
                    # 记录缺失字段的数据
                    error_log.append({"line": line_num, "error": "Missing required fields", "data": line})
            except json.JSONDecodeError:
                # 记录JSON格式错误
                error_log.append({"line": line_num, "error": "JSON Decode Error", "data": line})
            line_num += 1

    # 输出校验通过的数据和错误日志
    return valid_data, error_log

# 设置所需字段
required_fields = ["id", "text", "tags"]

# 处理大JSON文件并获取有效数据和错误日志
valid_data, error_log = process_large_json('data_large.json', required_fields)

# 将有效数据转换为DataFrame便于后续处理
df = pd.DataFrame(valid_data)

print("有效数据：")
print(df.head())

print("\n错误日志：")
for error in error_log[:5]:  # 只显示前5条错误日志示例
    print(f"行号 {error['line']} - 错误类型: {error['error']} - 数据: {error['data']}")



import pandas as pd
import numpy as np
import json

# 示例标签数据，包含多个标签
data = {
    "id": [1, 2, 3],
    "text": [
        "自然语言处理是一门重要的学科",
        "深度学习在图像识别领域表现优异",
        "机器学习可以用于预测股票价格"
    ],
    "tags": [
        ["NLP", "人工智能", "计算机科学"],
        ["深度学习", "计算机视觉", "人工智能"],
        ["机器学习", "金融", "预测"]
    ]
}

# 创建DataFrame
df = pd.DataFrame(data)

# 获取所有唯一标签并生成标签索引字典
all_tags = sorted(set(tag for sublist in df['tags'] for tag in sublist))
tag_to_index = {tag: idx for idx, tag in enumerate(all_tags)}

# Multi-Hot编码
def multi_hot_encode(tags, tag_to_index):
    multi_hot_vector = np.zeros(len(tag_to_index), dtype=int)
    for tag in tags:
        if tag in tag_to_index:
            multi_hot_vector[tag_to_index[tag]] = 1
    return multi_hot_vector

# 为每个样本生成Multi-Hot编码
df['multi_hot'] = df['tags'].apply(lambda tags: multi_hot_encode(tags, tag_to_index))

# 输出标签编码字典和每个样本的编码结果
print("标签编码字典:", tag_to_index)
print("\n每个样本的Multi-Hot编码:")
print(df[['id', 'text', 'multi_hot']])

# 存储优化：将编码后的标签矩阵保存为二进制格式
multi_hot_matrix = np.stack(df['multi_hot'].values)
np.save('multi_hot_labels.npy', multi_hot_matrix)

# 读取并展示存储的二进制标签矩阵
loaded_multi_hot_matrix = np.load('multi_hot_labels.npy')
print("\n加载后的Multi-Hot编码矩阵:")
print(loaded_multi_hot_matrix)

# 将DataFrame存储为CSV
df.drop(columns=['tags', 'multi_hot']).to_csv('data_with_labels.csv', index=False, encoding='utf-8')
print("\n已将数据和标签编码保存至CSV文件: data_with_labels.csv")

# 例2-7
import random
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

# 示例句子
sentence = "自然语言处理是人工智能和计算机科学的重要组成部分。"

# 同义词替换
def synonym_replacement(sentence, n=2):
    words = word_tokenize(sentence)
    new_words = words[:]
    random.shuffle(words)
    num_replaced = 0

    for word in words:
        synonyms = wordnet.synsets(word)
        if synonyms:
            synonym = synonyms[0].lemmas()[0].name()
            new_words = [synonym if w == word else w for w in new_words]
            num_replaced += 1
            if num_replaced >= n:
                break
    return ' '.join(new_words)

# 句子反转
def reverse_sentence(sentence):
    words = word_tokenize(sentence)
    reversed_sentence = ' '.join(words[::-1])
    return reversed_sentence

# 生成增强后的句子
synonym_replaced_sentence = synonym_replacement(sentence)
reversed_sentence = reverse_sentence(sentence)

print("原始句子:", sentence)
print("同义词替换后的句子:", synonym_replaced_sentence)
print("句子反转后的句子:", reversed_sentence)

# ------------------ 批量处理文本增强 ------------------

# 示例句子列表
sentences = [
    "机器学习可以用于分类和回归任务。",
    "深度学习的应用覆盖了计算机视觉和自然语言处理。",
    "数据增强方法可以帮助提高模型的泛化能力。"
]

# 批量同义词替换与句子反转
def augment_sentences(sentences, n=2):
    augmented_sentences = []
    for sentence in sentences:
        synonym_sentence = synonym_replacement(sentence, n)
        reversed_sentence = reverse_sentence(sentence)
        augmented_sentences.append((sentence, synonym_sentence, reversed_sentence))
    return augmented_sentences

# 增强后的句子集合
augmented_sentences = augment_sentences(sentences)
print("\n批量文本增强结果:")

for original, synonym_aug, reversed_aug in augmented_sentences:
    print("原句:", original)
    print("同义词替换:", synonym_aug)
    print("句子反转:", reversed_aug)
    print("-" * 40)

# 例2-8
import random
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

# 原始句子
sentence = "机器学习可以用于自然语言处理和计算机视觉等领域。"

# 同义词替换
def synonym_replacement(words, n):
    new_words = words[:]
    random_word_list = list(set([word for word in words if wordnet.synsets(word)]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = wordnet.synsets(random_word)
        if synonyms:
            synonym = synonyms[0].lemmas()[0].name()
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
            if num_replaced >= n:
                break
    return new_words

# 随机插入
def random_insertion(words, n):
    new_words = words[:]
    for _ in range(n):
        add_word(new_words)
    return new_words

def add_word(new_words):
    synonyms = []
    random_word = new_words[random.randint(0, len(new_words) - 1)]
    for syn in wordnet.synsets(random_word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    if synonyms:
        new_words.insert(random.randint(0, len(new_words) - 1), random.choice(synonyms))

# 随机删除
def random_deletion(words, p):
    if len(words) == 1:
        return words
    return [word for word in words if random.uniform(0, 1) > p]

# 随机交换
def random_swap(words, n):
    new_words = words[:]
    for _ in range(n):
        swap_word(new_words)
    return new_words

def swap_word(new_words):
    idx1, idx2 = random.sample(range(len(new_words)), 2)
    new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]

# EDA方法
def eda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=4):
    words = word_tokenize(sentence)
    num_words = len(words)
    augmented_sentences = []

    n_sr = max(1, int(alpha_sr * num_words))
    n_ri = max(1, int(alpha_ri * num_words))
    n_rs = max(1, int(alpha_rs * num_words))

    augmented_sentences.append(' '.join(synonym_replacement(words, n_sr)))
    augmented_sentences.append(' '.join(random_insertion(words, n_ri)))
    augmented_sentences.append(' '.join(random_swap(words, n_rs)))
    augmented_sentences.append(' '.join(random_deletion(words, p_rd)))

    return augmented_sentences

# 扩充后的句子
augmented_sentences = eda(sentence)

print("原始句子:", sentence)
print("\n增强后的句子:")
for idx, augmented_sentence in enumerate(augmented_sentences):
    print(f"增强句子 {idx + 1}: {augmented_sentence}")








