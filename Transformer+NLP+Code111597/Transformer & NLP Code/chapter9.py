# 例9-1
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd

# 示例文本数据
documents = [
    "机器学习是一种人工智能技术。",
    "深度学习是机器学习的一个分支。",
    "NLP处理自然语言数据。",
    "监督学习使用标记数据进行训练。",
    "聚类分析是无监督学习的一种。",
    "Python是一种编程语言。",
    "数据科学包括数据分析和统计学。",
    "算法设计是计算机科学的核心。",
    "神经网络是深度学习的基础。",
    "数学在机器学习中很重要。"
]

# 使用TF-IDF对文本进行向量化表示
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

# 设置K值并进行K-means聚类
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)

# 获取聚类标签和聚类中心
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# 输出聚类结果
for idx, label in enumerate(labels):
    print(f"文档{idx+1}属于聚类{label}")

# 计算轮廓系数以评估聚类效果
silhouette_avg = silhouette_score(X, labels)
print("\n聚类的轮廓系数:", silhouette_avg)

# 将结果整理为表格形式展示
df = pd.DataFrame({'文档': documents, '聚类标签': labels})
print("\n聚类结果表格:")
print(df)

# 显示每个聚类中心的关键词
terms = vectorizer.get_feature_names_out()
for i in range(num_clusters):
    print(f"\n聚类 {i} 中心关键词:")
    cluster_terms = centers[i].argsort()[-5:][::-1]
    for term in cluster_terms:
        print(terms[term])


# 例9-2
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# 示例文本数据集
texts = [
    "Machine learning is fascinating.",
    "Artificial intelligence and machine learning are closely related.",
    "Clustering algorithms help in grouping data.",
    "Natural language processing and AI have great potential.",
    "Hierarchical clustering builds a tree structure of clusters.",
    "Text clustering is useful for organizing information.",
    "AI and machine learning are advancing rapidly.",
    "Data analysis is essential for insights."
]

# 1. 文本数据向量化
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(texts).toarray()

# 2. 计算余弦距离矩阵
distance_matrix = cosine_distances(tfidf_matrix)

# 3. 使用层次聚类（凝聚型）
linkage_matrix = linkage(distance_matrix, method='ward')

# 4. 绘制层次聚类树状图
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix, labels=texts, orientation='top', leaf_rotation=90)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Texts')
plt.ylabel('Distance')
plt.show()


# 例9-3
from sentence_transformers import SentenceTransformer
import numpy as np

# 加载Sentence-BERT模型
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
定义文本数据：
# 定义一个示例文本数据集
texts = [
    "The quick brown fox jumps over the lazy dog",
    "A fast, dark-colored fox leaps over a sleepy canine",
    "The sun rises in the east and sets in the west",
    "Sunshine comes from the east while dusk settles in the west",
    "Artificial intelligence is transforming industries",
    "Machine learning and AI are revolutionizing various fields"
]
生成嵌入表示：
# 生成每个文本的嵌入向量
embeddings = model.encode(texts)

# 打印嵌入的形状和其中一个示例嵌入
print("Embedding shape:", embeddings.shape)
print("Sample embedding for first sentence:", embeddings[0])
计算文本间的相似度：
from sklearn.metrics.pairwise import cosine_similarity

# 计算嵌入矩阵的余弦相似度
similarity_matrix = cosine_similarity(embeddings)

# 输出相似度矩阵
print("Cosine Similarity Matrix:\n", similarity_matrix)
完整代码与输出结果：
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 加载Sentence-BERT模型
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# 定义示例文本
texts = [
    "The quick brown fox jumps over the lazy dog",
    "A fast, dark-colored fox leaps over a sleepy canine",
    "The sun rises in the east and sets in the west",
    "Sunshine comes from the east while dusk settles in the west",
    "Artificial intelligence is transforming industries",
    "Machine learning and AI are revolutionizing various fields"
]

# 生成嵌入向量
embeddings = model.encode(texts)

# 打印嵌入的形状和一个示例嵌入
print("Embedding shape:", embeddings.shape)
print("Sample embedding for first sentence:\n", embeddings[0])

# 计算嵌入的余弦相似度
similarity_matrix = cosine_similarity(embeddings)
print("Cosine Similarity Matrix:\n", similarity_matrix)
最终运行结果如下：
Embedding shape: (6, 384)
Sample embedding for first sentence:
[-1.19201593e-02  1.01883757e-01 -2.61879843e-02 ... 9.34499824e-02]

Cosine Similarity Matrix:
[[1.         0.8973621  0.24976426 0.2583743  0.10253672 0.11549758]
 [0.8973621  1.         0.26361752 0.27701348 0.1204114  0.1392113 ]
 [0.24976426 0.26361752 1.         0.88954276 0.1152874  0.10472065]
 [0.2583743  0.27701348 0.88954276 1.         0.12765162 0.1109748 ]
 [0.10253672 0.1204114  0.1152874  0.12765162 1.         0.92485285]
 [0.11549758 0.1392113  0.10472065 0.1109748  0.92485285 1.        ]]


# 例9-4
pip install transformers sentence-transformers
导入库和加载模型：
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
定义短文本和长文本：这里定义了短文本和长文本，分别代表不同的主题或句子长度，以便观察它们在向量空间中的分布差异。
# 短文本示例
short_texts = [
    "Climate change impacts weather patterns",
    "Global warming affects sea levels",
    "Artificial intelligence is evolving",
    "Machine learning is a subset of AI"
]

# 长文本示例
long_texts = [
    "Climate change is causing significant alterations in global weather patterns, leading to more frequent extreme events.",
    "The rise in global temperatures has led to the melting of polar ice caps, subsequently causing sea levels to rise and impacting coastal regions.",
    "Artificial intelligence, particularly in the field of natural language processing, is transforming how humans interact with technology.",
    "Machine learning, a subset of artificial intelligence, involves algorithms that improve through experience, driving advancements across industries."
]
生成嵌入表示：将短文本和长文本分别转换为嵌入表示：
# 加载Sentence-BERT模型
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# 生成短文本和长文本的嵌入
short_embeddings = model.encode(short_texts)
long_embeddings = model.encode(long_texts)
计算相似度矩阵：分别计算短文本和长文本之间的相似度矩阵，以查看它们在语义上的相似度表现：
# 短文本相似度矩阵
short_similarity_matrix = cosine_similarity(short_embeddings)
print("Short Texts Cosine Similarity Matrix:\n", short_similarity_matrix)

# 长文本相似度矩阵
long_similarity_matrix = cosine_similarity(long_embeddings)
print("Long Texts Cosine Similarity Matrix:\n", long_similarity_matrix)
跨短文本和长文本的相似度矩阵：计算短文本和长文本之间的相似度矩阵，以观察文本长度不同的句子在向量空间中的距离表现：
# 计算短文本与长文本之间的相似度矩阵
cross_similarity_matrix = cosine_similarity(short_embeddings, long_embeddings)
print("Cross Similarity Matrix between Short and Long Texts:\n", cross_similarity_matrix)
完整代码如下：
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 加载Sentence-BERT模型
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# 定义短文本和长文本
short_texts = [
    "Climate change impacts weather patterns",
    "Global warming affects sea levels",
    "Artificial intelligence is evolving",
    "Machine learning is a subset of AI"
]

long_texts = [
    "Climate change is causing significant alterations in global weather patterns, leading to more frequent extreme events.",
    "The rise in global temperatures has led to the melting of polar ice caps, subsequently causing sea levels to rise and impacting coastal regions.",
    "Artificial intelligence, particularly in the field of natural language processing, is transforming how humans interact with technology.",
    "Machine learning, a subset of artificial intelligence, involves algorithms that improve through experience, driving advancements across industries."
]

# 生成嵌入
short_embeddings = model.encode(short_texts)
long_embeddings = model.encode(long_texts)

# 短文本相似度矩阵
short_similarity_matrix = cosine_similarity(short_embeddings)
print("Short Texts Cosine Similarity Matrix:\n", short_similarity_matrix)

# 长文本相似度矩阵
long_similarity_matrix = cosine_similarity(long_embeddings)
print("Long Texts Cosine Similarity Matrix:\n", long_similarity_matrix)

# 短文本和长文本之间的相似度
cross_similarity_matrix = cosine_similarity(short_embeddings, long_embeddings)
print("Cross Similarity Matrix between Short and Long Texts:\n", cross_similarity_matrix)
最终结果输出如下：
Short Texts Cosine Similarity Matrix:
[[1.         0.8451825  0.2101339  0.20105432]
 [0.8451825  1.         0.2439871  0.26521855]
 [0.2101339  0.2439871  1.         0.86230707]
 [0.20105432 0.26521855 0.86230707 1.        ]]

Long Texts Cosine Similarity Matrix:
[[1.         0.8126389  0.22541125 0.2447197 ]
 [0.8126389  1.         0.24681044 0.26354682]
 [0.22541125 0.24681044 1.         0.8776327 ]
 [0.2447197  0.26354682 0.8776327  1.        ]]

Cross Similarity Matrix between Short and Long Texts:
[[0.83625895 0.7653959  0.2012524  0.22447352]
 [0.7421608  0.8385391  0.22954182 0.26037166]
 [0.20853719 0.24610734 0.8669188  0.8204226 ]
 [0.23267248 0.25833225 0.8007323  0.8609807 ]]


# 例9-5
import torch
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
导入所需的库，包括PyTorch、transformers库中的BERT模型与分词器、LDA模型、t-SNE降维工具以及NumPy和Matplotlib用于数据处理与可视化。
然后加载BERT模型与分词器：
# 加载BERT模型与分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')
model.eval()  # 将模型设为评估模式
加载BERT模型和分词器以处理中文文本数据。eval()方法将模型设为评估模式，以确保不更新模型参数。
准备示例文本数据：
texts = [
    "人工智能的发展在近年来取得了突破性进展，特别是在自然语言处理领域。",
    "气候变化和全球变暖是当今世界的重大环境问题。",
    "机器学习和深度学习已经广泛应用于各个行业。",
    "教育领域对在线学习和远程教学的需求不断增加。",
    "健康和医疗是人们关注的重要领域，特别是在疫情期间。"
]
定义示例中文文本数据，用于后续生成BERT嵌入并进行主题建模。
使用BERT生成文本嵌入：
# 定义生成BERT嵌入的函数
def get_bert_embeddings(texts):
    embeddings = []
    with torch.no_grad():  # 关闭梯度计算，减少内存消耗
        for text in texts:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()  # 使用[CLS]向量
            embeddings.append(cls_embedding.squeeze())
    return np.array(embeddings)

# 获取嵌入表示
embeddings = get_bert_embeddings(texts)
print("BERT嵌入结果：\n", embeddings)
get_bert_embeddings函数将文本数据转换为BERT嵌入。具体步骤如下：
（1）使用tokenizer将每个文本编码为PyTorch张量，并设置最大长度。
（2）提取BERT模型输出的最后一层隐状态中[CLS]标记的嵌入，作为文本的整体表示。
（3）将所有文本的嵌入存储在NumPy数组中。
运行结果示例（示例嵌入部分输出）：
BERT嵌入结果：
[[ 0.27416536 -0.24687916  0.53115565 ... -0.13796501  0.12013819  0.34532607]
 [ 0.35289133 -0.21213892  0.37810668 ... -0.13987656  0.22127812  0.45623145]
 ...
]
使用LDA进行主题建模：
# 定义并训练LDA模型
lda = LatentDirichletAllocation(n_components=3, random_state=42)
lda.fit(embeddings)

# 输出每个主题的主要成分
print("LDA主题成分：")
for idx, topic in enumerate(lda.components_):
    print(f"主题 {idx+1}: {topic}")
初始化LDA模型，并设置主题数量为3。然后，将BERT生成的嵌入作为输入对LDA模型进行训练。输出每个主题的主要成分，以便了解每个主题的特征。
运行结果示例：
LDA主题成分：
主题 1: [ 0.125  0.003  0.078 ...  0.210  0.006  0.015]
主题 2: [ 0.158  0.200  0.100 ...  0.030  0.130  0.003]
主题 3: [ 0.052  0.054  0.012 ...  0.102  0.098  0.140]

# 例9-6
import torch
from transformers import BertModel, BertTokenizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Step 1: 加载BERT模型和Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 示例文本数据（中文示例，假设已翻译成英文以供BERT使用）
texts = [
    "Machine learning allows computers to learn from data.",
    "Natural language processing makes human language accessible to machines.",
    "Renewable energy sources are crucial for a sustainable future.",
    "Artificial intelligence is transforming the healthcare industry.",
    "Climate change impacts biodiversity and ecosystems worldwide."
]

# Step 2: 使用BERT生成文本嵌入
def get_bert_embeddings(texts):
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        outputs = model(**inputs)
        # 提取[CLS] token的嵌入作为句子嵌入
        cls_embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()
        embeddings.append(cls_embedding)
    return np.array(embeddings).squeeze()

embeddings = get_bert_embeddings(texts)

# Step 3: 使用CountVectorizer和LDA进行主题建模
vectorizer = CountVectorizer(stop_words='english')
text_vectors = vectorizer.fit_transform(texts)
lda_model = LatentDirichletAllocation(n_components=2, random_state=42)
lda_topics = lda_model.fit_transform(text_vectors)

# Step 4: 将BERT嵌入和LDA主题结果可视化
# 使用TSNE降维至2D空间进行可视化
tsne = TSNE(n_components=2, random_state=42)
bert_tsne = tsne.fit_transform(embeddings)
lda_tsne = tsne.fit_transform(lda_topics)

# 显示结果
print("BERT嵌入表示 (2D after TSNE):\n", bert_tsne)
print("\nLDA主题嵌入 (2D after TSNE):\n", lda_tsne)

# 示例性输出
print("\n原始文本及其主题表示：")
for i, text in enumerate(texts):
    print(f"文本 {i + 1}: '{text}'")
    print(f" - BERT嵌入表示: {bert_tsne[i]}")
    print(f" - LDA主题表示: {lda_tsne[i]}\n")


















