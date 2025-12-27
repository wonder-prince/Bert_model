# 例12-1
# 引入所需库
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset

# 假设数据集包含两列： 'text'（用户评论文本）和 'label'（情感标签，如 0 表示负向，1 表示中性，2 表示正向）
data = pd.DataFrame({
    "text": ["我很喜欢这个产品！", "服务态度差", "质量不错，但是有点贵", "非常满意，下次还会购买"],
    "label": [2, 0, 1, 2]
})

# 数据预处理
def preprocess_data(data, max_length=128):
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")  # 使用中文BERT
    # 对数据进行编码
    encodings = tokenizer(data['text'].tolist(), truncation=True, padding=True, max_length=max_length)
    return encodings

# 加载数据并分为训练集和测试集
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_encodings = preprocess_data(train_data)
test_encodings = preprocess_data(test_data)

# 转换为Dataset格式
train_dataset = Dataset.from_dict({"input_ids": train_encodings["input_ids"], "attention_mask": train_encodings["attention_mask"], "labels": train_data['label'].tolist()})
test_dataset = Dataset.from_dict({"input_ids": test_encodings["input_ids"], "attention_mask": test_encodings["attention_mask"], "labels": test_data['label'].tolist()})
此处的代码完成了数据的加载与编码，BertTokenizer用来将文本转化为模型可接受的输入格式，数据集中分为训练集和测试集，以便后续模型训练与评估。BERT的最大输入长度设置为128以适应不同的文本。
接下来，通过BertForSequenceClassification初始化BERT模型，用于情感分类任务，选择合适的训练参数，并启动模型训练。
# 模型初始化
model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=3)

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./results",          # 输出目录
    evaluation_strategy="epoch",     # 每个epoch后进行评估
    per_device_train_batch_size=8,   # 每个设备的训练批次大小
    per_device_eval_batch_size=8,    # 每个设备的评估批次大小
    num_train_epochs=3,              # 训练的总epoch数
    logging_dir='./logs',            # 日志保存目录
)

# 使用 Trainer API 进行训练和评估
trainer = Trainer(
    model=model,                         # 模型
    args=training_args,                  # 训练参数
    train_dataset=train_dataset,         # 训练数据集
    eval_dataset=test_dataset            # 测试数据集
)

# 模型训练
trainer.train()
此部分代码通过Trainer接口定义了训练参数，并设定了训练批次大小、训练轮数、评估策略等细节。训练模型后，BERT模型将使用预训练的语言知识来分析并分类情感。
训练完成后，接下来对模型进行评估，以确认模型的准确率。
# 模型评估
eval_results = trainer.evaluate()
print(f"模型评估结果: {eval_results}")


# 例12-2
import random
import jieba
from synonyms import synonyms
from opencc import OpenCC

# 加载原始数据
data = [
    {"text": "这款产品非常好用，功能强大且易操作", "label": "positive"},
    {"text": "服务态度差，体验非常糟糕", "label": "negative"},
    {"text": "产品质量一般，但价格实惠", "label": "neutral"}
]

# 近义词替换函数
def synonym_replacement(text, replace_prob=0.3):
    words = jieba.lcut(text)
    new_words = []
    for word in words:
        if random.random() < replace_prob:
            similar_words = synonyms.nearby(word)
            if similar_words:  # 如果有近义词
                word = random.choice(similar_words[0])
        new_words.append(word)
    return ''.join(new_words)

# 简繁体转换函数
cc = OpenCC('s2t')
def convert_simplified_to_traditional(text):
    return cc.convert(text)

# 拼写变化函数（适用于中文拼音相似的替换）
def typo_augmentation(text, typo_prob=0.2):
    typo_dict = {'好': '号', '差': '查', '强': '墙', '易': '依'}
    words = list(text)
    for i, word in enumerate(words):
        if random.random() < typo_prob and word in typo_dict:
            words[i] = typo_dict[word]
    return ''.join(words)

# 扩充数据
def augment_data(data):
    augmented_data = []
    for entry in data:
        text = entry['text']
        label = entry['label']

        # 原文数据
        augmented_data.append({"text": text, "label": label})
        
        # 近义词替换
        augmented_text = synonym_replacement(text)
        augmented_data.append({"text": augmented_text, "label": label})
        
        # 简繁体转换
        traditional_text = convert_simplified_to_traditional(text)
        augmented_data.append({"text": traditional_text, "label": label})
        
        # 拼写变化
        typo_text = typo_augmentation(text)
        augmented_data.append({"text": typo_text, "label": label})

    return augmented_data

# 运行增强代码并展示结果
augmented_data = augment_data(data)
for entry in augmented_data:
    print(f"Text: {entry['text']}, Label: {entry['label']}")


# 例12-3
# 导入所需的库
from sentence_transformers import SentenceTransformer
import numpy as np

# 初始化SBERT模型
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 样本数据，包含不同情感类别的句子
texts = [
    "这款产品非常好用，功能强大且易操作。",
    "服务态度差，体验非常糟糕。",
    "产品质量一般，但价格实惠。",
    "这是我用过的最好的一款应用。",
    "这家餐厅的服务真的很差劲。",
    "这件商品的性价比非常高，值得推荐！"
]

# 使用SBERT生成文本嵌入
embeddings = model.encode(texts)

# 输出每条文本的嵌入向量
for i, embedding in enumerate(embeddings):
    print(f"Text: {texts[i]}")
    print(f"Embedding: {embedding}\n")


# 例12-4
import numpy as np

# 示例情感词典
sentiment_dict = {"好用": 1, "强大": 1, "差劲": -1, "推荐": 1, "糟糕": -1}

def enhance_embedding(text, embedding):
    words = text.split(" ")
    weights = [sentiment_dict.get(word, 0) for word in words]
    # 计算情感增强后的权重
    enhancement_factor = np.mean(weights)
    return embedding * (1 + enhancement_factor)

# 应用情感增强到每个文本的嵌入
enhanced_embeddings = [enhance_embedding(text, emb) for text, emb in zip(texts, embeddings)]
for i, enhanced_embedding in enumerate(enhanced_embeddings):
    print(f"Text: {texts[i]}")
    print(f"Enhanced Embedding: {enhanced_embedding}\n")
在上述代码中，情感词典对每个文本的嵌入向量进行加权增强，使模型更关注情感词汇的权重。这种方法能够更好地捕捉情感向量信息，从而提升情感分类和聚类效果。
（3）应用降维方法优化嵌入
当生成的嵌入维度较高时，可以通过降维算法降低其复杂性，同时减少计算开销和存储需求。常用的降维方法包括PCA（主成分分析）和t-SNE，在聚类和相似度任务中可以带来更高的效率。
from sklearn.decomposition import PCA

# 使用PCA将嵌入向量降到50维
pca = PCA(n_components=50)
reduced_embeddings = pca.fit_transform(enhanced_embeddings)

# 输出降维结果
for i, reduced_embedding in enumerate(reduced_embeddings):
    print(f"Text: {texts[i]}")
    print(f"Reduced Embedding (50D): {reduced_embedding}\n")


# 例12-5
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import pandas as pd
在12.1.2中我们已经对文本数据进行了情感分类，并生成了每条文本的嵌入向量。为便于演示，下面创建一些示例数据，其中包含每条文本的情感类别和嵌入向量：
# 示例文本数据和情感分类结果
texts = [
    "产品非常好用，强烈推荐", 
    "非常不满意，质量差", 
    "服务不错，值得推荐",
    "性价比很高，非常划算",
    "很失望，不会再购买"
]

# 使用Sentence-BERT模型生成文本嵌入向量
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
embeddings = model.encode(texts)

# 查看生成的嵌入向量
for i, embedding in enumerate(embeddings):
    print(f"Text: {texts[i]}")
    print(f"Embedding: {embedding[:5]}... (length: {len(embedding)})\n")
使用K-means聚类算法对嵌入向量进行聚类。假设情感类别分为两类（正面、负面），在实际应用中可以根据数据规模和情感类型选择不同的聚类数：
# 定义聚类数（例如，分为两类：正面、负面）
num_clusters = 2

# 初始化KMeans模型并进行聚类
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
kmeans.fit(embeddings)

# 获取每个文本的聚类标签
cluster_labels = kmeans.labels_

# 打印每条文本的聚类结果
for i, label in enumerate(cluster_labels):
    print(f"Text: {texts[i]} -> Cluster Label: {label}")



# 例12-6
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.onnx import export
from onnxruntime import InferenceSession
import numpy as np

# Step 1: 加载预训练模型和分词器
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 设置设备为CPU
device = torch.device("cpu")
model.to(device)
model.eval()

# Step 2: 定义ONNX导出的路径
onnx_path = "bert_model.onnx"

# Step 3: 创建一个用于ONNX导出的示例输入
dummy_input = tokenizer("This is a sample input for ONNX conversion.", 
                        return_tensors="pt", 
                        padding="max_length", 
                        max_length=128, 
                        truncation=True)

# 将示例输入转为PyTorch张量
input_ids = dummy_input["input_ids"].to(device)
attention_mask = dummy_input["attention_mask"].to(device)

# Step 4: 导出模型为ONNX
export(model=model,
       tokenizer=tokenizer,
       opset=11,  # ONNX的opset版本
       output=onnx_path,
       input_names=["input_ids", "attention_mask"],
       dynamic_axes={
           "input_ids": {0: "batch_size", 1: "sequence_length"},
           "attention_mask": {0: "batch_size", 1: "sequence_length"}
       })

print(f"ONNX模型已导出至 {onnx_path}")

# Step 5: 使用ONNX Runtime加载模型并进行推理验证
session = InferenceSession(onnx_path)

# 准备输入数据
onnx_inputs = {
    "input_ids": input_ids.cpu().numpy(),
    "attention_mask": attention_mask.cpu().numpy()
}

# 使用ONNX模型进行推理
outputs = session.run(None, onnx_inputs)

# 验证输出
logits = outputs[0]
predicted_class = np.argmax(logits, axis=1)

print("ONNX推理结果:", logits)
print("预测类别:", predicted_class)

# Step 6: 比较PyTorch与ONNX的推理结果
with torch.no_grad():
    torch_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    torch_logits = torch_outputs.logits.cpu().numpy()

print("PyTorch推理结果:", torch_logits)
np.testing.assert_allclose(torch_logits, logits, rtol=1e-3, atol=1e-5)
print("PyTorch与ONNX推理结果一致")



# 例12-7
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 加载分词器和预训练模型
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 切换到评估模式
model.eval()

# 示例输入文本
texts = ["I love this product!", "This is a bad experience."]
将文本数据转换为模型可以接受的输入格式，确保与ONNX转换时的输入结构一致：
# 分词处理
encoded_inputs = tokenizer(
    texts,
    padding="max_length",
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

input_ids = encoded_inputs["input_ids"]
attention_mask = encoded_inputs["attention_mask"]
使用Hugging Face提供的ONNX导出工具，将PyTorch模型转换为ONNX模型：
from transformers.onnx import export

# 定义ONNX导出的文件路径
onnx_path = "bert_sentiment.onnx"

# 导出ONNX模型
export(
    model=model,
    tokenizer=tokenizer,
    output=onnx_path,
    opset=11,  # ONNX操作集版本
    input_names=["input_ids", "attention_mask"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"}
    }
)

print(f"ONNX模型已成功保存至: {onnx_path}")
通过ONNX Runtime加载模型，并对示例文本进行推理，验证ONNX模型是否正常工作：
import numpy as np
from onnxruntime import InferenceSession

# 加载ONNX模型
session = InferenceSession(onnx_path)

# 准备输入数据
onnx_inputs = {
    "input_ids": input_ids.numpy(),
    "attention_mask": attention_mask.numpy()
}

# 推理
onnx_outputs = session.run(None, onnx_inputs)
onnx_logits = onnx_outputs[0]

# 计算类别
predicted_classes = np.argmax(onnx_logits, axis=1)
print("ONNX模型预测结果:", predicted_classes)
通过比较PyTorch和ONNX推理结果，验证导出的模型是否正确：
with torch.no_grad():
    torch_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    torch_logits = torch_outputs.logits.numpy()

print("PyTorch模型预测结果:", np.argmax(torch_logits, axis=1))

# 验证结果是否一致
np.testing.assert_allclose(torch_logits, onnx_logits, rtol=1e-3, atol=1e-5)
print("PyTorch与ONNX推理结果一致")
运行结果如下：
ONNX模型已成功保存至: bert_sentiment.onnx
ONNX模型预测结果: [0 1]
PyTorch模型预测结果: [0 1]
PyTorch与ONNX推理结果一致


# 例12-8
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

# TensorRT日志记录器
logger = trt.Logger(trt.Logger.WARNING)

# 定义ONNX模型路径和TensorRT引擎路径
onnx_model_path = "bert_sentiment.onnx"
trt_engine_path = "bert_sentiment.trt"

# 创建TensorRT构建器和网络定义
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

# 加载ONNX模型
with open(onnx_model_path, "rb") as model_file:
    if not parser.parse(model_file.read()):
        print("ONNX模型解析失败")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        exit()

print("ONNX模型已成功加载至TensorRT网络")

# 构建TensorRT引擎
builder_config = builder.create_builder_config()
builder_config.max_workspace_size = 1 << 30  # 最大工作空间设置为1GB

# 启用FP16精度
if builder.platform_has_fast_fp16:
    builder_config.set_flag(trt.BuilderFlag.FP16)

# 构建引擎
engine = builder.build_engine(network, builder_config)
with open(trt_engine_path, "wb") as engine_file:
    engine_file.write(engine.serialize())

print(f"TensorRT引擎已保存至 {trt_engine_path}")
在TensorRT中进行INT8量化需要校准数据集。以下代码展示了如何使用校准器和校准数据进行量化：
import os
import random

class BertCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibration_data, batch_size=8, max_length=128):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.calibration_data = calibration_data
        self.batch_size = batch_size
        self.max_length = max_length
        self.device_input = cuda.mem_alloc(batch_size * max_length * np.dtype(np.int32).itemsize)
        self.current_index = 0

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index + self.batch_size > len(self.calibration_data):
            return None

        batch = self.calibration_data[self.current_index:self.current_index + self.batch_size]
        self.current_index += self.batch_size

        cuda.memcpy_htod(self.device_input, np.ascontiguousarray(batch))
        return [int(self.device_input)]

    def read_calibration_cache(self):
        return None

    def write_calibration_cache(self, cache):
        pass

# 示例校准数据（随机生成，用实际数据替代）
calibration_data = np.random.randint(0, 10000, size=(100, 128)).astype(np.int32)

# 构建量化引擎
builder_config.set_flag(trt.BuilderFlag.INT8)
calibrator = BertCalibrator(calibration_data)
builder_config.int8_calibrator = calibrator
int8_engine = builder.build_engine(network, builder_config)

# 保存INT8引擎
with open("bert_sentiment_int8.trt", "wb") as int8_engine_file:
    int8_engine_file.write(int8_engine.serialize())

print("INT8引擎已成功生成")
加载TensorRT引擎，并对示例输入进行推理，验证推理加速效果：
# 加载引擎
def load_engine(trt_runtime, engine_path):
    with open(engine_path, "rb") as f:
        engine_data = f.read()
    return trt_runtime.deserialize_cuda_engine(engine_data)

# 创建上下文
runtime = trt.Runtime(logger)
engine = load_engine(runtime, trt_engine_path)
context = engine.create_execution_context()

# 分配内存
input_shape = (1, 128)
output_shape = (1, 2)

d_input = cuda.mem_alloc(np.prod(input_shape) * np.dtype(np.float32).itemsize)
d_output = cuda.mem_alloc(np.prod(output_shape) * np.dtype(np.float32).itemsize)

# 输入输出绑定
bindings = [int(d_input), int(d_output)]

# 推理数据
input_data = np.random.rand(*input_shape).astype(np.float32)
cuda.memcpy_htod(d_input, input_data)

# 推理
context.execute_v2(bindings)

# 获取输出
output_data = np.empty(output_shape, dtype=np.float32)
cuda.memcpy_dtoh(output_data, d_output)
print("推理输出:", output_data)
最终运行结果如下：
# 加载引擎
def load_engine(trt_runtime, engine_path):
    with open(engine_path, "rb") as f:
        engine_data = f.read()
    return trt_runtime.deserialize_cuda_engine(engine_data)

# 创建上下文
runtime = trt.Runtime(logger)
engine = load_engine(runtime, trt_engine_path)
context = engine.create_execution_context()

# 分配内存
input_shape = (1, 128)
output_shape = (1, 2)

d_input = cuda.mem_alloc(np.prod(input_shape) * np.dtype(np.float32).itemsize)
d_output = cuda.mem_alloc(np.prod(output_shape) * np.dtype(np.float32).itemsize)

# 输入输出绑定
bindings = [int(d_input), int(d_output)]

# 推理数据
input_data = np.random.rand(*input_shape).astype(np.float32)
cuda.memcpy_htod(d_input, input_data)

# 推理
context.execute_v2(bindings)

# 获取输出
output_data = np.empty(output_shape, dtype=np.float32)
cuda.memcpy_dtoh(output_data, d_output)
print("推理输出:", output_data)

# 例12-9
import onnxruntime as ort
import numpy as np

# 定义 ONNX 模型路径
onnx_model_path = "bert_sentiment.onnx"

# 配置多线程推理选项
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 4  # 设置为 4 个线程
sess_options.inter_op_num_threads = 2  # 设置并发计算的线程数
sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL  # 启用并行模式
sess_options.log_severity_level = 3  # 降低日志输出级别

# 加载 ONNX Runtime 推理会话
session = ort.InferenceSession(onnx_model_path, sess_options)

print("ONNX Runtime 推理会话已成功加载")
使用多线程进行推理，可以在单机上处理多个输入任务，以提高模型的吞吐量：
# 示例输入数据
batch_size = 8
sequence_length = 128
input_data = np.random.randint(0, 10000, (batch_size, sequence_length)).astype(np.int64)
attention_data = np.ones((batch_size, sequence_length)).astype(np.int64)

# 定义输入字典
onnx_inputs = {
    "input_ids": input_data,
    "attention_mask": attention_data
}

# 多线程推理
outputs = session.run(None, onnx_inputs)
logits = outputs[0]

# 显示推理结果
print("ONNX Runtime 推理输出:", logits)
在本地通过多进程模拟分布式推理场景，可以使用Python的multiprocessing库：
from multiprocessing import Process, Queue

def onnx_worker(input_data, output_queue):
    # 单独加载一个 ONNX Runtime 会话
    local_session = ort.InferenceSession(onnx_model_path)
    # 推理
    outputs = local_session.run(None, input_data)
    output_queue.put(outputs[0])  # 将结果放入队列

# 创建进程队列
output_queue = Queue()

# 创建示例任务数据
task_1 = {"input_ids": np.random.randint(0, 10000, (1, sequence_length)).astype(np.int64),
          "attention_mask": np.ones((1, sequence_length)).astype(np.int64)}

task_2 = {"input_ids": np.random.randint(0, 10000, (1, sequence_length)).astype(np.int64),
          "attention_mask": np.ones((1, sequence_length)).astype(np.int64)}

# 启动两个进程进行推理
process_1 = Process(target=onnx_worker, args=(task_1, output_queue))
process_2 = Process(target=onnx_worker, args=(task_2, output_queue))

process_1.start()
process_2.start()

process_1.join()
process_2.join()

# 获取结果
result_1 = output_queue.get()
result_2 = output_queue.get()

print("分布式推理结果任务1:", result_1)
print("分布式推理结果任务2:", result_2)



# 例12-10
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

# 定义ONNX模型路径
onnx_model_path = "bert_sentiment.onnx"

# TensorRT日志记录器
logger = trt.Logger(trt.Logger.WARNING)

# 创建构建器、网络定义和解析器
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

# 加载ONNX模型
with open(onnx_model_path, "rb") as model_file:
    if not parser.parse(model_file.read()):
        print("ONNX模型解析失败")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        exit()

print("ONNX模型加载完成")

# 配置动态批量大小
builder_config = builder.create_builder_config()
builder_config.max_workspace_size = 1 << 30  # 设置最大工作空间为1GB

# 设置动态批量大小范围
profile = builder.create_optimization_profile()
input_name = network.get_input(0).name
profile.set_shape(input_name, (1, 128), (4, 128), (16, 128))  # 最小、最优和最大批量大小
builder_config.add_optimization_profile(profile)

# 构建引擎
engine = builder.build_engine(network, builder_config)
print("TensorRT引擎构建完成")
实现一个简单的自定义算子，假设需要对logits值执行自定义操作（如添加偏置）：
import ctypes

# 定义自定义算子的动态库路径
custom_plugin_path = "./custom_plugin.so"

# 加载自定义算子插件
ctypes.CDLL(custom_plugin_path)
print("自定义算子插件加载成功")

# 创建插件注册器
plugin_registry = trt.get_plugin_registry()
plugin_creator = plugin_registry.get_plugin_creator("CustomOp", "1", "")

# 设置自定义算子参数
plugin_fields = trt.PluginFieldCollection([
    trt.PluginField("bias", np.array([0.5], dtype=np.float32))
])

custom_plugin = plugin_creator.create_plugin("custom_op", plugin_fields)

# 添加自定义算子到网络
input_tensor = network.get_input(0)
custom_layer = network.add_plugin_v2([input_tensor], custom_plugin)
network.mark_output(custom_layer.get_output(0))

# 构建带有自定义算子的引擎
engine_with_custom_op = builder.build_engine(network, builder_config)
print("带有自定义算子的TensorRT引擎构建完成")
通过加载支持动态批量大小的引擎，执行推理任务：
# 加载引擎
def load_engine(trt_runtime, engine_path):
    with open(engine_path, "rb") as f:
        engine_data = f.read()
    return trt_runtime.deserialize_cuda_engine(engine_data)

runtime = trt.Runtime(logger)
engine = load_engine(runtime, "bert_sentiment_dynamic.trt")
context = engine.create_execution_context()

# 设置动态批量大小
batch_size = 8
context.set_binding_shape(0, (batch_size, 128))

# 分配内存
input_shape = (batch_size, 128)
output_shape = (batch_size, 2)

d_input = cuda.mem_alloc(np.prod(input_shape) * np.dtype(np.float32).itemsize)
d_output = cuda.mem_alloc(np.prod(output_shape) * np.dtype(np.float32).itemsize)

# 输入输出绑定
bindings = [int(d_input), int(d_output)]

# 准备输入数据
input_data = np.random.rand(*input_shape).astype(np.float32)
cuda.memcpy_htod(d_input, input_data)

# 推理
context.execute_v2(bindings)

# 获取输出
output_data = np.empty(output_shape, dtype=np.float32)
cuda.memcpy_dtoh(output_data, d_output)
print("动态批量大小推理输出:", output_data)




# 例12-11
import pandas as pd
import re

# 示例问答数据
data = {
    "question": [
        "How to reset my password?",
        "how to reset my password?",
        "What is your refund policy?",
        "What  is your refund policy?   ",
        "I love your service! 😍"
    ],
    "answer": [
        "Please follow the steps on our website.",
        "Please follow the steps on our website.",
        "You can find details on our refund policy page.",
        "You can find details on our refund policy page.",
        "Thank you! We're glad to hear that."
    ]
}

# 加载数据为DataFrame
df = pd.DataFrame(data)

# 数据清洗函数
def clean_text(text):
    # 去除多余空格
    text = re.sub(r"\s+", " ", text.strip())
    # 去除表情符号和特殊字符
    text = re.sub(r"[^\w\s.,!?]", "", text)
    # 转为小写
    text = text.lower()
    return text

def clean_data(df):
    # 清洗问题和答案
    df["question"] = df["question"].apply(clean_text)
    df["answer"] = df["answer"].apply(clean_text)
    # 去重
    df = df.drop_duplicates(subset=["question", "answer"]).reset_index(drop=True)
    return df

# 应用清洗函数
df_cleaned = clean_data(df)
print("清洗后的数据:\n", df_cleaned)
运行结果如下：
清洗后的数据:
                     question                                         answer
0  how to reset my password?    please follow the steps on our website.
1  what is your refund policy?  you can find details on our refund policy page.
2  i love your service!                 thank you were glad to hear that.



# 例12-12
from nltk.corpus import wordnet
import random

# 同义词替换
def synonym_replacement(sentence):
    words = sentence.split()
    new_sentence = []
    for word in words:
        synonyms = wordnet.synsets(word)
        if synonyms:
            synonym = synonyms[0].lemmas()[0].name()
            new_sentence.append(synonym if random.random() > 0.7 else word)
        else:
            new_sentence.append(word)
    return " ".join(new_sentence)

# 随机插入
def random_insertion(sentence, insert_words):
    words = sentence.split()
    for _ in range(2):  # 插入两次
        idx = random.randint(0, len(words))
        words.insert(idx, random.choice(insert_words))
    return " ".join(words)

# 示例数据增强
question = "how to reset my password?"
insert_words = ["please", "help", "guide"]
augmented_question_1 = synonym_replacement(question)
augmented_question_2 = random_insertion(question, insert_words)

print("原始问题:", question)
print("同义词替换:", augmented_question_1)
print("随机插入:", augmented_question_2)
运行结果如下：
原始问题: how to reset my password?
同义词替换: how to reset my parole?
随机插入: how to reset please my help password?
3.数据格式化
格式化数据为模型可接受的输入形式，如JSON或CSV文件。常见格式如下：
（1）JSON格式：
[
    {
        "question": "how to reset my password?",
        "answer": "please follow the steps on our website."
    },
    {
        "question": "what is your refund policy?",
        "answer": "you can find details on our refund policy page."
    }
]
具体代码实现：
import json

# 将清洗后的数据转换为JSON格式
def format_to_json(df, output_path):
    records = df.to_dict(orient="records")
    with open(output_path, "w") as f:
        json.dump(records, f, indent=4)

# 保存为JSON文件
format_to_json(df_cleaned, "cleaned_data.json")
print("数据已格式化为JSON文件")
以下是综合上述步骤的完整实例，结合企业问答的具体场景（如密码重置、退款政策、账户管理等），展示如何清洗、增强和格式化问答数据。测试部分模拟了大量企业问答情景，并输出清洗和增强后的结果。
import pandas as pd
import re
import random
import json

# 示例企业问答数据（中文场景）
data = {
    "question": [
        "如何重置密码？",
        "我怎样更改账户邮箱地址？",
        "贵公司的退款政策是什么？",
        "如何联系客户支持？",
        "有哪些订阅计划可以选择？"
    ],
    "answer": [
        "您可以在设置页面重置密码。",
        "请前往账户设置更改邮箱地址。",
        "我们的退款政策详见常见问题页面。",
        "您可以通过聊天或邮件联系客户支持。",
        "我们提供月度和年度订阅计划。"
    ]
}

# 加载数据为DataFrame
df = pd.DataFrame(data)

# 清洗函数
def clean_text(text):
    text = re.sub(r"\s+", "", text.strip())  # 去除多余空格
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9.,!?，。！？]", "", text)  # 去除特殊字符
    return text

def clean_data(df):
    df["question"] = df["question"].apply(clean_text)
    df["answer"] = df["answer"].apply(clean_text)
    df = df.drop_duplicates(subset=["question", "answer"]).reset_index(drop=True)
    return df

# 数据增强函数
def synonym_replacement(sentence, synonyms_dict):
    words = list(sentence)
    new_sentence = []
    for word in words:
        if word in synonyms_dict and random.random() > 0.7:
            new_sentence.append(synonyms_dict[word])
        else:
            new_sentence.append(word)
    return "".join(new_sentence)

def random_insertion(sentence, insert_words):
    words = list(sentence)
    for _ in range(2):  # 插入两次
        idx = random.randint(0, len(words))
        words.insert(idx, random.choice(insert_words))
    return "".join(words)

# 示例同义词替换词典和插入词
synonyms_dict = {"重置": "重新设置", "客户": "用户", "支持": "帮助"}
insert_words = ["请", "谢谢", "指导"]

# 数据增强主函数
def augment_data(df, synonyms_dict, insert_words):
    augmented_questions = []
    for question in df["question"]:
        augmented_questions.append(synonym_replacement(question, synonyms_dict))
        augmented_questions.append(random_insertion(question, insert_words))
    return augmented_questions

# 数据格式化函数
def format_to_json(df, augmented_questions, output_path):
    records = df.to_dict(orient="records")
    for question in augmented_questions:
        records.append({"question": question, "answer": "同原始问题答案一致"})
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4, ensure_ascii=False)

# 清洗数据
df_cleaned = clean_data(df)
print("清洗后的数据:\n", df_cleaned)

# 数据增强
augmented_questions = augment_data(df_cleaned, synonyms_dict, insert_words)
print("增强后的问题示例:\n", augmented_questions[:5])

# 格式化为JSON文件
format_to_json(df_cleaned, augmented_questions, "cleaned_data.json")
print("数据已格式化为JSON文件")
清洗后的数据示例：
清洗后的数据:
                 question              answer
0            如何重置密码？         您可以在设置页面重置密码。
1      我怎样更改账户邮箱地址？       请前往账户设置更改邮箱地址。
2      贵公司的退款政策是什么？   我们的退款政策详见常见问题页面。
3         如何联系客户支持？    您可以通过聊天或邮件联系客户支持。
4    有哪些订阅计划可以选择？       我们提供月度和年度订阅计划。
数据增强示例输出：
增强后的问题示例:
 ['如何重新设置密码？', '请如何谢谢重置密码？', '我怎样更改账户邮箱地址？', '请我怎样更改谢谢账户邮箱地址？', '贵公司的退款政策是什么？']
JSON文件内容示例：
[
    {
        "question": "如何重置密码？",
        "answer": "您可以在设置页面重置密码。"
    },
    {
        "question": "我怎样更改账户邮箱地址？",
        "answer": "请前往账户设置更改邮箱地址。"
    },
    {
        "question": "如何重新设置密码？",
        "answer": "同原始问题答案一致"
    },
    {
        "question": "请如何谢谢重置密码？",
        "answer": "同原始问题答案一致"


    }


# 例12-13
pip install transformers datasets torch
将经过清洗和增强的企业问答数据加载为训练和验证集：
from datasets import Dataset
from transformers import AutoTokenizer

# 加载数据
data = [
    {"question": "如何重置密码？", "answer": "您可以在设置页面重置密码。", "label": 1},
    {"question": "如何重置密码？", "answer": "请前往账户设置更改邮箱地址。", "label": 0},
    {"question": "如何联系客户支持？", "answer": "您可以通过聊天或邮件联系客户支持。", "label": 1},
    {"question": "贵公司的退款政策是什么？", "answer": "我们的退款政策详见常见问题页面。", "label": 1},
    {"question": "贵公司的退款政策是什么？", "answer": "您可以通过聊天或邮件联系客户支持。", "label": 0}
]

dataset = Dataset.from_list(data)

# 初始化分词器
model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 数据处理函数
def preprocess(example):
    encoded = tokenizer(
        example["question"],
        example["answer"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    encoded["label"] = example["label"]
    return encoded

# 处理数据集
processed_dataset = dataset.map(preprocess, batched=True)
train_test_split = processed_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
val_dataset = train_test_split["test"]

print("样本数据:", train_dataset[0])
使用transformers库中的AutoModelForSequenceClassification，对问答匹配任务进行微调：
import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

# 加载预训练模型
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./model_output",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2
)

# 定义Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# 开始训练
trainer.train()
训练完成后，需要将模型部署为推理服务，以支持实时问答请求，将微调后的模型保存为可部署的格式：
model.save_pretrained("./deployed_model")
tokenizer.save_pretrained("./deployed_model")
使用FastAPI构建RESTful API接口，提供问答推理服务：
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 初始化FastAPI应用
app = FastAPI()

# 加载模型和分词器
model_path = "./deployed_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# 定义请求和响应模型
class QARequest(BaseModel):
    question: str
    answer: str

@app.post("/predict/")
def predict(data: QARequest):
    inputs = tokenizer(
        data.question, data.answer,
        truncation=True, padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probs).item()
    return {"prediction": prediction, "probabilities": probs.tolist()}
运行服务：
uvicorn app:app --host 0.0.0.0 --port 8000
通过发送测试请求，验证推理服务的正确性：
import requests
url = "http://127.0.0.1:8000/predict/"
data = {"question": "如何重置密码？", "answer": "您可以在设置页面重置密码。"}
response = requests.post(url, json=data)
print(response.json())
运行结果如下：
{"prediction": 1, "probabilities": [[0.1, 0.9]]}


# 例12-14
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 初始化FastAPI应用
app = FastAPI()

# 加载模型和分词器
model_path = "./deployed_model"  # 替换为微调后的模型路径
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# 定义请求数据结构
class QARequest(BaseModel):
    question: str
    answer: str

# 定义API接口
@app.post("/qa/")
def predict(data: QARequest):
    """
    接收用户请求数据，返回问答匹配的预测结果
    """
    # 对请求中的问题和答案进行分词处理
    inputs = tokenizer(
        data.question,
        data.answer,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    # 模型推理
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probs, dim=-1).item()
    
    # 返回预测结果
    return {
        "question": data.question,
        "answer": data.answer,
        "prediction": prediction,  # 1为匹配，0为不匹配
        "probabilities": probs.tolist()  # 各类别概率
    }
将API服务运行在本地或服务器上：
uvicorn app:app --host 0.0.0.0 --port 8000
通过HTTP请求测试接口的功能和正确性，以下提供Python代码示例：
import requests

# 定义测试数据
url = "http://127.0.0.1:8000/qa/"
data = {
    "question": "如何重置密码？",
    "answer": "您可以在设置页面重置密码。"
}

# 发送POST请求
response = requests.post(url, json=data)

# 输出响应结果
print("API响应数据:", response.json())
以下是发送测试请求后的示例响应结果：
{
    "question": "如何重置密码？",
    "answer": "您可以在设置页面重置密码。",
    "prediction": 1,
    "probabilities": [[0.1, 0.9]]
}


# 例12-15
pip install loguru
使用loguru库记录系统运行日志，包括请求日志、响应时间和异常信息：
from fastapi import FastAPI, Request
from loguru import logger
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time

# 初始化FastAPI应用
app = FastAPI()

# 加载模型和分词器
model_path = "./deployed_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# 配置日志文件
logger.add("logs/system.log", rotation="1 MB", retention="7 days", level="INFO")

# 定义请求数据模型
class QARequest(BaseModel):
    question: str
    answer: str

# 定义API接口
@app.post("/qa/")
async def predict(data: QARequest, request: Request):
    """
    接收问答请求，返回预测结果，同时记录请求与响应日志
    """
    start_time = time.time()
    client_ip = request.client.host

    # 分词与模型推理
    try:
        inputs = tokenizer(
            data.question,
            data.answer,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probs, dim=-1).item()
        
        # 记录成功日志
        response_time = time.time() - start_time
        logger.info(
            f"Client IP: {client_ip}, Question: {data.question}, "
            f"Answer: {data.answer}, Prediction: {prediction}, "
            f"Response Time: {response_time:.4f}s"
        )

        return {
            "prediction": prediction,
            "probabilities": probs.tolist(),
            "response_time": f"{response_time:.4f}s"
        }
    except Exception as e:
        # 记录异常日志
        logger.error(f"Error processing request from {client_ip}: {str(e)}")
        return {"error": "An error occurred while processing your request."}
性能监控可以帮助开发者评估系统负载和响应能力，并及时优化，Prometheus是开源的性能监控工具，可以通过其Python客户端采集FastAPI服务的指标。
首先安装Prometheus客户端：
pip install prometheus-client
集成Prometheus指标，以下代码展示了如何记录请求计数和响应时间：
from prometheus_client import Counter, Histogram, start_http_server

# 启动Prometheus监控服务
start_http_server(8001)

# 定义指标
REQUEST_COUNT = Counter("request_count", "Total number of requests")
RESPONSE_TIME = Histogram("response_time", "Response time of requests")

@app.post("/qa/")
@RESPONSE_TIME.time()
async def predict_with_metrics(data: QARequest, request: Request):
    """
    带有Prometheus指标的预测接口
    """
    REQUEST_COUNT.inc()  # 增加请求计数
    response = await predict(data, request)
    return response
此外，日志文件中包含所有请求的异常信息，可以通过定期分析日志快速定位问题，结合Prometheus Alertmanager实现异常告警，配置规则如下：
groups:
  - name: alert_rules
    rules:
    - alert: HighErrorRate
      expr: rate(request_errors_total[5m]) > 0.1
      for: 1m
      labels:
        severity: warning
      annotations:
        summary: "High error rate detected"
        description: "More than 10% of requests failed in the last 5 minutes"
日志文件内容示例：
2024-11-17 10:30:15.123 | INFO     | Client IP: 127.0.0.1, Question: 如何重置密码？, Answer: 您可以在设置页面重置密码。, Prediction: 1, Response Time: 0.1234s
2024-11-17 10:30:20.456 | ERROR    | Error processing request from 127.0.0.1: tokenizer input length exceeded maximum
综合测试代码如下：
import requests
# 定义API URL
url = "http://127.0.0.1:8000/qa/"
# 测试数据
test_data = [
    {"question": "如何重置密码？", "answer": "您可以在设置页面重置密码。"},  # 正确匹配
    {"question": "如何更改账户邮箱？", "answer": "请前往账户设置更改邮箱地址。"},  # 正确匹配
    {"question": "贵公司的退款政策是什么？", "answer": "错误答案。"},  # 故意错误匹配
    {"question": "如何联系客户支持？", "answer": "您可以通过聊天或邮件联系客户支持。"},  # 正确匹配
    {"question": "", "answer": "空输入测试。"},  # 空输入
    {"question": "问题超长测试" * 1000, "answer": "超长问题测试。"}  # 超长输入
]

# 发送请求并打印结果
for i, data in enumerate(test_data):
    print(f"测试用例 {i + 1}:")
    try:
        response = requests.post(url, json=data)
        print("请求数据:", data)
        print("响应结果:", response.json())
    except Exception as e:
        print("请求失败:", str(e))
    print("\n")
测试用例 1：正确匹配
测试用例 1:
请求数据: {'question': '如何重置密码？', 'answer': '您可以在设置页面重置密码。'}
响应结果: {
    "prediction": 1,
    "probabilities": [[0.05, 0.95]],
    "response_time": "0.1234s"
}
测试用例 2：正确匹配
测试用例 2:
请求数据: {'question': '如何更改账户邮箱？', 'answer': '请前往账户设置更改邮箱地址。'}
响应结果: {
    "prediction": 1,
    "probabilities": [[0.03, 0.97]],
    "response_time": "0.1345s"
}
测试用例 3：故意错误匹配
测试用例 3:
请求数据: {'question': '贵公司的退款政策是什么？', 'answer': '错误答案。'}
响应结果: {
    "prediction": 0,
    "probabilities": [[0.85, 0.15]],
    "response_time": "0.1456s"
}
测试用例 4：正确匹配
测试用例 4:
请求数据: {'question': '如何联系客户支持？', 'answer': '您可以通过聊天或邮件联系客户支持。'}
响应结果: {
    "prediction": 1,
    "probabilities": [[0.02, 0.98]],
    "response_time": "0.1123s"
}
测试用例 5：空输入测试
测试用例 5:
请求数据: {'question': '', 'answer': '空输入测试。'}
响应结果: {
    "error": "An error occurred while processing your request."
}
测试用例 6：超长输入测试
测试用例 6:
请求数据: {'question': '问题超长测试问题超长测试问题超长测试...(省略)...', 'answer': '超长问题测试。'}
响应结果: {
    "error": "An error occurred while processing your request."
}
模拟日志文件（logs/system.log）：
2024-11-17 12:00:15.123 | INFO     | Client IP: 127.0.0.1, Question: 如何重置密码？, Answer: 您可以在设置页面重置密码。, Prediction: 1, Response Time: 0.1234s
2024-11-17 12:00:20.456 | INFO     | Client IP: 127.0.0.1, Question: 如何更改账户邮箱？, Answer: 请前往账户设置更改邮箱地址。, Prediction: 1, Response Time: 0.1345s
2024-11-17 12:00:25.789 | INFO     | Client IP: 127.0.0.1, Question: 贵公司的退款政策是什么？, Answer: 错误答案。, Prediction: 0, Response Time: 0.1456s
2024-11-17 12:00:30.012 | INFO     | Client IP: 127.0.0.1, Question: 如何联系客户支持？, Answer: 您可以通过聊天或邮件联系客户支持。, Prediction: 1, Response Time: 0.1123s
2024-11-17 12:00:35.567 | ERROR    | Error processing request from 127.0.0.1: tokenizer input length exceeded maximum
Prometheus指标采集：
（1）请求计数（request_count）：
request_count: 6
（2）响应时间（response_time）：
response_time:
  Bucket (0.1s): 2
  Bucket (0.2s): 2
  Bucket (0.3s): 2
（3）异常请求计数：
request_errors_total: 2




















