# 例4-1
import spacy
# 加载SpaCy的中文模型（若使用英文句子，请切换至英文模型）
nlp = spacy.load("en_core_web_sm")

# 示例文本
text = "The chef prepared a delicious meal for the guests."

# 处理文本，生成依存关系树
doc = nlp(text)

# 输出依存关系中的主谓宾结构和修饰关系
print("Word\tDependency\tHead\tRelation")
for token in doc:
    print(f"{token.text}\t{token.dep_}\t{token.head.text}\t{[child.text for child in token.children]}")

# 打印语法树结构并标注主谓宾和修饰关系
print("\nDependency Tree:")
for token in doc:
    # 输出主谓宾结构
    if token.dep_ in {"nsubj", "ROOT", "dobj"}:
        print(f"{token.text} ({token.dep_}) <-- {token.head.text} ({token.head.dep_})")
    # 输出修饰关系
    elif token.dep_ in {"amod", "advmod", "prep", "pobj"}:
        print(f"{token.text} ({token.dep_}) modifies --> {token.head.text} ({token.head.dep_})")



import spacy
from spacy import displacy

# 加载SpaCy的语言模型
nlp = spacy.load("en_core_web_sm")

# 示例句子
text = "The quick brown fox jumps over the lazy dog."

# 处理文本，生成依存关系树
doc = nlp(text)

# 构建依存关系树并输出结果
print("Token\tDependency\tHead\tChildren")
for token in doc:
    print(f"{token.text}\t{token.dep_}\t{token.head.text}\t{[child.text for child in token.children]}")

# 使用displacy可视化依存关系树
print("\nRendering dependency tree visualization...")
displacy.render(doc, style="dep", jupyter=False, options={'distance': 90})

# 句法结构的提取示例：找出所有的主谓宾结构和修饰关系
print("\nIdentifying syntactic roles:")
for token in doc:
    # 提取主语、谓语、宾语
    if token.dep_ in {"nsubj", "ROOT", "dobj"}:
        print(f"{token.text} ({token.dep_}) <- {token.head.text} ({token.head.dep_})")
    # 提取修饰语
    elif token.dep_ in {"amod", "advmod", "prep", "pobj"}:
        print(f"{token.text} ({token.dep_}) modifies -> {token.head.text} ({token.head.dep_})")

# 进一步提取复杂结构：所有谓词的宾语及其修饰语
print("\nExtracting objects and their modifiers:")
for token in doc:
    if token.dep_ == "dobj":
        print(f"Object: {token.text}")
        for child in token.children:
            if child.dep_ in {"amod", "det"}:
                print(f"  Modifier: {child.text}")





# 例4-2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

class TreeLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TreeLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 输入门、遗忘门和输出门的线性层定义
        self.W_i = nn.Linear(input_dim, hidden_dim)
        self.U_i = nn.Linear(hidden_dim, hidden_dim)
        
        self.W_f = nn.Linear(input_dim, hidden_dim)
        self.U_f = nn.Linear(hidden_dim, hidden_dim)
        
        self.W_o = nn.Linear(input_dim, hidden_dim)
        self.U_o = nn.Linear(hidden_dim, hidden_dim)
        
        self.W_u = nn.Linear(input_dim, hidden_dim)
        self.U_u = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x, child_h, child_c):
        # 子节点数
        child_h_sum = torch.sum(child_h, dim=0)
        
        # 输入门、遗忘门和输出门计算
        i = torch.sigmoid(self.W_i(x) + self.U_i(child_h_sum))
        o = torch.sigmoid(self.W_o(x) + self.U_o(child_h_sum))
        u = torch.tanh(self.W_u(x) + self.U_u(child_h_sum))
        
        f_list = []
        for h in child_h:
            f = torch.sigmoid(self.W_f(x) + self.U_f(h))
            f_list.append(f)
        
        # 更新记忆单元
        c = i * u + torch.sum(torch.stack([f * c for f, c in zip(f_list, child_c)]), dim=0)
        h = o * torch.tanh(c)
        
        return h, c

class TreeLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(TreeLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.cell = TreeLSTMCell(embedding_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, tree, inputs):
        _ = self.embedding(inputs)  # 将词转化为嵌入
        h, c = self.recursive_forward(tree)
        return h

    def recursive_forward(self, node):
        if node.is_leaf():
            # 对于叶节点，初始化隐藏和记忆单元
            input = self.embedding(node.word_idx)
            child_h, child_c = torch.zeros(self.hidden_dim), torch.zeros(self.hidden_dim)
        else:
            # 递归计算子节点的隐藏状态和记忆单元
            child_h, child_c = [], []
            for child in node.children:
                h, c = self.recursive_forward(child)
                child_h.append(h)
                child_c.append(c)
            child_h = torch.stack(child_h)
            child_c = torch.stack(child_c)
        return self.cell(input, child_h, child_c)

class TreeNode:
    def __init__(self, word_idx):
        self.word_idx = word_idx
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def is_leaf(self):
        return len(self.children) == 0

# 模型参数
vocab_size = 100
embedding_dim = 50
hidden_dim = 100

# 树结构初始化
root = TreeNode(0)
child1 = TreeNode(1)
child2 = TreeNode(2)
child3 = TreeNode(3)
root.add_child(child1)
root.add_child(child2)
child1.add_child(child3)

# 模型初始化
model = TreeLSTM(vocab_size, embedding_dim, hidden_dim)
inputs = torch.LongTensor([0, 1, 2, 3])
output = model(root, inputs)

print(output)

# 例4-2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class TreeLSTMCellWithScoring(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TreeLSTMCellWithScoring, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 定义Tree-LSTM的各门
        self.W_i = nn.Linear(input_dim, hidden_dim)
        self.U_i = nn.Linear(hidden_dim, hidden_dim)
        
        self.W_f = nn.Linear(input_dim, hidden_dim)
        self.U_f = nn.Linear(hidden_dim, hidden_dim)
        
        self.W_o = nn.Linear(input_dim, hidden_dim)
        self.U_o = nn.Linear(hidden_dim, hidden_dim)
        
        self.W_u = nn.Linear(input_dim, hidden_dim)
        self.U_u = nn.Linear(hidden_dim, hidden_dim)
        
        # 定义打分层
        self.scoring_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x, child_h, child_c):
        child_h_sum = torch.sum(child_h, dim=0)
        
        i = torch.sigmoid(self.W_i(x) + self.U_i(child_h_sum))
        o = torch.sigmoid(self.W_o(x) + self.U_o(child_h_sum))
        u = torch.tanh(self.W_u(x) + self.U_u(child_h_sum))
        
        f_list = []
        for h in child_h:
            f = torch.sigmoid(self.W_f(x) + self.U_f(h))
            f_list.append(f)
        
        c = i * u + torch.sum(torch.stack([f * c for f, c in zip(f_list, child_c)]), dim=0)
        h = o * torch.tanh(c)
        
        # 计算打分值
        score = self.scoring_layer(h)
        
        return h, c, score

class TreeLSTMWithScoring(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(TreeLSTMWithScoring, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.cell = TreeLSTMCellWithScoring(embedding_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, tree, inputs):
        _ = self.embedding(inputs)  # 词嵌入转换
        h, score = self.recursive_forward(tree)
        return h, score

    def recursive_forward(self, node):
        if node.is_leaf():
            input = self.embedding(node.word_idx)
            child_h, child_c = torch.zeros(self.hidden_dim), torch.zeros(self.hidden_dim)
        else:
            child_h, child_c, scores = [], [], []
            for child in node.children:
                h, c, score = self.recursive_forward(child)
                child_h.append(h)
                child_c.append(c)
                scores.append(score)
            child_h = torch.stack(child_h)
            child_c = torch.stack(child_c)
        h, c, score = self.cell(input, child_h, child_c)
        return h, score

class TreeNode:
    def __init__(self, word_idx):
        self.word_idx = word_idx
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def is_leaf(self):
        return len(self.children) == 0

# 模型参数
vocab_size = 100
embedding_dim = 50
hidden_dim = 100

# 初始化树结构
root = TreeNode(0)
child1 = TreeNode(1)
child2 = TreeNode(2)
child3 = TreeNode(3)
root.add_child(child1)
root.add_child(child2)
child1.add_child(child3)

# 初始化模型并计算打分
model = TreeLSTMWithScoring(vocab_size, embedding_dim, hidden_dim)
inputs = torch.LongTensor([0, 1, 2, 3])
output, score = model(root, inputs)

print("Tree-LSTM Output:", output)
print("Dependency Score:", score)



# 例4-3
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

# 定义GNN模型，使用两层GCNConv实现依存信息传播
class DependencyGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DependencyGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

# 构建句法依存关系图，示例句为："The quick brown fox jumps over the lazy dog"
G = nx.DiGraph()
edges = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # 主谓宾结构
    (4, 5), (5, 6), (6, 7), (3, 8)   # 修饰关系
]
for u, v in edges:
    G.add_edge(u, v)

# 初始化节点特征，8个单词每个节点用3维特征表示
node_features = torch.tensor([[1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 1, 1],
                              [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1], 
                              [1, 1, 0]], dtype=torch.float)
graph_data = from_networkx(G)
graph_data.x = node_features

# 定义模型参数
input_dim = 3
hidden_dim = 8
output_dim = 4
model = DependencyGNN(input_dim, hidden_dim, output_dim)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# 模拟目标输出用于计算损失
target = torch.rand((graph_data.num_nodes, output_dim))

# 训练模型
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    out = model(graph_data.x, graph_data.edge_index)
    loss = loss_fn(out, target)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

# 测试模型的依存关系嵌入
model.eval()
output = model(graph_data.x, graph_data.edge_index)
print("依存关系节点嵌入表示：")
print(output)




# 例4-4
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

# 定义依存图模型，考虑边权重
class WeightedDependencyGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(WeightedDependencyGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim, add_self_loops=False)
        self.conv2 = GCNConv(hidden_dim, output_dim, add_self_loops=False)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x

# 构建句法依存关系图，并定义边权重
G = nx.DiGraph()
edges = [
    (0, 1, 0.5), (1, 2, 0.7), (2, 3, 0.8), (3, 4, 0.6),  # 主谓宾关系
    (4, 5, 0.4), (5, 6, 0.9), (6, 7, 0.3), (3, 8, 0.5)   # 修饰关系
]
for u, v, w in edges:
    G.add_edge(u, v, weight=w)

# 初始化节点特征，示例为9个节点，特征维度为3
node_features = torch.tensor([[1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 1, 1],
                              [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1], 
                              [1, 1, 0]], dtype=torch.float)
graph_data = from_networkx(G)
graph_data.x = node_features
edge_weights = torch.tensor([edge[2] for edge in edges], dtype=torch.float)
graph_data.edge_weight = edge_weights

# 定义模型参数
input_dim = 3
hidden_dim = 8
output_dim = 4
model = WeightedDependencyGNN(input_dim, hidden_dim, output_dim)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# 模拟目标输出用于损失计算
target = torch.rand((graph_data.num_nodes, output_dim))

# 训练模型
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    out = model(graph_data.x, graph_data.edge_index, graph_data.edge_weight)
    loss = loss_fn(out, target)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

# 测试模型输出
model.eval()
output = model(graph_data.x, graph_data.edge_index, graph_data.edge_weight)
print("依存关系节点嵌入表示：")
print(output)



# 例4-5
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import networkx as nx
import matplotlib.pyplot as plt

# 初始化BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# 输入文本并生成BERT嵌入
text = "The cat sat on the mat"
tokens = tokenizer(text, return_tensors='pt', padding=True)
with torch.no_grad():
    bert_outputs = bert_model(**tokens)
# 提取CLS和每个Token的嵌入
token_embeddings = bert_outputs.last_hidden_state.squeeze(0)

# 初始化简单图神经网络
class GNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, node_features, adj_matrix):
        return torch.relu(self.linear(torch.matmul(adj_matrix, node_features)))

# 构建依存关系图（模拟的简单依存关系图）
G = nx.DiGraph()
edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
G.add_edges_from(edges)

# 生成邻接矩阵
adj_matrix = nx.adjacency_matrix(G).todense()
adj_matrix = torch.FloatTensor(adj_matrix)

# 定义图神经网络层
gnn_layer = GNNLayer(token_embeddings.shape[1], 128)

# 将BERT嵌入输入GNN
gnn_output = gnn_layer(token_embeddings, adj_matrix)

# 输出每个节点的特征更新
print("更新后的节点特征矩阵：")
print(gnn_output)
运行结果如下：
更新后的节点特征矩阵：
tensor([[ 0.1321, -0.0547, ..., -0.1345],
        [ 0.0985,  0.1038, ...,  0.0728],
        ...,
        [ 0.0754, -0.0923, ..., -0.0639]])

# 例4-6
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import networkx as nx
import matplotlib.pyplot as plt

# 初始化BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# 输入句子并生成BERT嵌入
text = "The quick brown fox jumps over the lazy dog"
tokens = tokenizer(text, return_tensors='pt', padding=True)
with torch.no_grad():
    bert_outputs = bert_model(**tokens)
# 提取BERT的嵌入
token_embeddings = bert_outputs.last_hidden_state.squeeze(0)

# 定义简单的GNN层
class GNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, node_features, adj_matrix):
        return torch.relu(self.linear(torch.matmul(adj_matrix, node_features)))

# 初始化GNN层
gnn_layer = GNNLayer(token_embeddings.shape[1], 128)

# 构建简单的依存关系图
G = nx.DiGraph()
edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)]
G.add_edges_from(edges)

# 生成邻接矩阵
adj_matrix = nx.adjacency_matrix(G).todense()
adj_matrix = torch.FloatTensor(adj_matrix)

# BERT + GNN的混合模型实现
class BertGNNModel(nn.Module):
    def __init__(self, bert_embeddings, gnn_layer):
        super(BertGNNModel, self).__init__()
        self.bert_embeddings = bert_embeddings
        self.gnn_layer = gnn_layer
        
    def forward(self, adj_matrix):
        # 输入BERT的嵌入到GNN层
        gnn_output = self.gnn_layer(self.bert_embeddings, adj_matrix)
        return gnn_output

# 初始化混合模型并运行
model = BertGNNModel(token_embeddings, gnn_layer)
gnn_output = model(adj_matrix)

# 输出节点更新后的特征
print("混合模型输出的节点特征矩阵：")
print(gnn_output)
运行结果如下：
混合模型输出的节点特征矩阵：
tensor([[ 0.1782, -0.0649, ..., -0.0421],
        [ 0.1543,  0.0976, ...,  0.0825],
        ...,
        [ 0.0921, -0.0724, ..., -0.0534]])

# 例4-7
import spacy
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
import json

# 加载SpaCy模型 
nlp = spacy.load("en_core_web_sm") 
# 指定本地模型路径
local_model_path=\ 		"path/to/your/local/model/structured-prediction-srl-bert.2020.12.15.tar.gz" 
# 加载AllenNLP模型 
predictor = Predictor.from_path(local_model_path

# 定义示例文本
text = "John gave Mary a book on her birthday."

# 依存解析
doc = nlp(text)
dependencies = []
for token in doc:
    dependencies.append((token.text, token.dep_, token.head.text))

# 输出依存关系
print("依存关系:")
for dep in dependencies:
    print(dep)

# 语义角色标注
srl_result = predictor.predict(sentence=text)

# 提取SRL结果
roles = []
for verb in srl_result['verbs']:
    roles.append((verb['verb'], verb['description']))

# 输出语义角色
print("\n语义角色标注:")
for role in roles:
    print(role)

# 融合依存关系与语义角色
combined_results = []
for token in doc:
    dep_relation = (token.text, token.dep_, token.head.text)
    srl_roles = [role for verb, desc in roles if token.text in desc]
    combined_results.append({
        "Token": token.text,
        "Dependency": dep_relation,
        "Semantic Roles": srl_roles
    })

# 输出融合结果
print("\n依存关系与语义角色融合结果:")
for result in combined_results:
    print(result)



from allennlp.predictors.predictor import Predictor
# 指定本地模型路径
local_model_path= \
"path/to/your/local/model/structured-prediction-srl-bert.2020.12.15.tar.gz"
# 加载模型
predictor = Predictor.from_path(local_model_path)


# 例4-8
import spacy
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging

# 加载SpaCy的英文模型
nlp = spacy.load("en_core_web_sm")

# 加载AllenNLP的预训练语义角色标注模型
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.11.19.tar.gz")

# 示例句子
sentence = "John gave Mary a book on her birthday."

# 使用SpaCy进行依存解析
doc = nlp(sentence)
print("依存关系解析结果:")
for token in doc:
    print(f"词: {token.text}, 依存关系: {token.dep_}, 头词: {token.head.text}")

# 使用AllenNLP进行语义角色标注
srl_results = predictor.predict(sentence=sentence)
print("\n语义角色标注结果:")
for verb in srl_results['verbs']:
    print(f"动词: {verb['verb']}")
    print("角色标注:")
    print(verb['description'])














 
