from datasets import load_dataset
from transformers import BertTokenizer,BertForQuestionAnswering
# 使用SQuAD数据集来微调
datasets = load_dataset("squad_v2")
# 加载分词器和模型
tokenizer =  BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForQuestionAnswering.from_pretrainedd("bert-base-uncased")

# 数据预处理
def preprocess_data(examples):
    # 分词器设置
    inputs = tokenizer(examples['question'], 
                       examples['context'], 
                       truncation="only_second", 
                       padding='max_length', 
                       max_length=384,
                       return_offsets_mapping=True,
                       stride=128,
                       return_overflowing_tokens=True,
                       )
    # 初始化位置
    start_positions = []
    end_positions = []
    
    for i, offset in enumerate(inputs['offset_mapping']):
        input_ids = inputs['input_ids'][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        
        answer = examples['answers'][i] 
        if len(answer['answer_start']) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            start_char = answer['answer_start'][0]
            end_char = start_char + len(answer['text'][0])
            
            token_start_index,token_end_index = 0, 0
            for j,(start, end) in enumerate(offset):
                if start <= start_char < end:
                    token_start_index = j
                if start < end_char <= end:
                    token_end_index = j
            
            start_positions.append(token_start_index)
            end_positions.append(token_end_index)
    inputs['start_positions'] = start_positions
    inputs['end_positions'] = end_positions 
    return inputs

# 应用预处理函数
tokenized_dataset = datasets.map(preprocess_data, batched=True)

# 数据集切分
train_dataset = tokenized_dataset['train']
eval_dataset = tokenized_dataset['validation']

