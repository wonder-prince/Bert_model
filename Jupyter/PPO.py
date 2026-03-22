import os
import random
import json
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertModel, AutoModelForSequenceClassification


class EmotionDialogueDataset(Dataset):
    """情绪相关对话数据集: {'context': ..., 'reply': ..., 'emotion': ...}。"""

    def __init__(self, records: List[Dict[str, Any]], tokenizer: BertTokenizerFast, max_len=128):
        self.records = records
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        context = record["context"]
        reply = record["reply"]
        emotion = record.get("emotion", "neutral")

        enc = self.tokenizer(
            context,
            reply,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt',
        )

        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "reply_text": reply,
            "emotion": emotion,
        }


class PPOBertPolicy(nn.Module):
    """PPO Policy + Value, BERT Backbone"""

    def __init__(self, model_name='bert-base-uncased', hidden_size=768, action_dim=2):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.policy_head = nn.Linear(hidden_size, action_dim)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output  # [batch, hidden]
        logits = self.policy_head(pooled)
        value = self.value_head(pooled).squeeze(-1)
        return logits, value


def build_emotion_reward_fn(emo_model_name='nlptown/bert-base-multilingual-uncased-sentiment'):
    """情绪评估器，给每条 文本reward"""
    emo_tokenizer = BertTokenizerFast.from_pretrained(emo_model_name)
    emo_model = AutoModelForSequenceClassification.from_pretrained(emo_model_name)
    emo_model.eval()

    def reward_fn(reply_texts: List[str], target_emotion='positive'):
        encoded = emo_tokenizer(reply_texts, truncation=True, padding=True, return_tensors='pt')
        with torch.no_grad():
            outputs = emo_model(**encoded)
        probs = torch.softmax(outputs.logits, dim=-1)
        # nlptown 1-5 stars, 1、2->负向，3 中性，4、5 正向
        if target_emotion == 'positive':
            r = probs[:, 3:].sum(dim=-1)
        elif target_emotion == 'negative':
            r = probs[:, :2].sum(dim=-1)
        else:
            r = probs[:, 2]
        return r.cpu().numpy().tolist()

    return reward_fn


def compute_gae(rewards, values, masks, gamma=0.99, lam=0.95):
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(len(rewards))):
        nextnonterminal = masks[t]
        nextvalues = values[t + 1] if t + 1 < len(values) else 0
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
    returns = advantages + values
    return advantages, returns


def ppo_update(model, optimizer, old_log_probs, states, actions, returns, advantages,
               clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01, epoch=4, batch_size=16):
    dataset = torch.utils.data.TensorDataset(states['input_ids'], states['attention_mask'], actions, old_log_probs, returns, advantages)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for _ in range(epoch):
        for batch in dataloader:
            input_ids, attention_mask, action_batch, old_log_prob_batch, return_batch, advantage_batch = batch

            logits, values = model(input_ids=input_ids, attention_mask=attention_mask)
            dist = torch.distributions.Categorical(logits=logits)
            log_prob = dist.log_prob(action_batch)
            entropy = dist.entropy().mean()

            ratio = torch.exp(log_prob - old_log_prob_batch)
            surr1 = ratio * advantage_batch
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage_batch
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = (return_batch - values).pow(2).mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

    return loss.item(), policy_loss.item(), value_loss.item(), entropy.item()


def collect_batch(model, tokenizer, env_data, reward_fn, device, batch_size=16):
    model.eval()
    states = {'input_ids': [], 'attention_mask': []}
    actions = []
    log_probs = []
    values = []
    rewards = []
    masks = []

    # 这里做简化：每个 sample 视为一步 episode
    for rec in random.sample(env_data, batch_size):
        enc = tokenizer(rec['context'], rec['reply'], truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        input_ids = enc['input_ids'].to(device)
        attention_mask = enc['attention_mask'].to(device)

        with torch.no_grad():
            logits, value = model(input_ids=input_ids, attention_mask=attention_mask)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()  # from response_candidates(示意)
            log_prob = dist.log_prob(action)

        states['input_ids'].append(input_ids.squeeze(0).cpu())
        states['attention_mask'].append(attention_mask.squeeze(0).cpu())
        actions.append(action.item())
        log_probs.append(log_prob.item())
        values.append(value.item())

        # 使用情绪奖励函数
        r = reward_fn([rec['reply']], target_emotion=rec.get('emotion', 'positive'))[0]
        rewards.append(r)
        masks.append(1.0)

    for k in states:
        states[k] = torch.stack(states[k])
    actions = torch.tensor(actions, dtype=torch.int64)
    old_log_probs = torch.tensor(log_probs, dtype=torch.float32)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    values = torch.tensor(values, dtype=torch.float32)
    masks = torch.tensor(masks, dtype=torch.float32)

    advantages, returns = compute_gae(rewards, values, masks)

    return states, actions, old_log_probs, returns, advantages


def load_sample_data() -> List[Dict[str, Any]]:
    # 这里给出简化示例；实际请替换为情绪对话数据集 jsonl(tsv)形式
    return [
        {
            'context': '今天我心情很低落，我不知道怎么办。',
            'reply': '抱抱你，能够理解你的感受，试着深呼吸和倾诉一下。',
            'emotion': 'positive',
        },
        {
            'context': '我很生气，工作压力太大了。',
            'reply': '这种压力确实很难受，我们可以一起拆解问题。',
            'emotion': 'positive',
        },
        {
            'context': '我觉得很孤独，没有朋友。',
            'reply': '你不是一个人，我在这里听你说。',
            'emotion': 'positive',
        },
        {
            'context': '我想逃避一切。',
            'reply': '这个情绪可以慢慢调试，我们先从小目标开始。',
            'emotion': 'positive',
        },
    ]


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = PPOBertPolicy(model_name='bert-base-uncased', action_dim=5).to(device)

    reward_fn = build_emotion_reward_fn('nlptown/bert-base-multilingual-uncased-sentiment')

    env_data = load_sample_data()

    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    ppo_steps = 120
    for step in range(1, ppo_steps + 1):
        states, actions, old_log_probs, returns, advantages = collect_batch(
            model=model,
            tokenizer=tokenizer,
            env_data=env_data,
            reward_fn=reward_fn,
            device=device,
            batch_size=min(64, len(env_data)),
        )

        loss, policy_loss, value_loss, entropy = ppo_update(
            model=model,
            optimizer=optimizer,
            old_log_probs=old_log_probs,
            states=states,
            actions=actions,
            returns=returns,
            advantages=advantages,
            clip_epsilon=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            epoch=4,
            batch_size=8,
        )

        if step % 10 == 0:
            # print(f"Step {step}, loss {loss:.4f}")
            print(f"Step {step}, policy={policy_loss:.4f}, value={value_loss:.4f}, entropy={entropy:.4f}, total={loss:.4f}")

    os.makedirs('output', exist_ok=True)
    torch.save(model.state_dict(), 'output/ppo_bert_emotion.pth')
    print('训练完毕，模型已保存到 output/ppo_bert_emotion.pth')


if __name__ == '__main__':
    main()
