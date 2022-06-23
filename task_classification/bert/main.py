#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/23 16:49
# @Author  : jiaguoqing 
# @Email   : jiaguoqing12138@gmail.com
# @File    : main.py
import json
import torch
from torch import nn
from torch.utils import data
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup

event_type_map = {
    "Experiment": 0,
    "Manoeuvre": 1,
    "Deploy": 2,
    "Indemnity": 3,
    "Support": 4,
    "Accident": 5,
    "Exhibit": 6,
    "Non-event": 7,
}
model_name_or_path = "../../models/chinese_L-12_H-768_A-12"
train_dataset_path = "../../datasets/Event_Competition/train_7000.json"
n_epochs = 10
lr = 5e-5
warmup_proportion = 0.1
max_grad_norm=1.0
device = 'cpu'


def load_json(path):
    with open(path, 'r', encoding='utf-8') as reader:
        js = json.load(reader)
    return js


class BertDataset(data.Dataset):
    def __init__(self, dataset_path, tokenizer: BertTokenizer):
        js = load_json(dataset_path)
        self.examples = []
        self.labels = []
        for j in js:
            d = tokenizer(j['sentence'], max_length=128, padding='max_length', return_token_type_ids=True,
                          truncation=True,)
            # print(d)
            l = event_type_map[j["event_mention"]["event_type"]] if 'trigger' in j["event_mention"] else 8
            self.examples.append(d)
            self.labels.append(l)

    def __getitem__(self, item):
        return self.examples[item], self.labels[item]

    def __len__(self):
        return len(self.labels)


def collate_fn(batch):
    input_ids, token_type_ids, attention_mask, labels = [], [], [], []
    for b in batch:
        example, label = b
        input_ids.append(example['input_ids'])
        token_type_ids.append(example['token_type_ids'])
        attention_mask.append(example['attention_mask'])
        labels.append(label)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    return {
        'input_ids': input_ids,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

model = BertForSequenceClassification.from_pretrained(model_name_or_path, )
tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
model = model.to(device)

train_dataset = BertDataset(train_dataset_path, tokenizer)
train_dataloader = data.DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

total_steps = len(train_dataloader) * n_epochs
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_group_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_group_parameters, lr=lr, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=int(warmup_proportion * total_steps), num_training_steps=total_steps)


for ep in range(n_epochs):
    model.train()
    epoch_loss = []
    for i_batch, batch in enumerate(train_dataloader):
        input_ids = batch['input_ids'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        labels = labels.to(device)

        output = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)
        loss = output.loss

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        if i_batch % 10 == 0:
            print("Epoch=%d,batch_idx=%d,loss=%.4f" % (ep, i_batch, loss.item()))
        epoch_loss.append(loss.item())
        optimizer.step()
        scheduler.step()
