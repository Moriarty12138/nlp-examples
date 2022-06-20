#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/16 10:24
# @Author  : jiaguoqing 
# @Email   : jiaguoqing12138@gmail.com
# @File    : datasets.py
import ast

import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer

from set_args import get_args

template = {
    "Experiment": "请找出句子中的试验",
    "Manoeuvre": "请找出句子中的演习",
    "Deploy": "请找出句子中的部署",
    "Indemnity": "请找出句子中的保障",
    "Support": "请找出句子中的支援",
    "Accident": "请找出句子中的意外事故",
    "Exhibit": "请找出句子中的展示",
    "Non-event": "句子中是否不包含事件",
}


class EventExtractionDataset(Dataset):
    def __init__(
            self,
            dataset_path,
            model_name_or_path,
            max_len,
    ):
        super(EventExtractionDataset, self).__init__()
        print("Load tokenizer.")
        self.tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
        self.dataset = []

        print("Load dataset.")
        df = pd.read_csv(dataset_path, converters={'token': ast.literal_eval})
        for idx, row in df.iterrows():
            # print("process dataset {}.".format(idx))
            # 问题模板
            input_ids1 = self.tokenizer.encode(template[row['event_type']])
            token_type_ids1 = [0] * len(input_ids1)
            start_ids1 = [0] * len(input_ids1)
            end_ids1 = [0] * len(input_ids1)

            # 输入原文
            input_ids2 = self.tokenizer.convert_tokens_to_ids(row['token']) +\
                              [self.tokenizer.sep_token_id]
            token_type_ids2 = [1] * len(input_ids2)
            start_ids2 = [0] * len(input_ids2)
            end_ids2 = [0] * len(input_ids2)
            start_ids2[row['start']] = 1
            end_ids2[row['end']] = 1

            input_ids = input_ids1 + input_ids2
            token_type_ids = token_type_ids1 + token_type_ids2
            start_ids = start_ids1 + start_ids2
            end_ids = end_ids1 + end_ids2
            attention_mask = [1] * len(input_ids)
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
                token_type_ids = token_type_ids[:max_len]
                start_ids = start_ids[:max_len]
                end_ids = end_ids[:max_len]
                input_ids[-1] = self.tokenizer.sep_token_id
                start_ids[-1] = 0
                end_ids[-1] = 0

            self.dataset.append({
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
                "start_ids": start_ids,
                "end_ids": end_ids,
            })

            # # 数据扩展，不属于类别的将其置不存在触发词
            # start_ids_ = [0 for  _ in start_ids]
            # end_ids_ = [0 for _ in end_ids]
            # for k, v in template.items():
            #     if k != row['event_type']:
            #         self.dataset.append({
            #             "input_ids": input_ids,
            #             "token_type_ids": token_type_ids,
            #             "attention_mask": attention_mask,
            #             "start_ids": start_ids_,
            #             "end_ids": end_ids_,
            #         })
        # torch.save(self.dataset, dataset_path[:-3]+"pt")

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)


def collate_fn(batch):
    input_ids_batch, token_type_ids_batch, attention_mask_batch, start_ids_batch, end_ids_batch = [], [], [], [], []
    for example in batch:
        input_ids_batch.append(torch.tensor(example['input_ids'], dtype=torch.long))
        token_type_ids_batch.append(torch.tensor(example['token_type_ids'], dtype=torch.long))
        attention_mask_batch.append(torch.tensor(example['attention_mask'], dtype=torch.long))
        start_ids_batch.append(torch.tensor(example['start_ids'], dtype=torch.long))
        end_ids_batch.append(torch.tensor(example['end_ids'], dtype=torch.long))
    # pad
    input_ids_batch = pad_sequence(input_ids_batch, batch_first=True, padding_value=0)
    token_type_ids_batch = pad_sequence(token_type_ids_batch, batch_first=True, padding_value=1)
    attention_mask_batch = pad_sequence(attention_mask_batch, batch_first=True, padding_value=0)
    start_ids_batch = pad_sequence(start_ids_batch, batch_first=True, padding_value=0)
    end_ids_batch = pad_sequence(end_ids_batch, batch_first=True, padding_value=0)
    return {
        "input_ids": input_ids_batch,
        "token_type_ids": token_type_ids_batch,
        "attention_mask": attention_mask_batch,
        "start_ids": start_ids_batch,
        "end_ids": end_ids_batch,
    }


if __name__ == '__main__':
    args = get_args()
    train_dataset = EventExtractionDataset(
        args.valid_dataset_path, args.model_name_or_path, args.max_len)
    print(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    for batch in train_dataloader:
        print(batch)
