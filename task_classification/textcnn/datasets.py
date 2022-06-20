#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/15 14:51
# @Author  : jiaguoqing 
# @Email   : jiaguoqing12138@gmail.com
# @File    : datasets.py

import pandas as pd
import torch

from pandas import DataFrame
from torch.utils import data
from utils import load_dataset
from tokenizers import TextCNNTokenizer


class TextCNNDataset(data.Dataset):
    def __init__(self, data_path, tokenizer: TextCNNTokenizer, max_len: int):
        super(TextCNNDataset, self).__init__()
        if isinstance(data_path, DataFrame):
            self.sentences, self.labels = data_path['question_text'].tolist(), data_path['target'].tolist()
        else:
            self.sentences, self.labels = load_dataset(dataset_path=data_path)
        self.tokenizer = tokenizer
        self.sentences = [self.tokenizer.words2id(sentence.lower()) for sentence in self.sentences]
        self.sentences = [sentence + [0 for _ in range(max_len-len(sentence))]
                          if len(sentence) < max_len else sentence[:max_len] for sentence in self.sentences]

    def __getitem__(self, item):
        return self.sentences[item], self.labels[item]

    def __len__(self):
        return len(self.labels)


def collate_fn(batch):
    # sentences, label = batch
    sentences, labels = [], []
    for sen, label in batch:
        sentences.append(sen)
        labels.append(label)
    return torch.tensor(sentences), torch.tensor([labels])


if __name__ == '__main__':
    df = pd.read_csv("../../datasets/Quora_Insincere_Questions_Classification/train.csv")
    tokenizer = TextCNNTokenizer()
    dataset = TextCNNDataset(df, tokenizer, max_len=32)
    dataloader = data.DataLoader(dataset=dataset, batch_size=8)
    for batch in dataloader:
        sentences, labels = batch
        print(sentences)

