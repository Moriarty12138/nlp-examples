#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import hashlib
import random

import torch
from torch.utils import data
from itertools import groupby


def load_data(path):
    with open(path, 'r', encoding='utf-8') as reader:
        data_ = [s.strip().lower() for s in reader.readlines()]
    return data_


class DataPair(data.Dataset):
    """ 数据处理 """
    def __init__(
            self, data_0_path, data_1_path, min_word_count=4,
            base_corpus=None, model_path="./outputs/", amount=1, device=None):
        data_0 = load_data(data_0_path)
        data_1 = load_data(data_1_path)

        # 持久化
        corpus = "  ".join(data_0+data_1)
        xcode = hashlib.sha1("{}-{}-{}-{}".format(
            corpus, min_word_count, device, base_corpus.xcode if base_corpus is not None else 0).encode('utf-8'))
        self.xcode = int(xcode.hexdigest(), 16) % 10 ** 8
        model_file_path = "{}DataPair_{}.pk".format(model_path, self.xcode)

        if os.path.exists(model_file_path):  # 数据已经保存
            info = torch.load(model_file_path)
            print("load exist data : {}".format(model_file_path))
            self.xcode = info['xcode']
            self.data_0 = info['data_0']
            self.data_1 = info['data_1']
            self.word_id = info['word_id']
            self.id_word = info['id_word']
        else:
            self._make_dic(corpus, min_word_count, base_corpus)  # 构建词典
            label_0 = torch.tensor([1.0, -1.0], device=device)
            label_1 = torch.tensor([-1.0, 1.0], device=device)
            self.data_0 = [self.sentence_to_tensor(s.split(" "), device=device) for s in data_0]
            self.data_1 = [self.sentence_to_tensor(s.split(" "), device=device) for s in data_1]
            info = {
                "xcode": self.xcode,
                "data_0": self.data_0,
                "data_1": self.data_1,
                "word_id": self.word_id,
                "id_word": self.id_word,
            }
            torch.save(info, model_file_path)

        self.data_0 = info["data_0"][:int(len(self.data_0) * amount)]
        self.data_1 = info["data_1"][:int(len(self.data_1) * amount)]
        self.vocab_size = len(self.word_id)
        self.data_0_len = len(self.data_0)
        self.data_1_len = len(self.data_1)

    def _make_dic(self, corpus, min_word_count, base_corpus=None):
        if base_corpus:  # 使用基础语料库
            self.word_id = base_corpus.word_id
            self.id_word = base_corpus.id_word
        else:
            words = corpus.split(" ")
            words = sorted(words)
            group = groupby(words)
            word_count = [(w, sum(1 for _ in c)) for w, c in group]
            word_count = [(w, c) for w, c in word_count if c >= min_word_count]
            word_count.sort(key=lambda x: x[1], reverse=True)
            word_id = dict([(w, i+4) for i, (w, _) in enumerate(word_count)])
            word_id['<pad>'] = 0
            word_id['<unk>'] = 1
            word_id['<sos>'] = 2
            word_id['<eos>'] = 3
            self.word_id = word_id
            self.id_word = dict([(i, w) for w, i in word_id.items()])

    def sentence_to_tensor(self, sentence, device):
        v = [self.word_id.get(w, 1) for w in sentence]
        v = [2] + v + [3]
        v = torch.tensor(v, device=device)
        return v

    def shuffle(self):
        random.shuffle(self.data_0)
        random.shuffle(self.data_1)

    def to_text(self, sen):
        text = [self.id_word[i] for i in sen]
        return " ".join(text)

    def __getitem__(self, idx):
        idx_0 = idx_1 = idx
        if idx_0 > self.data_0_len:
            idx_0 = random.randint(0, self.data_0_len-1)
        if idx_1 > self.data_1_len:
            idx_1 = random.randint(0, self.data_1_len-1)
        print(self.data_0[idx_0], self.data_1[idx_1])
        return self.data_0[idx_0], self.data_1[idx_1]

    def __len__(self):
        return max(self.data_0_len, self.data_1_len)


def pad_sequences(sequences, bsz, max_len, pad_id=0):
    out_tensor = torch.tensor([[0 for _ in range(max_len)] for _ in range(bsz)], dtype=torch.long)
    for i, tens in enumerate(sequences):
        length = min(max_len, tens.size(0))
        out_tensor[i, :length, ...] = tens[:length]
    return out_tensor


def collate_fn(batch):
    data_0s = []
    data_1s = []
    for data_0, data_1 in batch:
        data_0s.append(data_0)
        data_1s.append(data_1)
    max_len_0 = max([data_0.size() for data_0 in data_0s])
    max_len_1 = max([data_1.size() for data_1 in data_1s])
    bsz = len(batch)
    tensor_0 = pad_sequences(data_0s, bsz, max_len_0, pad_id=0)
    tensor_1 = pad_sequences(data_1s, bsz, max_len_1, pad_id=0)
    return tensor_0, tensor_1


def get_dataloader(dataset):
    dataloader = data.DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    return dataloader


if __name__ == '__main__':
    train_0_path = "../../datasets/Yelp/train.0"
    train_1_path = "../../datasets/Yelp/train.1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = DataPair(train_0_path, train_1_path, device=device)
    train_dataloader = get_dataloader(train_dataset)
    for batch in train_dataloader:
        print(batch)
