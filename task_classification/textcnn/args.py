#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/15 14:53
# @Author  : jiaguoqing 
# @Email   : jiaguoqing12138@gmail.com
# @File    : opts.py

import wuyi


tokenizer = wuyi.BasicTokenizer()

train_dataset_path = "../../datasets/Quora_Insincere_Questions_Classification/train.csv"

vocab_size = 32768
max_features = 120000
embed_size = 300
max_len = 32
batch_size = 64
seed = 1202
embedding_path = "../../models/glove.840B.300d.txt"

epochs = 10
device = 'cpu'
lr = 0.001
