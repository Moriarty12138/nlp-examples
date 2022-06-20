#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/10 10:23
# @Author  : jiaguoqing 
# @Email   : jiaguoqing12138@gmail.com
# @File    : utils.py

import torch

from wuyi import BasicTokenizer


tokenizer = BasicTokenizer()


language = 'zh'
stopwords_path = "../../datasets/stopwords_hit.txt"

# 训练一个 fasttext 分类器
# yelp_train_dataset_path = "../../datasets/Yelp/fasttext.train"
# yelp_test_dataset_path = "../../datasets/Yelp/fasttext.train"
yelp_train_dataset_path = "../../datasets/Yelp_zh/fasttext.train"
yelp_test_dataset_path = "../../datasets/Yelp_zh/fasttext.test"
fasttext_classifier_save_path = "./outputs/fastest.Yelp.pk"


batch_size = 256
embedding_size = 128
version = "DGST-LSTM_4"
corpus_name = "Yelp"
model_save = './model_save/'
output_dir = './outputs/'
# train_0_path = "../../datasets/Yelp/train.0"
# train_1_path = "../../datasets/Yelp/train.1"
# test_0_path = "../../datasets/Yelp/test.0"
# test_1_path = "../../datasets/Yelp/test.1"
train_0_path = "../../datasets/Yelp_zh/train.0"
train_1_path = "../../datasets/Yelp_zh/train.1"
test_0_path = "../../datasets/Yelp_zh/test.0"
test_1_path = "../../datasets/Yelp_zh/test.1"
# ref_0_path = "../../datasets/Yelp/reference.0"
# ref_1_path = "../../datasets/Yelp/reference.1"
ref_0_path = "../../datasets/Yelp_zh/reference.0"
ref_1_path = "../../datasets/Yelp_zh/reference.1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

p1 = 0.3
p2 = 0.3
