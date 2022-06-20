#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/9 14:35
# @Author  : jiaguoqing 
# @Email   : jiaguoqing12138@gmail.com
# @File    : AutoEvaluationF.py
import os
# import jieba
import fasttext

from utils import *
from datasets import load_data, save_data


if not os.path.exists(yelp_train_dataset_path) or not os.path.exists(yelp_test_dataset_path):
    train_0, train_1 = load_data(train_0_path), load_data(train_1_path)
    if language == 'zh':
        train_dataset = ["__label__0 "+" ".join(tokenizer.tokenize(text=t)) for t in train_0] + \
                        ["__label__1 "+" ".join(tokenizer.tokenize(text=t)) for t in train_1]
    else:
        train_dataset = ["__label__0 " + t for t in train_0] + \
                        ["__label__1 " + t for t in train_1]
    save_data(yelp_train_dataset_path, train_dataset)
    test_0, test_1 = load_data(test_0_path), load_data(test_1_path)
    if language == 'zh':
        test_dataset = ["__label__0 "+" ".join(tokenizer.tokenize(text=t)) for t in test_0] + \
                       ["__label__1 "+" ".join(tokenizer.tokenize(text=t)) for t in test_1]
    else:
        test_dataset = ["__label__0 " + t for t in test_0] + \
                       ["__label__1 " + t for t in test_1]
    save_data(yelp_test_dataset_path, test_dataset)


if not os.path.exists(fasttext_classifier_save_path):
    fasttext_classifier = fasttext.train_supervised(yelp_train_dataset_path, label='__label__', epoch=1000, lr=0.5)
    fasttext_classifier.save_model(fasttext_classifier_save_path)
    result = fasttext_classifier.test(yelp_test_dataset_path)

    print('P@!: {}'.format(result[1]))
    print('R@!: {}'.format(result[2]))
    print('Number of examples: {}'.format(result[0]))

else:
    fasttext_classifier = fasttext.load_model(fasttext_classifier_save_path)
