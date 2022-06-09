#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/9 14:35
# @Author  : jiaguoqing 
# @Email   : jiaguoqing12138@gmail.com
# @File    : AutoEvaluationF.py

import fasttext


# 训练一个 fasttext 分类器
yelp_train_dataset_path = "../../datasets/Yelp/fasttest.train"
yelp_test_dataset_path = "../../datasets/Yelp/fasttest.train"
fasttext_classifier_save_path = "./outputs/fastest.Yelp.pk"

fasttext_classifier = fasttext.train_supervised(yelp_train_dataset_path, label='__label__', epoch=20, lr=1)
fasttext_classifier.save_model(fasttext_classifier_save_path)
result = fasttext_classifier.test(yelp_test_dataset_path)

print('P@!: {}'.format(result[1]))
print('R@!: {}'.format(result[2]))
print('Number of examples: {}'.format(result[0]))
