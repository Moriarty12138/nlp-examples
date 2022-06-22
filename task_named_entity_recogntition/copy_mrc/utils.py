#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/21 16:14
# @Author  : jiaguoqing 
# @Email   : jiaguoqing12138@gmail.com
# @File    : utils.py
import collections
import logging
import argparse
import random
import torch
import numpy as np
from wuyi import BasicTokenizer

bt = BasicTokenizer()


def set_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler_stream = logging.StreamHandler()
    handler_stream.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
                                  datefmt='%Y/%m/%d %H:%M:%S')
    handler_stream.setFormatter(formatter)
    logger.addHandler(handler_stream)
    return logger


def set_seed(seed):
    # set seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="../../models/chinese_L-12_H-768_A-12")
    parser.add_argument('--train_dataset_path', type=str, default="../../datasets/Event_Competition/train_7000.json")
    parser.add_argument('--valid_dataset_path', type=str, default="../../datasets/Event_Competition/valid_1500.json")
    parser.add_argument('--test_dataset_path', type=str, default="../../datasets/Event_Competition/valid_1500.json")
    parser.add_argument('--model_save_path', type=str, default="./output")
    parser.add_argument('--max_len', type=int, default=256)
    parser.add_argument('--seed', type=int, default=1202)
    parser.add_argument('--no_cuda', type=bool, default=False)
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--valid_batch_size', type=int, default=2)
    parser.add_argument('--n_epoch', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    parser.add_argument('--do_train', type=bool, default=True)
    parser.add_argument('--do_test', type=bool, default=True)
    parser.add_argument('--do_eval', type=bool, default=True)

    args = parser.parse_args()

    args.device = torch.device('cuda') if not args.no_cuda and torch.cuda.is_available() else torch.device('cpu')

    return args
