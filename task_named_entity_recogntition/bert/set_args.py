#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="/data/gqjia/projects/models/bert-base-chinese")
    parser.add_argument('--train_dataset_path', type=str, default="../../datasets/Event_Competition/train.csv")
    parser.add_argument('--valid_dataset_path', type=str, default="../../datasets/Event_Competition/val.csv")
    parser.add_argument('--test_dataset_path', type=str, default="../../datasets/Event_Competition/testA.csv")
    parser.add_argument('--model_save_path', type=str, default="./output")
    parser.add_argument('--max_len', type=int, default=256)
    parser.add_argument('--no_cuda', type=bool, default=False)
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--valid_batch_size', type=int, default=32)
    parser.add_argument('--n_epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    args = parser.parse_args()

    args.device = torch.device('cuda') if not args.no_cuda and torch.cuda.is_available() else torch.device('cpu')

    return args
