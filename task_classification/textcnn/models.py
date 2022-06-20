#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/15 11:19
# @Author  : jiaguoqing 
# @Email   : jiaguoqing12138@gmail.com
# @File    : models.py

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def load_glove(word_index, args):
    print("Load embedding from {}".format(args.embedding_path))
    EMBEDDING_FILE = args.embedding_path

    def get_coef(word, *arr):
        return word, np.asarray(arr, dtype='float32')[:300]

    embedding_index = dict(get_coef(*o.split(' ')) for o in open(EMBEDDING_FILE, 'r', encoding='utf-8'))
    all_embs = np.stack(embedding_index.values())
    emb_mean, emb_std = -0.005838499, 0.48782197
    embed_size = all_embs.shape[1]

    nb_words = min(args.max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    for word, i in word_index.items():
        if i >= args.max_features:
            continue
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_vector = embedding_index.get(word.capitalize())
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    return embedding_matrix


class TextCNN(nn.Module):
    def __init__(self, args, word_index):
        super(TextCNN, self).__init__()
        filter_sizes = [1, 2, 3, 5]
        num_filters = 36
        embedding_matrix = load_glove(word_index, args)
        self.embedding = nn.Embedding(args.max_features, args.embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float))
        self.embedding.weight.requires_grad = False
        self.convs1 = nn.ModuleList([nn.Conv2d(1, num_filters, (K, args.embed_size)) for K in filter_sizes])
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(len(filter_sizes)*num_filters, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).unsqueeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logit = self.fc1(x)
        return logit


class Trainer:
    """ 训练器"""
    def __init__(self, **kwargs):
        self.n_epochs = kwargs["epochs"]
        self.batch_size = kwargs["batch_size"]
        self.validate = kwargs["validate"]
        self.save_best_dev = kwargs["save_best_dev"]
        self.use_cuda = kwargs["use_cuda"]
        self.print_every_step = kwargs["print_every_step"]
        self.optimizer = kwargs["optimizer"]
        self.model_path = kwargs["model_path"]
        self.eval_metrics = kwargs["eval_metrics"]

        self._best_accuracy = 0.0
        self.device = "cuda:0" if self.use_cuda and torch.cuda.is_available() else "cpu"

    def train(self, network, train_data, dev_data=None):
        # transfer model to gpu if available.
        network = network.to(self.device)

        # define batch iter