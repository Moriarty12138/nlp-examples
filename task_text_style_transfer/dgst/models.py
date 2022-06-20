#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from utils import *
from torch import nn
from torch.nn import functional as F


class TranLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layers=4, bidirectional=True, batch_first=True):
        super(TranLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.n_directional = 2 if bidirectional else 1
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.encoder = nn.LSTM(
            self.hidden_size, self.hidden_size, self.n_layers, batch_first=batch_first, bidirectional=bidirectional)
        self.decoder = nn.LSTM(
            self.hidden_size, self.hidden_size, self.n_layers, batch_first=batch_first, bidirectional=bidirectional)
        self.linear = nn.Linear(self.hidden_size*2 if bidirectional else self.hidden_size, vocab_size, bias=True)

    def forward(self, x):
        bsz, seq_len = x.data.size()  # bsz, max_len
        hidden = torch.zeros(self.n_layers*2, bsz, self.hidden_size).to(device)  # h_n
        cell = torch.zeros(self.n_layers*2, bsz, self.hidden_size).to(device)  # c_n

        embedding = self.embedding(x)  # (bsz, max_len, hidden_size)
        _, (hidden, cell) = self.encoder(embedding, (hidden, cell))  # hidden_size:

        input = embedding[:, 0:1, :]  # (bsz, 1, hidden_size) 当前 batch 下 每个输入的第一个值 <sos>
        sos = torch.zeros(bsz, 1, self.vocab_size).to(device)  # (bsz, 1, vocab_size)
        sos[:, 0, x[0, 0]] = 1.  # x[0, 0] 是 <sos> 的 id，  将第三个元素置为1
        outputs = [sos]
        for t in range(1, seq_len):  # 按照顺序将其输入
            output, (hidden, cell) = self.decoder(input, (hidden, cell))  # 从第一个值开始 output: (bsz, 1, hidden_size * n_direction)
            output = self.linear(output)  # 将输出 output 转为在词典上的分布 (bsz, 1, vocab_size)
            outputs.append(output)
            input = self.embedding(output.argmax(2))  #
        outputs = torch.cat(outputs, dim=1)
        return outputs

    def loss(self, x, prod):
        vocab_size = prod.shape[-1]
        loss = F.cross_entropy(prod.reshape(-1, vocab_size), x.view(-1), reduction='none')
        loss = loss.sum()
        return loss

    def argmax(self, prod):
        return prod.argmax(2)

