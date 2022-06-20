#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/9 17:24
# @Author  : jiaguoqing 
# @Email   : jiaguoqing12138@gmail.com
# @File    : test.py

import torch
import random
from torch import Tensor
from typing import List
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence


# a = torch.ones(25, 300)
# b = torch.ones(22, 300)
# c = torch.ones(15, 300)
# c = pad_sequence([a, b, c], batch_first=True, padding_value=0)
# c= pack_padded_sequence(c, torch.tensor([3]))
# # print(c)
# # print(c.size())
# print(c.batch_sizes())

while True:
    r = random.randint(0, 10-1)
    print(r)
