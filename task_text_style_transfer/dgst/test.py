#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/9 17:24
# @Author  : jiaguoqing 
# @Email   : jiaguoqing12138@gmail.com
# @File    : test.py

import torch


max_len = 12
bsz = 10

a = torch.tensor([[0 for _ in range(max_len)] for _ in range(bsz)], dtype=torch.long)

print(a.size())
