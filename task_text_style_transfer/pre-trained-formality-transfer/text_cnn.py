#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from opts import text_cnn_opt as opt



if __name__ == '__main__':
    torch.manual_seed(opt.seed)
