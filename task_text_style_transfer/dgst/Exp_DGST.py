#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from datasets import DataPair


train_0_path = "../../datasets/Yelp/train.0"
train_1_path = "../../datasets/Yelp/train.1"
test_0_path = "../../datasets/Yelp/test.0"
test_1_path = "../../datasets/Yelp/test.1"
ref_0_path = "../../datasets/Yelp/reference.0"
ref_1_path = "../../datasets/Yelp/reference.1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("Load dataset.")
train_data = DataPair(train_0_path, train_1_path, device=device)
test_data = DataPair(test_0_path, test_1_path, device=device)
gt_data = DataPair(ref_0_path, ref_1_path, device=device)


print(train_data, test_data, gt_data)

