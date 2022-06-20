#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TextCNN
text_cnn_opt = {
    'seed': 1202,
    'lr': 1e-3,
    'dataset_path': "../../datasets/em",
    'train_src_path': "../../datasets/em/train_sp.0",
    'train_tgt_path': "../../datasets/em/train_sp.1",
    'valid_src_path': "../../datasets/em/valid_sp.0",
    'valid_tgt_path': "../../datasets/em/valid_sp.1",
    'test_src_path': "../../datasets/em/test_sp.0",
    'test_tgt_path': "../../datasets/em/test_sp.1",
    'embed_dim': 300,
    'dropout': 0.5,
    'max_len': 50,
    'log_step': 100,
    'eval_step': 1000,
    'batch_size': 32,
    'epoch': 50,
}

no_reward = False
only_sc_reward = False
only_bl_reward = False
order_of_training = 0
style = 0




