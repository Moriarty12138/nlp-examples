#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/15 11:16
# @Author  : jiaguoqing 
# @Email   : jiaguoqing12138@gmail.com
# @File    : utils.py

import pandas as pd
import torch


def save_model(model, model_path):
    """ save model. """
    torch.save(model.state_dict(), model_path)


def load_model(model, model_path, use_cuda=False):
    """Load model."""
    map_location = 'cpu'
    if use_cuda and torch.cuda.is_available():
        map_location = 'cuda:0'
    model.load_state_dict(torch.load(model_path, map_location=map_location))
    return model


def load_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    sentences = df['question_text'].tolist()
    labels = df['target'].tolist()
    return sentences, labels
