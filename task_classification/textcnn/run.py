#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import random
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_

import args
from tokenizers import TextCNNTokenizer
from datasets import TextCNNDataset, collate_fn
from models import TextCNN


def seed_everything(seed=10):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True


def main():
    # tokenizer
    tokenizer = TextCNNTokenizer()
    df = pd.read_csv(args.train_dataset_path)
    train_dataset = TextCNNDataset(df[:int(len(df) * 0.8)],
                                   tokenizer, max_len=args.max_len)
    test_dataset = TextCNNDataset(df[int(len(df) * 0.8):int(len(df) * 0.9)],
                                  tokenizer, max_len=args.max_len)
    valid_dataset = TextCNNDataset(df[int(len(df) * 0.9):],
                                   tokenizer, max_len=args.max_len)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=8, collate_fn=collate_fn)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=8, collate_fn=collate_fn)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=8, collate_fn=collate_fn)

    # load model
    model = TextCNN(args, tokenizer.word_index)
    model.to(device=args.device)

    # loss & optimizer
    loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    seed_everything(args.seed)

    # train
    avg_losses_f = []
    avg_val_loss_f = []

    for ep in range(1, args.epochs + 1):
        model.train()

        avg_loss = 0.0
        for i, (sentences, labels) in enumerate(train_dataloader):
            sentences = sentences.to(args.device)
            labels = labels.to(args.device)

            pred = model(sentences)
            loss = loss_fn(pred, labels)

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            avg_loss += loss.item() / len(train_dataloader)

            print("LOSS: {}".format(loss.item()))

        # model.eval()
        # avg_val_loss = 0.0
        # for i, (sentences, labels) in enumerate(valid_dataloader):
        #     pred = model(sentences)
        #     loss = loss_fn(pred, labels).detach()
        #
        #     avg_val_loss += loss / len(valid_dataloader)


if __name__ == '__main__':
    main()
