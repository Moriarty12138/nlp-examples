#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer
from torch.optim import AdamW
from datasets import EventExtractionDataset, collate_fn
from models import MRCModel
from set_args import get_args


def get_eval_score(start_true, end_true, start_pred, end_pred):
    assert len(start_true) == len(end_true) == len(start_pred) == len(end_pred)
    n = len(start_true)
    TP = 0
    for i in range(n):
        if start_true[i] == start_pred[i] and start_pred[i] == end_pred[i]:
            TP += 1
    return TP/n


def do_test(model, tokenizer: BertTokenizer, args):
    test_dataset = EventExtractionDataset(args.test_dataset_path, args.model_name_or_path, args.max_len)
    test_dataloader = DataLoader(test_dataset, batch_size=args.valid_batch_size, shuffle=False, collate_fn=collate_fn)
    model = model.to(args.device)
    model.eval()
    start_true = []
    end_true = []
    start_pred = []
    end_pred = []
    entities = []
    # 保存输出结果
    ss, es = [], []
    for i, batch in enumerate(test_dataloader):
        input_ids = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        token_type_ids = batch["token_type_ids"].to(args.device)
        # 正确结果
        start_ids = batch["start_ids"].tolist()
        end_ids = batch["end_ids"].tolist()

        start_, end_ = model(input_ids, attention_mask, token_type_ids)
        start_ = start_.to(torch.device('cpu')).tolist()
        end_ = end_.to(torch.device('cpu')).tolist()

        # 第二句子的起始位置
        ttis = []
        tti = token_type_ids.to(torch.device('cpu')).tolist()
        for t in tti:
            for it, c in enumerate(t):
                if c == 1:
                    ttis.append(it)
                    continue

        start_true.extend(start_ids)
        end_true.extend(end_ids)
        start_pred.extend(start_)
        end_pred.extend(end_)

        entity_batch = []
        for i_batch in range(len(batch)):
            assert len(start_[i_batch]) == len(end_[i_batch])
            n = len(start_[i_batch])
            start_id = 0
            end_id = 0
            for idx in range(n):
                if start_[i_batch][idx] == 1:
                    start_id = idx
                if end_[i_batch][idx] == 1:
                    end_id = idx

            ss.append(start_id - ttis[i_batch])
            es.append(end_id - ttis[i_batch])
            entity_batch.append(tokenizer.decode(input_ids[i_batch][start_id:end_id]))

        entities.extend(entity_batch)
    score = get_eval_score(start_true, end_true, start_pred, end_pred)
    print("Score: {}".format(score))

    with open('./pred-test.txt', 'w', encoding='utf-8') as writer:
        for i, e in enumerate(entities):
            writer.write("{}\t{}\t{}\n".format(e, ss[i], es[i]))


def do_eval(ep, model, tokenizer: BertTokenizer, args):
    valid_dataset = EventExtractionDataset(args.valid_dataset_path, args.model_name_or_path, args.max_len)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=False, collate_fn=collate_fn)
    model = model.to(args.device)
    model.eval()
    start_true = []
    end_true = []
    start_pred = []
    end_pred = []
    entities = []
    for i, batch in enumerate(valid_dataloader):
        input_ids = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        token_type_ids = batch["token_type_ids"].to(args.device)
        # 正确结果
        start_ids = batch["start_ids"].tolist()
        end_ids = batch["end_ids"].tolist()

        start_, end_ = model(input_ids, attention_mask, token_type_ids)
        start_ = start_.to(torch.device('cpu')).tolist()
        end_ = end_.to(torch.device('cpu')).tolist()

        start_true.extend(start_ids)
        end_true.extend(end_ids)
        start_pred.extend(start_)
        end_pred.extend(end_)

        entity_batch = []
        for i_batch in range(len(batch)):
            assert len(start_[i_batch]) == len(end_[i_batch])
            n = len(start_[i_batch])
            start_id = 0
            end_id = 0
            for idx in range(n):
                if start_[i_batch][idx] == 1:
                    start_id = idx
                if end_[i_batch][idx] == 1:
                    end_id = idx
            entity_batch.append(tokenizer.decode(input_ids[i_batch][start_id:end_id]))

        entities.extend(entity_batch)
    score = get_eval_score(start_true, end_true, start_pred, end_pred)
    print("Score: {}".format(score))

    with open('./pred-{}.txt'.format(ep), 'w', encoding='utf-8') as writer:
        for e in entities:
            writer.write(e + "\n")


def main():
    args = get_args()
    print("Using device is {}".format(args.device))

    # load dataset
    train_dataset = EventExtractionDataset(args.train_dataset_path, args.model_name_or_path, args.max_len)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn)

    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    model = MRCModel.from_pretrained(args.model_name_or_path)
    model = model.to(device=args.device)

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_group_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    total_steps = len(train_dataloader) * args.n_epoch
    optimizer = AdamW(optimizer_group_parameters, lr=args.lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps), num_training_steps=total_steps)

    loss_vals = []
    print("Start Train.")
    for ep in range(1, args.n_epoch + 1):
        model.train()
        epoch_loss = []
        pbar = tqdm(train_dataloader)
        for i, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            token_type_ids = batch["token_type_ids"].to(args.device)
            start_ids = batch["start_ids"].to(args.device)
            end_ids = batch["end_ids"].to(args.device)
            model.zero_grad()
            loss = model(input_ids, attention_mask, token_type_ids, start_ids, end_ids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            if i % 10 == 0:
                print("Epoch=%d,batch_idx=%d,loss=%.4f" % (ep, i, loss.item()))
            epoch_loss.append(loss.item())
            optimizer.step()
            scheduler.step()
        loss_vals.append(np.mean(epoch_loss))

        do_eval(ep, model, tokenizer, args)

    print("End train.")
    model.save_pretrained(args.model_save_path)

    do_test(model, tokenizer, args)


if __name__ == '__main__':
    main()
