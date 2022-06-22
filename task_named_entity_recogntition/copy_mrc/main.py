#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/21 20:34
# @Author  : jiaguoqing 
# @Email   : jiaguoqing12138@gmail.com
# @File    : main.py

import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch import nn
from transformers import get_linear_schedule_with_warmup

from models import BertForQuestionAnswering
from utils import get_args, set_seed, set_logger
from datasets import MRCDataset, collate_fn
from datasets import load_dataset, Example, event_type_map
from typing import List


logger = set_logger()


def evaluation(examples: List[Example], start_id_pred, end_id_pred):
    # 对七个类别求 macro-F1
    TPs, FPs, FNs = [0] * 7, [0] * 7, [0] * 7
    for i, example in enumerate(examples):
        text_, start_, end_ = example.get_trigger(start_id_pred[i], end_id_pred[i])
        pred_label = event_type_map[example.trigger.event_type]  # 默认预测标签正确
        if event_type_map[example.trigger.event_type] == pred_label and \
                example.trigger.text == text_ and \
                example.trigger.start == start_ and \
                example.trigger.end == end_ and pred_label < 7:
            TPs[event_type_map[example.trigger.event_type]] += 1
        if event_type_map[example.trigger.event_type] < 7:
            FPs[event_type_map[example.trigger.event_type]] += 1
        if pred_label < 7:
            FNs[pred_label] += 1

    # 计算 macro f1
    f1s = [0.0] * 7
    for i in range(7):
        Pi = TPs[i] / FPs[i]
        Ri = TPs[i] / FNs[i]
        f1s[i] = (2 * Pi * Ri) / (Pi + Ri)
    score = sum(f1s) / 7
    return score


def do_test(model: BertForQuestionAnswering, examples, dataloader, args):
    model.eval()
    start_id_pred = []
    end_id_pred = []
    for i, batch in enumerate(dataloader):
        input_ids_batch = batch["input_ids"].to(args.device)  # tensor
        token_type_ids_batch = batch["token_type_ids"].to(args.device)  # tensor
        attention_mask_batch = batch["attention_mask"].to(args.device)  # tensor
        p_mask_batch = batch["p_mask"].to(args.device)  # tensor
        # start_position_batch = batch["start_position"].tolist()  # tensor
        # end_position_batch = batch["end_position"].tolist()  # tensor
        # start_id_batch = batch["start_id"]  # list
        # end_id_batch = batch["end_id"]  # list
        # start_id_valid.extend(batch["start_id"])
        # end_id_valid.extend(batch["end_id"])
        start_of_context_batch = batch["start_of_context"]  # list

        start_logit, end_logit = model(
            input_ids_batch, attention_mask=attention_mask_batch,
            token_type_ids=token_type_ids_batch, p_mask=p_mask_batch)
        # start_logit = start_logit + p_mask_batch.unsqueeze(-1) * -10000.0
        # end_logit = end_logit + p_mask_batch.unsqueeze(-1) * -10000.0
        start_logit = torch.argmax(start_logit, dim=-1)
        end_logit = torch.argmax(end_logit, dim=-1)

        start_logit = start_logit.to(torch.device('cpu')).tolist()
        end_logit = end_logit.to(torch.device('cpu')).tolist()
        for i_batch in range(len(batch)):
            start_id = start_logit[i_batch].index(1) - start_of_context_batch[i_batch]
            end_id = end_logit[i_batch].index(1) - start_of_context_batch[i_batch]
            start_id_pred.append(start_id if start_id > 0 else -1)
            end_id_pred.append(end_id if end_id > 0 else -1)

    score = evaluation(examples, start_id_pred, end_id_pred)
    print("Test, score={}".format(score))

    preds, starts, ends = [], [], []
    for i, example in enumerate(examples):
        text_, start_, end_ = example.get_trigger(start_id_pred[i], end_id_pred[i])
        preds.append(text_)
        starts.append(start_)
        ends.append(end_)
    with open('pred.txt', 'w', encoding='utf-8') as writer:
        for i in range(len(preds)):
            writer.write("{}\t{}\t{}\n".format(preds[i], starts[i], ends[i]))


def do_valid(ep, model: BertForQuestionAnswering, examples, dataloader, args):
    model.eval()
    start_id_pred = []
    end_id_pred = []
    for i, batch in enumerate(dataloader):
        input_ids_batch = batch["input_ids"].to(args.device)  # tensor
        token_type_ids_batch = batch["token_type_ids"].to(args.device)  # tensor
        attention_mask_batch = batch["attention_mask"].to(args.device)  # tensor
        p_mask_batch = batch["p_mask"].to(args.device)  # tensor
        # start_position_batch = batch["start_position"].tolist()  # tensor
        # end_position_batch = batch["end_position"].tolist()  # tensor
        # start_id_batch = batch["start_id"]  # list
        # end_id_batch = batch["end_id"]  # list
        # start_id_valid.extend(batch["start_id"])
        # end_id_valid.extend(batch["end_id"])
        start_of_context_batch = batch["start_of_context"]  # list

        start_logit, end_logit = model(
            input_ids_batch, attention_mask=attention_mask_batch,
            token_type_ids=token_type_ids_batch, p_mask=p_mask_batch)
        # start_logit = start_logit + p_mask_batch.unsqueeze(-1) * -10000.0
        # end_logit = end_logit + p_mask_batch.unsqueeze(-1) * -10000.0
        start_logit = torch.argmax(start_logit, dim=-1)
        end_logit = torch.argmax(end_logit, dim=-1)

        start_logit = start_logit.to(torch.device('cpu')).tolist()
        end_logit = end_logit.to(torch.device('cpu')).tolist()
        for i_batch in range(len(batch)):
            start_id = start_logit[i_batch].index(1) - start_of_context_batch[i_batch]
            end_id = end_logit[i_batch].index(1) - start_of_context_batch[i_batch]
            start_id_pred.append(start_id if start_id > 0 else -1)
            end_id_pred.append(end_id if end_id > 0 else -1)

    score = evaluation(examples, start_id_pred, end_id_pred)
    print("ep:{}, score={}".format(ep, score))


def main():
    args = get_args()
    for k, v in vars(args).items():
        logger.info("{}, {}".format(k, v))

    set_seed(args.seed)

    # tokenizer & model
    # tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    model = BertForQuestionAnswering.from_pretrained(args.model_name_or_path)
    model = model.to(device=args.device)

    # dataset & dataloader
    if args.do_train:
        train_dataset = MRCDataset(args.train_dataset_path, args=args)
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn)
    if args.do_eval:
        examples_valid = load_dataset(args.valid_dataset_path)
        valid_dataset = MRCDataset(args.valid_dataset_path, args=args)
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=args.valid_batch_size, shuffle=False, collate_fn=collate_fn)
    if args.do_test:
        examples_test = load_dataset(args.test_dataset_path)
        test_dataset = MRCDataset(args.test_dataset_path, args=args)
        test_dataloader = DataLoader(
            test_dataset, batch_size=args.valid_batch_size, shuffle=False, collate_fn=collate_fn)

    # optim
    if args.do_train:
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

    if args.do_train:
        # loss_val = []
        print("Start Train.")
        for ep in range(1, args.n_epoch+1):
            model.train()
            epoch_loss = []
            pbar = tqdm(train_dataloader)
            for i, batch in enumerate(pbar):
                input_ids_batch = batch["input_ids"].to(args.device)  # tensor
                token_type_ids_batch = batch["token_type_ids"].to(args.device)  # tensor
                attention_mask_batch = batch["attention_mask"].to(args.device)  # tensor
                p_mask_batch = batch["p_mask"].to(args.device)  # tensor
                start_position_batch = batch["start_position"].to(args.device)  # tensor
                end_position_batch = batch["end_position"].to(args.device)  # tensor
                # start_id_batch = batch["start_id"]
                # end_id_batch = batch["end_id"]
                # start_of_context_batch = batch["start_of_context"]

                model.zero_grad()
                loss, start_logit, end_logit = model(
                    input_ids=input_ids_batch, attention_mask=attention_mask_batch, token_type_ids=token_type_ids_batch,
                    start_positions=start_position_batch, end_positions=end_position_batch, p_mask=p_mask_batch)

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if i % 10 == 0:
                    print("Epoch=%d,batch_idx=%d,loss=%.4f" % (ep, i, loss.item()))
                epoch_loss.append(loss.item())
                optimizer.step()
                scheduler.step()

            if args.do_eval:
                do_valid(ep, model=model, examples=examples_valid, dataloader=valid_dataloader, args=args)
    if args.do_test:
        do_test(model=model, examples=examples_test, dataloader=test_dataloader, args=args)


if __name__ == '__main__':
    main()
