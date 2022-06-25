#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/21 20:34
# @Author  : jiaguoqing 
# @Email   : jiaguoqing12138@gmail.com
# @File    : main.py

import time
import torch
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch import nn
from transformers import get_linear_schedule_with_warmup

from models import BertForQuestionAnswering
from utils import get_args, set_seed, set_logger, load_pred
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
        if example.trigger.text == text_:
            TPs[event_type_map[example.trigger.event_type]] += 1
        if event_type_map[example.trigger.event_type] < 7:
            FPs[event_type_map[example.trigger.event_type]] += 1
        if pred_label < 7:
            FNs[pred_label] += 1

    # 计算 macro f1
    eps = 1e-7
    f1s = [0.0] * 7
    for i in range(7):
        Pi = TPs[i] / FPs[i]
        Ri = TPs[i] / FNs[i]
        f1s[i] = (2 * Pi * Ri) / (Pi + Ri + eps)
    score = sum(f1s) / 7
    return score


def do_test(examples, dataloader, args, log_path):
    model = BertForQuestionAnswering.from_pretrained("{}/best_model/".format(args.model_save_path), num_labels=8)
    model = model.to(args.device)
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
        start_logit = start_logit + p_mask_batch * -10000.0
        end_logit = end_logit + p_mask_batch * -10000.0
        start_logit = torch.argmax(start_logit, dim=-1)
        end_logit = torch.argmax(end_logit, dim=-1)

        start_logit = start_logit.to(torch.device('cpu')).tolist()
        end_logit = end_logit.to(torch.device('cpu')).tolist()
        for i_batch in range(len(start_logit)):
            start_id = start_logit[i_batch] - start_of_context_batch[i_batch]
            end_id = end_logit[i_batch] - start_of_context_batch[i_batch]
            start_id_pred.append(start_id if start_id > 0 else -1)
            end_id_pred.append(end_id if end_id > 0 else -1)


    score = evaluation(examples, start_id_pred, end_id_pred)
    print("Test, score={}".format(score))
    with open(log_path, 'a', encoding="utf-8") as writer:
        writer.write("Test: score: {}".format(score))

    preds, starts, ends = [], [], []
    for i, example in enumerate(examples):
        text_, start_, end_ = example.get_trigger(start_id_pred[i], end_id_pred[i])
        preds.append(text_)
        starts.append(start_)
        ends.append(end_)
        # if end_ - start_ > 4:
        #     end_ = start_ + 1
    with open('./pred-test.txt', 'w', encoding='utf-8') as writer:
        for i in range(len(preds)):
            writer.write("{}\t{}\t{}\n".format(preds[i], starts[i], ends[i]))

    # 数据后处理
    with open('../../datasets/Event_Competition/test_A.json', 'r', encoding='utf-8') as reader:
        js = json.load(reader)

    texts_pred, starts_pred, ends_pred = load_pred("./pred-test.txt")
    for i in range(len(texts_pred)):
        # new_pred, new_start, new_end = examples[i].get_trigger(int(starts_pred[i]), int(ends_pred[i]))
        new_start, new_end = int(starts_pred[i]), int(ends_pred[i])
        # if new_end - new_start > 4:
        #     new_end = new_start + 1
        new_pred = "".join(js[i]['tokens'][new_start: new_end])
        del js[i]['sentence'], js[i]['tokens'], js[i]['tokens_count']
        del js[i]['entity_mention'], js[i]['relation_mention']
        if len(new_pred) < 1 or starts_pred == ends_pred:
            js[i]['event_mention'] = {}
        else:
            if 'trigger' not in js[i]['event_mention']:
                # js[i]['event_mention'] = {"trigger": {"text": "", "offset": [0, 0]}, "event_type": "Manoeuvre"}
                continue
            js[i]['event_mention']['trigger']['text'] = new_pred
            js[i]['event_mention']['trigger']['offset'] = [new_start, new_end]

    f = json.dumps(js, ensure_ascii=False)
    with open("./{}.json".format(log_path[:-4]), 'w', encoding='utf-8') as writer:
        writer.write(f)


def do_valid(ep, model: BertForQuestionAnswering, examples, dataloader, args, best_score, log_path):
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
        start_logit = start_logit + p_mask_batch * -10000.0
        end_logit = end_logit + p_mask_batch * -10000.0
        start_logit = torch.argmax(start_logit, dim=-1)
        end_logit = torch.argmax(end_logit, dim=-1)

        start_logit = start_logit.to(torch.device('cpu')).tolist()
        end_logit = end_logit.to(torch.device('cpu')).tolist()
        for i_batch in range(len(start_logit)):
            start_id = start_logit[i_batch] - start_of_context_batch[i_batch]
            end_id = end_logit[i_batch] - start_of_context_batch[i_batch]
            start_id_pred.append(start_id if start_id > 0 else -1)
            end_id_pred.append(end_id if end_id > 0 else -1)

    score = evaluation(examples, start_id_pred, end_id_pred)
    if ep % 10 == 0:
        model.save_pretrained("{}/ep_{}".format(args.model_save_path, ep))
    if score > best_score:
        best_score = score
        model.save_pretrained("{}/best_model/".format(args.model_save_path))


    # print("ep:{}, score={}".format(ep, score))
    # with open("valid.log", 'a', encoding='utf-8') as writer:
    #     writer.write("ep:{}, score={}\n".format(ep, score))
    print("ep:{}, score={}".format(ep, score))
    with open(log_path, 'a', encoding="utf-8") as writer:
        writer.write("EP {}: score: {}\n".format(ep, score))

    preds, starts, ends = [], [], []
    for i, example in enumerate(examples):
        text_, start_, end_ = example.get_trigger(start_id_pred[i], end_id_pred[i])
        preds.append(text_)
        starts.append(start_)
        ends.append(end_)
    with open("{}/ep_{}_valid.txt".format(args.model_save_path, ep), 'w', encoding='utf-8') as writer:
        for i in range(len(preds)):
            writer.write("{}\t{}\t{}\n".format(preds[i], starts[i], ends[i]))

    return best_score


def main():
    # args
    args = get_args()
    train_time = time.time()
    train_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(train_time))
    output_path = "./output/model_{}".format(train_time)
    args.model_save_path = output_path
    log_path = "train_{}.log".format(train_time)
    with open(log_path, 'w', encoding='utf-8') as writer:
        for k, v in vars(args).items():
            writer.write("{}, \t{}\n".format(k, v))

    # seed
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

    # optimizer & scheduler
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
        best_score = 0
        print("Start Train.")
        for ep in range(1, args.n_epoch+1):
            model.train()
            epoch_loss = []
            pbar = tqdm(train_dataloader)
            for i_batch, batch in enumerate(pbar):
                input_ids_batch = batch["input_ids"].to(args.device)  # tensor
                token_type_ids_batch = batch["token_type_ids"].to(args.device)  # tensor
                attention_mask_batch = batch["attention_mask"].to(args.device)  # tensor
                p_mask_batch = batch["p_mask"].to(args.device)  # tensor
                start_position_batch = batch["start_position"].to(args.device)  # tensor
                end_position_batch = batch["end_position"].to(args.device)  # tensor

                model.zero_grad()
                loss, start_logit, end_logit = model(
                    input_ids=input_ids_batch, attention_mask=attention_mask_batch, token_type_ids=token_type_ids_batch,
                    start_positions=start_position_batch, end_positions=end_position_batch, p_mask=p_mask_batch)

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if i_batch % 10 == 0:
                    print("Epoch=%d,batch_idx=%d,loss=%.4f" % (ep, i_batch, loss.item()))
                    with open(log_path, 'a', encoding="utf-8") as writer:
                        writer.write("Epoch={}, \tbatch_idx={}, \tloss={}\n".format(ep, i_batch, loss.item()))
                epoch_loss.append(loss.item())
                optimizer.step()
                scheduler.step()

            if args.do_eval:
                best_score = do_valid(
                    ep=ep, model=model, examples=examples_valid,
                    dataloader=valid_dataloader, args=args, best_score=best_score, log_path=log_path)
    if args.do_test:
        do_test(examples=examples_test, dataloader=test_dataloader, args=args, log_path=log_path)


if __name__ == '__main__':
    main()
