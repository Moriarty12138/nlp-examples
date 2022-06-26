#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import json
from torch import nn
from torch.utils import data
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score

from datasets import BertDataset, collate_fn, load_json, id_event_map
from utils import get_args


def evaluation(y_true, y_pred, average='macro'):
    score = f1_score(y_true, y_pred, average=average)
    acc = accuracy_score(y_true, y_pred)
    return score, acc


def get_test_data(y_pred, args):
    js = load_json(args.test_dataset_path)
    for i in range(len(js)):
        if y_pred[i] < 7:
            js[i]["event_mention"] = {
                "trigger": {
                    "text": "",
                    "offset": [
                        0,
                        0
                    ]
                },
                "event_type": id_event_map[y_pred[i]]
            }
        else:
            js[i]["event_mention"] = {}
    f = json.dumps(js, ensure_ascii=False)
    with open("./do_test.json", 'w', encoding='utf-8') as writer:
        writer.write(f)


def do_test(dataloader, args, log_path):
    model = BertForSequenceClassification.from_pretrained(
        "{}/best_model/".format(args.model_save_path),
        num_labels=8
    )
    model = model.to(args.device)
    model.eval()
    y_true, y_pred = [], []
    for i_batch, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(args.device)
        token_type_ids = batch['token_type_ids'].to(args.device)
        attention_mask = batch['attention_mask'].to(args.device)
        labels = batch['labels'].to(args.device)

        outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        predictions = outputs.logits.argmax(dim=-1)
        y_true.extend(labels.tolist())
        y_pred.extend(predictions.tolist())

    f1, acc = evaluation(y_true, y_pred)
    print("Test: F1 score: {}, Acc: {}\n".format(f1, acc))
    with open(log_path, 'a', encoding="utf-8") as writer:
        writer.write("Test: F1 score: {}, Acc: {}".format(f1, acc))
    get_test_data(y_pred, args)


def do_vaild(model, dataloader, ep, args, log_path, score):
    model.eval()
    y_true, y_pred = [], []
    for i_batch, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(args.device)
        token_type_ids = batch['token_type_ids'].to(args.device)
        attention_mask = batch['attention_mask'].to(args.device)
        labels = batch['labels'].to(args.device)

        outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        predictions = outputs.logits.argmax(dim=-1)
        y_true.extend(labels.tolist())
        y_pred.extend(predictions.tolist())

    # 获取bad case
    with open("./bad_case/bad_case_ep_{}.txt".format(ep), 'w', encoding='utf-8') as writer:
        for i in range(len(y_pred)):
            if y_true[i] != y_pred[i]:
                writer.write("id {}: y_pred={}, y_true={}\n".format(i, y_pred[i], y_true[i]))

    f1, acc = evaluation(y_true, y_pred, average='macro')
    print("EP {}: F1 score: {}, Acc: {}".format(ep, f1, acc))
    with open(log_path, 'a', encoding="utf-8") as writer:
        writer.write("EP {}: F1 score: {}, Acc: {}\n".format(ep, f1, acc))

    if ep % 10 == 0:
        model.save_pretrained("{}/ep_{}".format(args.model_save_path, ep))

    if f1 > score:
        model.save_pretrained("{}/best_model/".format(args.model_save_path))
        score = f1

    return score


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

    # model & tokenizer
    model = BertForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=8,
    )
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    model = model.to(args.device)

    # dataset & dataloader
    train_dataset = BertDataset(args.train_dataset_path, tokenizer, mode='train')
    train_dataloader = data.DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    valid_dataset = BertDataset(args.valid_dataset_path, tokenizer, mode='valid')
    valid_dataloader = data.DataLoader(valid_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    test_dataset = BertDataset(args.test_dataset_path, tokenizer, mode='test')
    test_dataloader = data.DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    # optimizer & scheduler
    total_steps = len(train_dataloader) * args.n_epochs
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_group_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [
            p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_group_parameters, lr=args.lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps), num_training_steps=total_steps)

    if args.do_train:
        score = 0
        for ep in range(args.n_epochs):
            model.train()
            epoch_loss = []
            for i_batch, batch in enumerate(train_dataloader):
                input_ids = batch['input_ids'].to(args.device)
                token_type_ids = batch['token_type_ids'].to(args.device)
                attention_mask = batch['attention_mask'].to(args.device)
                labels = batch['labels'].to(args.device)

                output = model(
                    input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)
                loss = output.loss

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if i_batch % 10 == 0:
                    print("Epoch={}, batch_idx={}, loss={}".format(ep, i_batch, loss.item()))
                    with open(log_path, 'a', encoding="utf-8") as writer:
                        writer.write("Epoch={}, \tbatch_idx={}, \tloss={}\n".format(ep, i_batch, loss.item()))
                epoch_loss.append(loss.item())
                optimizer.step()
                scheduler.step()

            if args.do_eval:
                score = do_vaild(
                    model, dataloader=valid_dataloader, ep=ep, args=args, log_path=log_path, score=score)

    if args.do_test:
        do_test(dataloader=test_dataloader, args=args, log_path=log_path)


if __name__ == '__main__':
    main()
