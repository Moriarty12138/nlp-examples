#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/21 16:13
# @Author  : jiaguoqing 
# @Email   : jiaguoqing12138@gmail.com
# @File    : dataset.py


import json
from typing import List
import numpy as np
import torch
from wuyi import BasicTokenizer
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from utils import get_args


tokenizer = BasicTokenizer()
event_type_map = {
    "Experiment": 0,
    "Manoeuvre": 1,
    "Deploy": 2,
    "Indemnity": 3,
    "Support": 4,
    "Accident": 5,
    "Exhibit": 6,
    "Non-event": 7,
}
template = {
    "Experiment": "请找出句子中的试验",
    "Manoeuvre": "请找出句子中的演习",
    "Deploy": "请找出句子中的部署",
    "Indemnity": "请找出句子中的保障",
    "Support": "请找出句子中的支援",
    "Accident": "请找出句子中的意外事故",
    "Exhibit": "请找出句子中的展示",
    "Non-event": "句子中是否不包含事件",
}


def load_pred(path):
    texts = []
    starts = []
    ends = []
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader.readlines():
            text, start, end = line.split('\t')
            texts.append(text)
            starts.append(start)
            ends.append(end)
    return texts, starts, ends


def load_dataset(path):
    with open(path, 'r', encoding='utf-8') as reader:
        js = json.load(reader)
    examples = []
    for j in js:
        text = None
        start = 0
        end = 0
        event_type = "Non-event"
        if 'event_mention' in j and 'trigger' in j['event_mention']:
            if 'text' in j['event_mention']['trigger']:
                text = j['event_mention']['trigger']['text']
            if 'offset' in j['event_mention']['trigger']:
                # start = j['event_mention']['trigger']['offset'][0]
                # end = j['event_mention']['trigger']['offset'][1]
                start, end = j['event_mention']['trigger']['offset']
        if 'event_mention' in j and 'event_type' in j['event_mention']:
            assert j['event_mention']['event_type'] in event_type_map, "事件类型不匹配"
            # event_type = event_type_map[j['event_mention']['event_type']]
            event_type = j['event_mention']['event_type']

        t = trigger(
            text,
            start,
            end,
            event_type,
        )
        e = Example(int(j['id']), j['sentence'], j['tokens'], j['tokens_count'], t)
        examples.append(e)
    return examples


class trigger:
    def __init__(self, text: str, offset_start: int, offset_end: int, event_type: int):
        self.text = text
        self.start = offset_start
        self.end = offset_end
        self.event_type = event_type


class Example:
    def __init__(self, example_idx: int, sentence: str, tokens: List[str], tokens_count: int, t: trigger = None):
        self.idx = example_idx
        self.sentence = sentence
        self.tokens = tokens

        # 构建映射
        self.old_tokens = tokens
        self.map = {}

        self.tokens_count = tokens_count
        self.trigger = t
        # print("原数据(id={}),触发词为 '{}', 数据给出位置截出的词为 '{}'. 起始位置为 {}, 结束位置为 {} ".format(
        #             self.idx, self.trigger.text,
        #             ''.join(self.tokens[self.trigger.start:self.trigger.end]),
        #             self.trigger.start, self.trigger.end
        #         ))

        # 构建测试集时去掉
        if self.trigger.text:
            assert self.trigger.text == "".join(self.tokens[self.trigger.start:self.trigger.end]), \
                "原数据(id={})存在错误,触发词为 '{}', 数据给出位置截出的词为 '{}'. 起始位置为 {}, 结束位置为 {} ".format(
                    self.idx, self.trigger.text,
                    ''.join(self.tokens[self.trigger.start:self.trigger.end]),
                    self.trigger.start, self.trigger.end
                )

        self.re_token()

    def re_token(self):
        tokens_ = []
        start_, end_ = self.trigger.start, self.trigger.end
        for idx, token in enumerate(self.tokens):
            if self.trigger.start and idx == self.trigger.start:
                start_ = len(tokens_)

            t = tokenizer.tokenize(token)
            x = len(tokens_)
            tokens_.extend(t)
            y = len(tokens_)

            # 映射关系
            self.map[(x, y)] = (idx, token)

            if self.trigger.end and idx + 1 == self.trigger.end:
                end_ = len(tokens_)

        text_ = "".join(tokens_[start_:end_])
        # print("原文为 '{}', 转换完后为 '{}'. ".format(self.trigger.text, text_))

        # 构建测试集时去掉
        if self.trigger.start and self.trigger.end:
            assert self.trigger.text == text_, \
                "不匹配, 原文为 '{}', 转换完后为 '{}'. 起始位置为 {}, 结束位置为 {} ".format(
                    self.trigger.text, text_, start_, end_)

        self.trigger.start = start_
        self.trigger.end = end_
        self.tokens = tokens_
        self.tokens_count = len(tokens_)

    def get_trigger(self, start, end):
        s, e = 0, 0
        for (x, y), (i, token) in self.map.items():
            if x <= start < y:
                s = i
            if x <= end <= y:
                e = i
        ts = self.old_tokens[s:e]
        return "".join(ts), s, e


class Feature:
    def __init__(
            self, input_ids, attention_mask, token_type_ids,
            p_mask, start_position=None, end_position=None,
            start_of_context=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.p_mask = p_mask
        # self.tokens = tokens
        self.start_position = start_position
        self.end_position = end_position
        self.start_of_context = start_of_context
        # self.text = text


def cover_example_to_feature(
        example: Example, tokenizer: BertTokenizer, max_seq_len):
    query_tokens = tokenizer.tokenize(template[example.trigger.event_type])
    # text = example.sentence
    feature = tokenizer(
        text=" ".join(query_tokens), text_pair=" ".join(example.tokens), max_length=max_seq_len, padding='max_length',
        truncation="only_second", return_token_type_ids=True, add_special_tokens=True)

    cls_index = feature['input_ids'].index(tokenizer.cls_token_id)
    start_of_context = len(query_tokens) + 2

    p_mask = np.ones_like(feature['input_ids'])
    p_mask[start_of_context:] = 0
    pad_token_indices = np.where(np.array(feature["input_ids"]) == tokenizer.pad_token_id)
    special_token_indices = np.where(np.array(tokenizer.get_special_tokens_mask(
        feature["input_ids"], already_has_special_tokens=True)))
    p_mask[pad_token_indices] = 1
    p_mask[special_token_indices] = 1
    p_mask[cls_index] = 0

    # start_position = np.zeros_like(feature["input_ids"])
    # end_position = np.zeros_like(feature["input_ids"])
    start, end = example.trigger.start, example.trigger.end
    if start == 0 and end == 0:
        start = cls_index
        end = cls_index
    else:
        start = start_of_context+example.trigger.start
        end = start_of_context+example.trigger.end
    # start_position[start] = 1
    # end_position[end] = 1

    return Feature(
        input_ids=feature['input_ids'],
        attention_mask=feature['attention_mask'],
        token_type_ids=feature['token_type_ids'],
        # cls_index=cls_index,
        p_mask=p_mask.tolist(),
        # tokens=example.tokens,
        # start_position=start_position.tolist(),
        # end_position=end_position.tolist(),
        start_position=start,
        end_position=end,
        # start_id=example.trigger.start,
        # end_id=example.trigger.end,
        start_of_context=start_of_context,
        # text=example.trigger.text,
    )


class MRCDataset(Dataset):
    def __init__(self, dataset_path, args):
        examples = load_dataset(path=dataset_path)
        self.tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
        self.features = [cover_example_to_feature(e, self.tokenizer, args.max_len, ) for e in examples]

    def __getitem__(self, item):
        # return self.features[item].token_type_ids, self.features[item].attention_mask, \
        #        self.features[item].token_type_ids, \
        #        self.features[item].p_mask, self.features[item].start_position, \
        #        self.features[item].end_position, self.features[item].start_id, \
        #        self.features[item].end_id, self.features[item].start_of_context
        return self.features[item]

    def __len__(self):
        return len(self.features)


def collate_fn(batch):
    input_ids_batch, token_type_ids_batch, attention_mask_batch = [], [], []
    start_position_batch, end_position_batch = [], []
    p_mask_batch, start_of_context_batch = [], []

    for feature in batch:
        input_ids_batch.append(feature.input_ids)
        attention_mask_batch.append(feature.attention_mask)
        token_type_ids_batch.append(feature.token_type_ids)
        start_position_batch.append(feature.start_position)
        end_position_batch.append(feature.end_position)
        p_mask_batch.append(feature.p_mask)
        # start_id_batch.append(feature.start_id)
        # end_id_batch.append(feature.end_id)
        start_of_context_batch.append(feature.start_of_context)

    # to tensor
    input_ids_batch = torch.tensor(input_ids_batch, dtype=torch.long)
    attention_mask_batch = torch.tensor(attention_mask_batch, dtype=torch.long)
    token_type_ids_batch = torch.tensor(token_type_ids_batch, dtype=torch.long)
    start_position_batch = torch.tensor(start_position_batch, dtype=torch.long)
    end_position_batch = torch.tensor(end_position_batch, dtype=torch.long)
    p_mask_batch = torch.tensor(p_mask_batch, dtype=torch.long)

    return {
        "input_ids": input_ids_batch,
        "token_type_ids": token_type_ids_batch,
        "attention_mask": attention_mask_batch,
        "start_position": start_position_batch,
        "end_position": end_position_batch,
        "p_mask": p_mask_batch,
        # "start_id": start_id_batch,
        # "end_id": end_id_batch,
        "start_of_context": start_of_context_batch,
    }


if __name__ == '__main__':
    # # examples = load_dataset('../../datasets/Event_Competition/train_7000.json')
    # # examples = load_dataset('../../datasets/Event_Competition/valid_1500.json')
    # examples = load_dataset('../../datasets/Event_Competition/testA.json')
    # print("数据总量为: {}".format(len(examples)))
    # # print("验证集数据总量: {}".format(len(val_examples)))
    # # for example in examples:
    # #     print(example.id)
    # #     print(example.sentences)
    # df = pd.DataFrame()
    # ids, sentences, tokens, token_counts, texts, starts, ends, event_types = [], [], [], [], [], [], [], []
    # for example in examples:
    #     ids.append(example.idx)
    #     sentences.append(example.sentence)
    #     tokens.append(example.tokens)
    #     token_counts.append(example.tokens_count)
    #     texts.append(example.trigger.text)
    #     starts.append(example.trigger.start)
    #     ends.append(example.trigger.end)
    #     event_types.append(example.trigger.event_type)
    # # df["id"] = ids
    # df["sentence"] = sentences
    # df["token"] = tokens
    # df["token_count"] = token_counts
    # df["text"] = texts
    # df["start"] = starts
    # df["end"] = ends
    # df["event_type"] = event_types
    #
    # # df.to_csv("../../datasets/Event_Competition/train.csv", index=False)
    # # df.to_csv("../../datasets/Event_Competition/val.csv", index=False)
    # # df.to_csv("../../datasets/Event_Competition/testA.csv", index=False)
    #
    # # 数据后处理
    # with open('../../datasets/Event_Competition/testA.json', 'r', encoding='utf-8') as reader:
    #     js = json.load(reader)
    #
    # texts_pred, starts_pred, ends_pred = load_pred("./pred-test.txt")
    # for i in range(len(texts_pred)):
    #     new_pred, new_start, new_end = examples[i].get_trigger(int(starts_pred[i]), int(ends_pred[i]))
    #     del js[i]['sentence'], js[i]['tokens'], js[i]['tokens_count']
    #     del js[i]['entity_mention'], js[i]['relation_mention']
    #     if len(new_pred) < 1 or starts_pred == ends_pred:
    #         js[i]['event_mention'] = {}
    #
    #     if 'trigger' not in js[i]['event_mention']:
    #         js[i]['event_mention'] = {"trigger": {"text": "", "offset": [0, 0]}, "event_type": "Manoeuvre"}
    #     js[i]['event_mention']['trigger']['text'] = new_pred
    #     js[i]['event_mention']['trigger']['offset'] = [new_start, new_end]
    #
    # f = json.dumps(js, ensure_ascii=False)
    # with open("./res.json", 'w', encoding='utf-8') as writer:
    #     writer.write(f)
    args = get_args()
    dataset = MRCDataset('../../datasets/Event_Competition/valid_1500.json', args)
    print(dataset)

    dataloader = DataLoader(dataset, batch_size=2)
    for idx, batch in enumerate(dataloader):
        print(batch)
