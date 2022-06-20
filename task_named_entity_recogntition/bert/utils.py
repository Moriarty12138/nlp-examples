#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from set_args import get_args
from typing import List
import pandas as pd
from wuyi import BasicTokenizer


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
        e = example(int(j['id']), j['sentence'], j['tokens'], j['tokens_count'], t)
        examples.append(e)
    return examples


class trigger:
    def __init__(self, text: str, offset_start: int, offset_end: int, event_type: int):
        self.text = text
        self.start = offset_start
        self.end = offset_end
        self.event_type = event_type


class example:
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
        # if self.trigger.text:
        #     assert self.trigger.text == "".join(self.tokens[self.trigger.start:self.trigger.end]), \
        #         "原数据(id={})存在错误,触发词为 '{}', 数据给出位置截出的词为 '{}'. 起始位置为 {}, 结束位置为 {} ".format(
        #             self.idx, self.trigger.text,
        #             ''.join(self.tokens[self.trigger.start:self.trigger.end]),
        #             self.trigger.start, self.trigger.end
        #         )

        self.re_token()

    def re_token(self):
        """
        ['近', '日', ',', '南', '海', '某', '海', '域', '骄', '阳',
        '似', '火', ',', '南', '部', '战', '区', '海', '军', '某',
        '支', '队', '舰', '艇', '编', '队', '在', '此', '排', '兵',
        '布', '阵', '｡', '针', '对', '未', '来', '复', '杂', '多',
        '变', '战', '场', '环', '境', ',', '他', '们', '有', '针',
        '对', '性', '设', '置', '急', '难', '险', '情', '､', '特',
        '情', ',', '有', '效', '锤', '炼', '指', '挥', '员', '指',
        '挥', '决', '策', '和', '舰', '员', '应', '急', '处', '置',
        '能', '力', '｡']
        """
        tokens_ = []
        start_, end_ = self.trigger.start, self.trigger.end
        for i, token in enumerate(self.tokens):
            if self.trigger.start and i == self.trigger.start:
                start_ = len(tokens_)

            t = tokenizer.tokenize(token)
            x = len(tokens_)
            tokens_.extend(t)
            y = len(tokens_)

            # 映射关系
            self.map[(x, y)] = (i, token)

            if self.trigger.end and i+1 == self.trigger.end:
                end_ = len(tokens_)

        text_ = "".join(tokens_[start_:end_])
        # print("原文为 '{}', 转换完后为 '{}'. ".format(self.trigger.text, text_))

        # 构建测试集时去掉
        # if self.trigger.start and self.trigger.end:
        #     assert self.trigger.text == text_, \
        #         "不匹配, 原文为 '{}', 转换完后为 '{}'. 起始位置为 {}, 结束位置为 {} ".format(
        #             self.trigger.text, text_, start_, end_)

        self.trigger.start = start_
        self.trigger.end = end_
        self.tokens = tokens_
        self.tokens_count = len(tokens_)

    def get_trigger(self, start, end):
        s, e = 0, 0
        for (x, y), (i, token) in self.map.items():
            if  x <= start < y:
                s = i
            if x <= end <= y:
                e = i
        ts = self.old_tokens[s:e]
        return "".join(ts), s, e


if __name__ == '__main__':
    args = get_args()
    # examples = load_dataset('../../datasets/Event_Competition/train_7000.json')
    # examples = load_dataset('../../datasets/Event_Competition/valid_1500.json')
    examples = load_dataset('../../datasets/Event_Competition/testA.json')
    print("数据总量为: {}".format(len(examples)))
    # print("验证集数据总量: {}".format(len(val_examples)))
    # for example in examples:
    #     print(example.id)
    #     print(example.sentences)
    df = pd.DataFrame()
    ids, sentences, tokens, token_counts, texts, starts, ends, event_types = [], [], [], [], [], [], [], []
    for example in examples:
        ids.append(example.idx)
        sentences.append(example.sentence)
        tokens.append(example.tokens)
        token_counts.append(example.tokens_count)
        texts.append(example.trigger.text)
        starts.append(example.trigger.start)
        ends.append(example.trigger.end)
        event_types.append(example.trigger.event_type)
    # df["id"] = ids
    df["sentence"] = sentences
    df["token"] = tokens
    df["token_count"] = token_counts
    df["text"] = texts
    df["start"] = starts
    df["end"] = ends
    df["event_type"] = event_types

    # df.to_csv("../../datasets/Event_Competition/train.csv", index=False)
    # df.to_csv("../../datasets/Event_Competition/val.csv", index=False)
    df.to_csv("../../datasets/Event_Competition/testA.csv", index=False)


    # 数据后处理
    with open('../../datasets/Event_Competition/testA.json', 'r', encoding='utf-8') as reader:
        js = json.load(reader)

    texts_pred, starts_pred, ends_pred = load_pred("./pred-test.txt")
    for i in range(len(texts_pred)):
        new_pred, new_start, new_end = examples[i].get_trigger(int(starts_pred[i]), int(ends_pred[i]))
        del js[i]['sentence'], js[i]['tokens'], js[i]['tokens_count']
        del js[i]['entity_mention'], js[i]['relation_mention']
        if len(new_pred) < 1 or starts_pred == ends_pred:
            js[i]['event_mention'] = {}

        if 'trigger' not in js[i]['event_mention']:
            js[i]['event_mention'] = {"trigger": {"text": "", "offset": [0, 0]}, "event_type": "Manoeuvre"}
        js[i]['event_mention']['trigger']['text'] = new_pred
        js[i]['event_mention']['trigger']['offset'] = [new_start, new_end]

    f = json.dumps(js, ensure_ascii=False)
    with open("./res.json", 'w', encoding='utf-8') as writer:
        writer.write(f)
