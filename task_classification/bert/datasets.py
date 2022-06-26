import json
import torch
from torch.utils import data
from transformers import BertTokenizer


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
id_event_map = {
    0: "Experiment",
    1: "Manoeuvre",
    2: "Deploy",
    3: "Indemnity",
    4: "Support",
    5: "Accident",
    6: "Exhibit",
    7: "Non-event",
}
data_aug = {
    "Experiment": 1,
    "Manoeuvre": 0,
    "Deploy": 2,
    "Indemnity": 4,
    "Support": 4,
    "Accident": 2,
    "Exhibit": 1,
    "Non-event": 0,
}
accident_triggers = ['坠毁', '事故', '相撞', '爆炸', '火灾', '碰撞', '起火', '失踪', '出现意外', '决口']
deploy_triggers = ['部署', '驻扎', '进驻', '运输', '抵达']
exhibit_triggers = ['展示', '亮相', '参展', '公开', '曝光', '展出', '参观', '仪式', '航展',
                    '展览', '开放', '演示', '表演', '珠海航展', '公开亮相', '曝光｡', '显示',
                    '报道', '举行', '推出', '公布', '现身', '曝出', '发布', '出现', '展会',
                    '推销', '展销', '首次亮相', '宣传', '开放参观', '游行', '爆料', '公次开',
                    '对外开放', '露面', '亮相｡', '精彩表演', '博览会', '开幕', '首飞试飞',
                    '举办', '登场', ]
experiment_triggers = ['测试', '试射', '海试', '首飞', '试验', '试飞', '试航', '下水', '实验',
                       '首次飞行', '核试验', '发射', '验证', '试用']
indemnity_triggers = ['补给', '运送', '交付', '空中加油', '赠送']
manoeuvre_triggers = ['训练', '演习', '阅兵', '演练', '实兵演习', '实战演习', '特训',
                      '飞行训练', '联合演习', '军事演习', '军事训练', '军演', '比赛',
                      '军事训练', '军训', '比拼', '对抗', '对决', '考核', '联演', '比武考核']
support_triggers = ['护航', '提供支援', '支援', '营救', '搜救', '提供支援｡',
                    '救援', '救灾', '搜寻', '协助', '野营训练', '强化训练']
nonevent_triggers = ['15日', '动用', '车出', '展现', '巡航', '行军', '航行', '表明']


def load_json(path):
    with open(path, 'r', encoding='utf-8') as reader:
        js = json.load(reader)
    return js


def save_dict(dic, path):
    dic = {k: v for k, v in sorted(dic.items(), key=lambda item: item[1], reverse=True)}
    with open(path, 'w', encoding='utf-8') as writer:
        for k, v in dic.items():
            writer.write("{}\t{}\n".format(k, v))


def preprocessor(js):
    accident = dict()
    deploy = dict()
    exhibit = dict()
    experiment = dict()
    indemnity = dict()
    manoeuvre = dict()
    support = dict()

    js_ = []
    for j in js:
        # del j['tokens'], j['tokens_count'], j['entity_mention'], j['relation_mention']
        if 'trigger' in j['event_mention']:
            text = j['event_mention']['trigger']['text']

            # 处理包含事件的数据
            if any(arg["role"] == "Content" for arg in j['event_mention']['arguments']):
                j['event_mention']['event_type'] = "Manoeuvre"
            elif any(arg["role"] == "Militaryforce" for arg in j['event_mention']['arguments']):
                j['event_mention']['event_type'] = "Deploy"
            elif any(arg["role"] == "Materials" for arg in j['event_mention']['arguments']):
                j['event_mention']['event_type'] = "Indemnity"
            elif any(arg["role"] == "Result" for arg in j['event_mention']['arguments']):
                j['event_mention']['event_type'] = "Accident"
            else:
                if text in accident_triggers:
                    j['event_mention']['event_type'] = "Accident"
                elif text in deploy_triggers:
                    j['event_mention']['event_type'] = "Deploy"
                elif text in exhibit_triggers:
                    j['event_mention']['event_type'] = "Exhibit"
                elif text in experiment_triggers:
                    j['event_mention']['event_type'] = "Experiment"
                elif text in indemnity_triggers:
                    j['event_mention']['event_type'] = "Indemnity"
                elif text in manoeuvre_triggers:
                    j['event_mention']['event_type'] = "Manoeuvre"
                elif text in support_triggers:
                    j['event_mention']['event_type'] = "Support"
                elif text in nonevent_triggers:
                    continue

            # 触发词统计
            if j['event_mention']['event_type'] == "Accident":
                accident[text] = accident.get(text, 0) + 1
            elif j['event_mention']['event_type'] == "Deploy":
                deploy[text] = deploy.get(text, 0) + 1
            elif j['event_mention']['event_type'] == "Exhibit":
                exhibit[text] = exhibit.get(text, 0) + 1
            elif j['event_mention']['event_type'] == "Experiment":
                experiment[text] = experiment.get(text, 0) + 1
            elif j['event_mention']['event_type'] == "Indemnity":
                indemnity[text] = indemnity.get(text, 0) + 1
            elif j['event_mention']['event_type'] == "Manoeuvre":
                manoeuvre[text] = manoeuvre.get(text, 0) + 1
            elif j['event_mention']['event_type'] == "Support":
                support[text] = support.get(text, 0) + 1

            with open("event_{}.json".format(j['event_mention']['event_type']), 'a', encoding='utf-8') as writer:
                s = json.dumps(j, ensure_ascii=False)
                writer.write(s + '\n')
        else:
            with open("Non-event.json", 'a', encoding='utf-8') as writer:
                s = json.dumps(j, ensure_ascii=False)
                writer.write(s + '\n')
        js_.append(j)

    save_dict(accident, "trigger_Accident.txt")
    save_dict(deploy, "trigger_Deploy.txt")
    save_dict(exhibit, "trigger_Exhibit.txt")
    save_dict(experiment, "trigger_Experiment.txt")
    save_dict(indemnity, "trigger_Indemnity.txt")
    save_dict(manoeuvre, "trigger_Manoeuvre.txt")
    save_dict(support, "trigger_Support.txt")
    return js_


class BertDataset(data.Dataset):
    def __init__(self, dataset_path, tokenizer: BertTokenizer, mode='train'):
        js = load_json(dataset_path)
        self.examples = []
        self.labels = []
        # print(len(js))
        js = preprocessor(js)
        # print(len(js))
        for j in js:
            # print(d)
            l = event_type_map[j["event_mention"]["event_type"]] if 'trigger' in j["event_mention"] else 7
            # l = 0 if 'trigger' in j["event_mention"] else 1

            del j['tokens'], j['tokens_count'], j['id']
            if "event_mention" in j:
                del j["event_mention"]
            text = json.dumps(j, ensure_ascii=False)
            d = tokenizer(text, max_length=128, padding='max_length', return_token_type_ids=True,
                          truncation=True,)

            # # 数据填充 过采样 -> 效果不明显
            # if mode == 'train' and 'trigger' in j["event_mention"]:
            #     n = data_aug[j["event_mention"]["event_type"]]
            #     for _ in range(n):
            #         self.examples.append(d)
            #         self.labels.append(l)

            self.examples.append(d)
            self.labels.append(l)

    def __getitem__(self, item):
        return self.examples[item], self.labels[item]

    def __len__(self):
        return len(self.labels)


def collate_fn(batch):
    input_ids, token_type_ids, attention_mask, labels = [], [], [], []
    for b in batch:
        example, label = b
        input_ids.append(example['input_ids'])
        token_type_ids.append(example['token_type_ids'])
        attention_mask.append(example['attention_mask'])
        labels.append(label)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    return {
        'input_ids': input_ids,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }
