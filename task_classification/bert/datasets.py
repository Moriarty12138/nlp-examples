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

def load_json(path):
    with open(path, 'r', encoding='utf-8') as reader:
        js = json.load(reader)
    return js


class BertDataset(data.Dataset):
    def __init__(self, dataset_path, tokenizer: BertTokenizer, mode='Train'):
        js = load_json(dataset_path)
        self.examples = []
        self.labels = []
        for j in js:
            # print(d)
            l = event_type_map[j["event_mention"]["event_type"]] if 'trigger' in j["event_mention"] else 7

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

