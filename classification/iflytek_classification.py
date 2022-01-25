#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/1/10 10:38
# @Author  : jiaguoqing 
# @Email   : jiaguoqing12138@gmail.com
# @File    : train.py

"""
分类任务demo
"""
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from transformers import TrainingArguments, Trainer
from datasets import load_dataset


train_dataset_path = "iflytek_public/train.json"
valid_dataset_path = "iflytek_public/dev.json"
test_dataset_path = "iflytek_public/test_dev.json"
max_len = 512
model_name_or_path = r"D:\models\google\distilbert-base-uncased"
output_dir = "output/"
logging_dir = "log/"
num_train_epochs = 2
per_device_train_batch_size = 128
per_device_eval_batch_size = 128
warmup_steps = 10
weight_decay = 0.1
logging_steps = 100
save_steps = 100
save_strategy = "steps"

# config
config = BertConfig.from_pretrained(model_name_or_path, num_labels=119)

# tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
# model
model = BertForSequenceClassification(config)


def preprocess_function(example):
    """数据预处理"""
    result = tokenizer(example['sentence'], max_length=512, truncation=True)
    label = int(example['label'])
    result['label'] = label
    return result


# dataset
data_files = {'train': train_dataset_path,
              'validation': valid_dataset_path,
              'test': test_dataset_path}
dataset = load_dataset('json', data_files=data_files)
dataset = dataset.map(preprocess_function)
print(dataset)


train_args = TrainingArguments(
    output_dir=output_dir,
    do_train=True,
    do_eval=True,
    do_predict=True,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    warmup_steps=warmup_steps,
    weight_decay=weight_decay,
    logging_dir=logging_dir,
    logging_steps=logging_steps,
    save_steps=save_steps,
    save_strategy=save_strategy,
)
print(train_args)


trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    tokenizer=tokenizer,
)

train_result = trainer.train()
trainer.save_model()
metrics = train_result.metrics
metrics['train_samples'] = len(dataset['train'])

trainer.log_metrics("train", metrics=metrics)
trainer.log_metrics("train", metrics=metrics)
trainer.save_state()
