import os
import numpy as np
from rouge import Rouge
from datasets import Dataset
from transformers import BertTokenizer, BartForConditionalGeneration, Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq, TrainerCallback, TrainingArguments, TrainerState, TrainerControl


model_name_or_path = "/data/gqjia/models/fnlp/bart-base-chinese"
train_src_path = "dataset/train.src"
train_tgt_path = "dataset/train.tgt"
valid_src_path = "dataset/valid.src"
valid_tgt_path = "dataset/valid.tgt"
test_src_path = "dataset/test.src"
test_tgt_path = "dataset/test.tgt"
max_src_length = 512
max_tgt_length = 100
label_pad_token_id = -100

output_dir = "outputs/"
num_train_epochs = 100
per_device_train_batch_size = 64
per_device_eval_batch_size = 64
warmup_steps = 100
weight_decay = 0.01
logging_dir = "logs/"
logging_steps = 100
save_steps = 100
save_strategy = "steps"


def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as reader:
        for line in reader:
            data.append(line.strip())
    return data

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    # print(decoded_preds, decoded_labels)
    scores = rouge.get_scores(decoded_preds, decoded_labels, avg=True)
    for key in scores:
        scores[key] = scores[key]['f'] * 100

    result = scores

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

def preprocess_function(example, tokenizer:BertTokenizer):
    """数据预处理"""
    inputs = example['article']
    target = example['summarization']
    model_inputs = tokenizer(inputs, max_length=max_src_length, padding=False, truncation=True)
    labels = tokenizer(target, max_length=max_tgt_length, padding=False, truncation=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    while '' in preds:
        idx = preds.index('')
        preds[idx] = '。'
    return preds, labels


# dataset
train_src = load_data(train_src_path)
train_tgt = load_data(train_tgt_path)
valid_src = load_data(valid_src_path)
valid_tgt = load_data(valid_tgt_path)
test_src = load_data(test_src_path)
test_tgt = load_data(test_tgt_path)
assert len(train_src) == len(train_tgt), "训练集数据不匹配。"
assert len(valid_src) == len(valid_tgt), "验证集数据不匹配。"
assert len(test_src) == len(test_tgt), "测试集数据不匹配。"

# model
tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
model = BartForConditionalGeneration.from_pretrained(model_name_or_path)
print(model)


train_dict = {"article": train_src, "summarization": train_tgt}
train_datasets = Dataset.from_dict(train_dict)
train_datasets = train_datasets.map(
    lambda x: preprocess_function(x, tokenizer=tokenizer),
    batched=True,
    num_proc=1,
    remove_columns=train_datasets.column_names,
    load_from_cache_file=False
)
valid_dict = {"article": valid_src, "summarization": valid_tgt}
valid_datasets = Dataset.from_dict(valid_dict)
valid_datasets = valid_datasets.map(
    lambda x: preprocess_function(x, tokenizer=tokenizer),
    batched=True,
    num_proc=1,
    remove_columns=valid_datasets.column_names,
    load_from_cache_file=False
)
test_dict = {"article": test_src, "summarization": test_tgt}
test_datasets = Dataset.from_dict(test_dict)
test_datasets = test_datasets.map(
    lambda x: preprocess_function(x, tokenizer=tokenizer),
    batched=True,
    num_proc=1,
    remove_columns=test_datasets.column_names,
    load_from_cache_file=False
)


# trainer
class TestCallback(TrainerCallback):
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        predictions, labels, metrics = trainer.predict(test_dataset=test_datasets, metric_key_prefix="predict")
        metrics['epoch'] = state.epoch
        state.log_history.append(metrics)


data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)

train_args = TrainingArguments(
    output_dir=output_dir,
    do_train = True,
    do_eval = True,
    do_predict = True,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    warmup_steps=warmup_steps,
    weight_decay=weight_decay,
    logging_dir=logging_dir,
    logging_steps=logging_steps,
    save_steps=save_steps,
    save_strategy=save_strategy,
    predict_with_generate = True,
)
print(train_args)
rouge = Rouge()

trainer = Seq2SeqTrainer(
    model=model,
    args=train_args,
    train_dataset=train_datasets,
    eval_dataset=valid_datasets,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[TestCallback],
)


#############
# do train
#############
train_result = trainer.train()
trainer.save_model()
metrics = train_result.metrics
metrics['train_samples'] = len(train_datasets)

trainer.log_metrics("train", metrics=metrics)
trainer.log_metrics("train", metrics=metrics)
trainer.save_state()


##########
# do test
##########
prediction, labels, metrics = trainer.predict(test_dataset=test_datasets, metric_key_prefix="predict")
test_preds = tokenizer.batch_decode(prediction, skip_special_tokens=True)
test_preds = [pred.strip() for pred in test_preds]
output_test_pred_file = os.path.join(output_dir, "test_pred.txt")
with open(output_test_pred_file, 'w', encoding='utf-8') as writer:
    writer.write("\n".join(test_preds))
