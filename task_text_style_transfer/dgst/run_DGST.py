#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import re
# import jieba

from datetime import datetime
from tqdm import tqdm
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from datasets import DataPair, get_dataloader
from utils import *
from models import TranLSTM
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

from train_classifier import fasttext_classifier


print("using device is {}.".format(device))


# data
print("Load dataset.")
train_data = DataPair(train_0_path, train_1_path, device=device)
test_data = DataPair(test_0_path, test_1_path, device=device)
gt_data = DataPair(ref_0_path, ref_1_path, device=device)

train_dataloader = get_dataloader(train_data, bsz=batch_size)
test_dataloader = get_dataloader(test_data, bsz=batch_size)
gt_dataloader = get_dataloader(gt_data, bsz=batch_size)

vocab_size = train_data.vocab_size


# model
print("Load model.")
model_name = "{}-{}-h{}".format(version, corpus_name, embedding_size)

T_0to1 = TranLSTM(vocab_size, embedding_size)
T_1to0 = TranLSTM(vocab_size, embedding_size)
# T_0to1 = T_0to1.to(device=device)
# T_1to0 = T_1to0.to(device=device)

# optimizer
optimizer = Adam(list(T_0to1.parameters()) + list(T_1to0.parameters()), lr=1e-3, weight_decay=1e-5)


# starting training.
def clean_text(txt):
    txt = re.sub(r"<pad>|<sos>|<eos>","",txt)
    txt = re.sub(r"\s+"," ",txt)
    return txt.strip()


def random_replace(data, p=0.4):
    shape = data.shape
    for _ in range(int(shape[0] * shape[1] * p)):
        data[random.randint(0, shape[0]-1), random.randint(0, shape[1]-1)] = random.randint(3, vocab_size-1)
    return data


def conbine_fix(data, p=0.4):
    data = data.clone()
    data = random_replace(data, p)
    return data


def evaluate(source_texts, transfer_texts, other_source_texts, origin_sent):
    if language == 'zh':
        source_texts_ = [tokenizer.tokenize(text=st) for st in source_texts]
        transfer_texts_ = [tokenizer.tokenize(text=tt) for tt in transfer_texts]
        other_source_texts_ = [tokenizer.tokenize(text=ost) for ost in other_source_texts]
    else:
        source_texts_ = [st.split(' ') for st in source_texts]
        transfer_texts_ = [tt.split(' ') for tt in transfer_texts]
        other_source_texts_ = [ost.split(' ') for ost in other_source_texts]

    # corpus bleu
    cbleu4 = corpus_bleu([[i] for i in source_texts_], transfer_texts_) * 100
    n_sents = len(source_texts_)
    all_bleu_score3 = 0.0
    all_bleu_score4 = 0.0
    for sou, tran in zip(source_texts_, transfer_texts_):
        all_bleu_score4 += sentence_bleu([sou], tran)
    bleu4 = (all_bleu_score4 / n_sents) ** 100.0

    # transfer
    labels = fasttext_classifier.predict(transfer_texts)
    truth = str(1 - origin_sent)
    transfer = (float(sum([truth in l for l1 in labels[0] for l in l1])) / n_sents) * 100.0

    # ground_true_bleu
    n_sents = len(other_source_texts_)
    all_bleu_scores = 0.0
    for i in range(len(other_source_texts_)):
        sou = other_source_texts_[i]
        tran = transfer_texts_[i]
        all_bleu_scores += sentence_bleu([sou], tran)
    gBleu = all_bleu_scores / n_sents * 100.0

    print('GroungTrueBleu:{:4.4f} | SelfBleu4: {:4.4f} | SelfBleu4(c): {:4.4f} | Transfer Acc: {:4.4f}'.format(gBleu, bleu4, cbleu4, transfer))
    return gBleu, bleu4, cbleu4, transfer


def do_test(ep):
    input_1 = []
    input_0 = []
    outputs_1to0 = []
    outputs_0to1 = []
    ref_1 = []
    ref_0 = []
    T_1to0.eval()
    T_0to1.eval()
    with torch.no_grad():
        for data_0, data_1 in test_dataloader:
            bsz = data_0.size(0)
            data_0 = data_0.to(device)
            data_1 = data_1.to(device)

            prod_0to1 = T_0to1.argmax(T_0to1(data_0)).tolist()
            prod_1to0 = T_1to0.argmax(T_1to0(data_1)).tolist()

            data_0 = data_0.tolist()
            data_1 = data_1.tolist()

            for i in range(bsz):
                input_0.append(clean_text(train_data.to_text(data_0[i])))
                input_1.append(clean_text(train_data.to_text(data_1[i])))
                outputs_0to1.append(clean_text(train_data.to_text(prod_0to1[i])))
                outputs_1to0.append(clean_text(train_data.to_text(prod_1to0[i])))
        for ref0, ref1 in gt_dataloader:
            ref0 = ref0.tolist()
            ref1 = ref1.tolist()
            for i in range(len(ref0)):
                ref_0.append(clean_text(train_data.to_text(ref0[i])))
                ref_1.append(clean_text(train_data.to_text(ref1[i])))

    gb_1, bleu3_1to0_1, bleu4_1to0_1, acc_1to0_1 = evaluate(input_1, outputs_1to0, ref_1, 1)
    gb_2, bleu3_0to1_2, bleu4_0to1_2, acc_0to1_2 = evaluate(input_0, outputs_0to1, ref_0, 0)

    with open("{}-{}-{}-stest.txt".format(output_dir, model_name, ep), 'w', encoding='utf-8') as writer:
        texts = []
        for in_0, in_1, out_0to1, out_1to0 in zip(input_0, input_1, outputs_0to1, outputs_1to0):
            texts.append(" in - neg: {}".format(in_0))
            texts.append("out - pos: {}".format(out_0to1))
            texts.append("---------")
            texts.append(" in - pos: {}".format(in_1))
            texts.append("out - neg: {}".format(out_1to0))
            texts.append("_________")
        writer.write("=== ep: {} === \n {} \n".format(ep, "\n".join(texts)))

    with open("{}-{}-numbers.txt".format(output_dir, model_name), 'a', encoding='utf-8') as writer:
        if ep == 1:
            writer.write("\n\n ========== New Start: {} ==========\n".format(model_name))
        if ep == 0:
            writer.write("-----------------------------------------\n")
        if ep > 0:
            nums = [ep, (gb_1+gb_2)/2, (bleu4_1to0_1+bleu4_0to1_2)/2, (acc_1to0_1+acc_0to1_2)/2, p1, p2, (bleu4_1to0_1+bleu4_0to1_2+acc_1to0_1+acc_0to1_2)/2]
            nums = ["{:.4f}".format(i) for i in nums]
            writer.write("\t".join(nums)+"\n")
    print("---------------------------------------------------------------")


def do_train(ep):
    start = datetime.now()
    T_0to1.train()
    T_1to0.train()
    L_total_t, L_total_c, total = 0.0, 0.0, 0.0
    Dr, Df = [], []
    data_provider = tqdm(
        train_dataloader, total=len(train_dataloader), bar_format="{desc}{percentage:3.0f}%|{bar:30}{r_bar}")
    
    for i, (data_0, data_1) in enumerate(data_provider):
        bsz = data_0.shape[0]
        data_0 = data_0.to(device)
        data_1 = data_1.to(device)
        
        optimizer.zero_grad()
        # 一次循环
        prod_0to1 = T_0to1(data_0)
        prod_0to1 = conbine_fix(T_0to1.argmax(prod_0to1), p2)
        prod_0to1to0 = T_1to0(prod_0to1)
        
        prod_1to0 = T_1to0(data_1)
        prod_1to0 = conbine_fix(T_1to0.argmax(prod_1to0), p2)
        prod_1to0to1 = T_0to1(prod_1to0)

        L_t = T_1to0.loss(data_0, prod_0to1to0) + T_0to1.loss(data_1, prod_1to0to1)

        # 加噪声后进行还原
        data_0_ = conbine_fix(data_0, p1)
        data_1_ = conbine_fix(data_1, p1)
        prod_0to0 = T_1to0(data_0_)
        prod_1to1 = T_0to1(data_1_)
        L_c = T_1to0.loss(data_0, prod_0to0) + T_0to1.loss(data_1, prod_1to1)

        # 总的损失函数
        L = L_t + L_c
        L.backward()
        clip_grad_norm_(list(T_0to1.parameters()) + list(T_1to0.parameters()), 1)
        optimizer.step()

        L_total_t += L_t.item()
        L_total_c += L_c.item()

        total += bsz
        data_provider.set_description(
            "=== ep: {} === Loss(t): {:.2f}, Loss(c): {:.2f}".format(ep, L_total_t/total, L_total_c/total))


if __name__ == '__main__':
    T_0to1 = T_0to1.to(device=device)
    T_1to0 = T_1to0.to(device=device)
    do_test(0)
    for ep in range(1, 100):
        do_train(ep)
        torch.save({
            "T_to1": T_0to1.state_dict(),
            "T_to0": T_1to0.state_dict(),
        }, '{}/{}.mod.tch'.format(model_save, model_name))

        do_test(ep)

    print("refinement training")
    p1 = 0.0
    p2 = 0.0
    for ep in range(100, 150):
        do_train(ep)
        torch.save({
            "T_to1": T_0to1.state_dict(),
            "T_to0": T_1to0.state_dict(),
        }, '{}/{}.mod.tch'.format(model_save, model_name))

        do_test(ep)
