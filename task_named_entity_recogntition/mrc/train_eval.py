import config
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Sampler
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
import os
import pickle
from tqdm import tqdm, trange
from transformers.data.processors.squad import SquadResult
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)
import json
import numpy as np
import glob
from collections import OrderedDict
from tensorboardX import SummaryWriter


class InfiniteSampler(Sampler):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __len__(self):
        return 2 ** 30

    def __iter__(self):
        while True:
            order = np.random.permutation(self.num_samples)
            for i in range(self.num_samples):
                yield order[i]


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def predict(model, eval_dataset, eval_examples, eval_features, step, device, logger, tokenizer):
    args = config.args
    logger.info("***** Running predictions *****")
    logger.info("  Num orig examples = %d", len(eval_examples))
    logger.info("  Num split examples = %d", len(eval_features))
    logger.info("  Batch size = %d", args.predict_batch_size)

    # dataset & dataloader
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.predict_batch_size)

    model.eval()
    all_results = []
    logger.info("Start evaluating")

    if os.path.exists('all_results.tmp') and not config.args.do_train:
        all_results = pickle.load(open('all_results.tmp', 'rb'))
    else:
        for batch in tqdm(eval_dataloader, desc="Evaluating", disable=None):
            if len(all_results) % 1000 == 0:
                logger.info("Processing example: %d" % (len(all_results)))

            batch = tuple(t.to(device) for t in batch)
            input_ids = batch[0]
            input_mask = batch[1]
            token_type_ids = batch[2]
            feature_indices = batch[3]
            p_mask = batch[5]

            with torch.no_grad():
                batch_start_logits, batch_end_logits = model(input_ids, token_type_ids=token_type_ids, p_mask=p_mask,
                                                             attention_mask=input_mask)  # , ent, pos)
            for i, feature_index in enumerate(feature_indices):
                eval_feature = eval_features[feature_index.item()]
                unique_id = int(eval_feature.unique_id)

                start_logits = batch_start_logits[i].detach().cpu().tolist()
                end_logits = batch_end_logits[i].detach().cpu().tolist()

                all_results.append(SquadResult(unique_id, start_logits, end_logits))

        if not config.args.do_train:
            try:
                pickle.dump(all_results, open('all_results.tmp', 'wb'))
            except:
                logger.info("can't save all_results.tmp")

    logger.info("Write multi...")
    output_prediction_file = os.path.join(args.output_dir, "predictions_%d.json" % (step))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_%d.json" % (step))
    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_multi_%d.json" % (step))
    else:
        output_null_log_odds_file = None
    all_predictions = compute_predictions_logits(
        eval_examples,
        eval_features,
        all_results,
        args.n_best_size,
        args.max_answer_length,
        args.do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        args.verbose_logging,
        args.version_2_with_negative,
        args.null_score_diff_threshold,
        tokenizer,
    )

    if args.do_eval:
        results = squad_evaluate(eval_examples, all_predictions)
        f1 = results['f1']
        em = results['exact']
        logger.info(f"{results}")
        logger.info(f"F1:{f1:.4f}, EM:{em:.4f}")
    else:
        f1 = em = -1
    model.train()
    return f1, em


def train(model, train_dataset, device, logger, optimizer,
          n_gpu, eval_dataset=None, eval_examples=None, eval_features=None, semi_dataset=None, tokenizer=None):
    global_step = 0

    args = config.args
    train_data = train_dataset  # 迷惑操作
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.forward_batch_size, drop_last=True)
    train_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    checkpoints = [int(train_steps_per_epoch * ci / args.ckpt_frequency) for ci in range(args.ckpt_frequency)]
    logger.info(f"{train_steps_per_epoch}, {checkpoints}")

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    if semi_dataset:
        logger.info("with semi-supervised training")
        # all_input_ids, all_attention_masks, all_token_type_ids, all_example_index, all_cls_index, all_p_mask
        semi_dataloader = DataLoader(semi_dataset, sampler=InfiniteSampler(len(semi_dataset)),
                                     batch_size=args.forward_semi_batch_size)
        semi_dataiter = iter(semi_dataloader)

    def do_vat(input_ids, token_type_ids, p_mask, input_mask, start_logits, end_logits, model_embeddings):
        word_embeddings = model_embeddings(input_ids)

        with torch.no_grad():
            start_probs = F.softmax(start_logits, dim=-1)
            end_probs = F.softmax(end_logits, dim=-1)
        vat_batch_size = word_embeddings.size(0)
        d = torch.randn_like(word_embeddings)
        d_norm = 1
        d = d / d_norm * args.si
        d.requires_grad_()
        start_logits_d, end_logits_d = model(input_ids=None, token_type_ids=token_type_ids, p_mask=p_mask,
                                             attention_mask=input_mask, inputs_embeds=d + word_embeddings)

        kl_div_loss_start = F.kl_div(input=F.log_softmax(start_logits_d, dim=-1), target=start_probs, reduction='sum')
        kl_div_loss_end = F.kl_div(input=F.log_softmax(end_logits_d, dim=-1), target=end_probs, reduction='sum')
        kl_div_loss = (kl_div_loss_start + kl_div_loss_end) / 2 / vat_batch_size

        r_adv = torch.autograd.grad(kl_div_loss, d)[0]
        r_adv_norm = torch.norm(r_adv.view(vat_batch_size, -1), dim=-1, keepdim=True).unsqueeze(-1)

        embedding_norm = torch.norm(word_embeddings.view(vat_batch_size, -1), dim=-1, keepdim=True).unsqueeze(
            -1).detach()
        perturbation = r_adv / r_adv_norm * embedding_norm * args.epsilon
        word_embeddings = model_embeddings(input_ids)
        start_logits_v, end_logits_v = model(input_ids=None, token_type_ids=token_type_ids, p_mask=p_mask,
                                             attention_mask=input_mask, inputs_embeds=perturbation + word_embeddings)
        lds_start = F.kl_div(input=F.log_softmax(start_logits_v, dim=-1), target=start_probs, reduction='sum')
        lds_end = F.kl_div(input=F.log_softmax(end_logits_v, dim=-1), target=end_probs, reduction='sum')
        VAT_loss = ((lds_start + lds_end) / 2) / vat_batch_size
        VAT_loss = VAT_loss / args.gradient_accumulation_steps
        VAT_loss.backward()

    if args.model_type == 'roberta':
        model_embeddings = model.roberta.embeddings.word_embeddings
    elif args.model_type == 'bert':
        model_embeddings = model.bert.embeddings.word_embeddings

    model.train()
    TB_writer = SummaryWriter(log_dir=args.output_dir)
    TB_writer_step = 0
    for _ in trange(int(args.num_train_epochs), desc="Epoch", disable=None):
        model.zero_grad()
        logger.info(f"Length of epoch: {len(train_dataloader)}")
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", disable=None)):
            batch = tuple(t.to(device) for t in batch)
            input_ids = batch[0]
            input_mask = batch[1]
            token_type_ids = batch[2]
            start_positions = batch[3]
            end_positions = batch[4]
            p_mask = batch[6]

            word_embeddings = model_embeddings(input_ids)
            word_embeddings.retain_grad()

            loss, start_logits, end_logits = \
                model(input_ids=None, attention_mask=input_mask, token_type_ids=token_type_ids, p_mask=p_mask,
                      start_positions=start_positions, end_positions=end_positions, inputs_embeds=word_embeddings)
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if not args.disable_normal_training:
                loss = loss / args.gradient_accumulation_steps
                loss.backward()

            # Adversarial Training
            if args.enable_AT:
                if args.disable_normal_training:
                    embedding_grad = torch.autograd.grad(loss, word_embeddings)[0]
                else:
                    embedding_grad = word_embeddings.grad
                embedding_norm = torch.norm(word_embeddings.view(word_embeddings.size(0), -1), dim=-1,
                                            keepdim=True).unsqueeze(-1).detach()
                embedding_grad_norm = torch.norm(embedding_grad.view(embedding_grad.size(0), -1), dim=-1,
                                                 keepdim=True).unsqueeze(-1)
                perturbation = embedding_grad / embedding_grad_norm * embedding_norm * args.epsilon
                adversarial_example = perturbation + model_embeddings(input_ids)
                AT_loss, _, _ = model(input_ids=None, attention_mask=input_mask, token_type_ids=token_type_ids,
                                      p_mask=p_mask,
                                      start_positions=start_positions, end_positions=end_positions,
                                      inputs_embeds=adversarial_example)
                if n_gpu > 1:
                    AT_loss = AT_loss.mean()
                AT_loss = AT_loss / args.gradient_accumulation_steps
                AT_loss.backward()
            # End of Adversarial Training

            if args.enable_VAT:
                do_vat(input_ids=input_ids, token_type_ids=token_type_ids, p_mask=p_mask, input_mask=input_mask,
                       start_logits=start_logits, end_logits=end_logits,
                       model_embeddings=model_embeddings)

            if args.enable_semi_VAT and semi_dataset:
                semi_batch = next(semi_dataiter)
                semi_input_ids, semi_input_mask, semi_token_type_ids, _, _, semi_p_mask = (t.to(device) for t in
                                                                                           semi_batch)
                with torch.no_grad():
                    semi_start_logits, semi_end_logits = model(semi_input_ids, token_type_ids=semi_token_type_ids,
                                                               p_mask=semi_p_mask, attention_mask=semi_input_mask)
                do_vat(input_ids=semi_input_ids, token_type_ids=semi_token_type_ids, p_mask=semi_p_mask,
                       input_mask=semi_input_mask,
                       start_logits=semi_start_logits, end_logits=semi_end_logits,
                       model_embeddings=model_embeddings)

            TB_writer_step += 1

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()  # We have accumulated enought gradients
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.PRINT_EVERY == 0:
                    logger.info(f"Step: {step + 1}, global step: {global_step}")

                if (global_step % train_steps_per_epoch in checkpoints) and args.do_predict:
                    f1, em = predict(model, eval_dataset, eval_examples, eval_features, global_step, device, logger,
                                     tokenizer)
                    logger.info(f"saving at step {step + 1} ; global step {global_step}")
                    coreModel = model.module if 'DataParallel' in model.__class__.__name__ else model
                    str_all = 'all'
                    state_dict = coreModel.state_dict()
                    if f1 is None:
                        torch.save(state_dict, os.path.join(args.output_dir, "gs%d_%s.pkl" % (global_step, str_all)))
                    else:
                        torch.save(state_dict, os.path.join(
                            args.output_dir, "F{:.4f}_EM{:.4f}_gs{}_{}.pkl".format(f1, em, global_step, str_all)))
