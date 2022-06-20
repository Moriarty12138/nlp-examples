import logging
import collections
import os
import random
import numpy as np
import torch
import config

from utils import read_and_convert, divide_parameters, load_and_cache_examples
from functools import partial
from train_eval import predict, train
from transformers import RobertaConfig, RobertaForQuestionAnswering, RobertaTokenizer
from transformers import BertConfig, BertForQuestionAnswering, BertTokenizer
from optimization import BERTAdam

# logger
logger = logging.getLogger("Main")
logger.setLevel(logging.INFO)
handler_stream = logging.StreamHandler()
handler_stream.setLevel(logging.INFO)
formatter = logging.Formatter(
    fmt='%(asctime)s - %(levelname)s - %(name)s -  %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
handler_stream.setFormatter(formatter)
logger.addHandler(handler_stream)


def main():
    # parse arguments
    config.parse()
    args = config.args
    for k, v in vars(args).items():
        logger.info(f"{k}, {v}")

    # set seeds
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # arguments check
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    args.device = device
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    # args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.do_train:
        if not args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if args.do_predict:
        if not args.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        logger.info("Output directory () already exists and is not empty.")
    os.makedirs(args.output_dir, exist_ok=True)

    # read data
    if args.model_type == 'roberta':
        tokenizer = RobertaTokenizer(vocab_file=args.vocab_file, merges_file=args.merges_file,
                                     do_lower_case=args.do_lower_case)
    elif args.model_type == 'bert':
        tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    args.forward_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    args.forward_semi_batch_size = int(args.semi_batch_size / args.gradient_accumulation_steps)

    train_examples = None
    train_features = None
    eval_examples = None
    eval_dataset = None
    eval_features = None
    semi_dataset = None
    num_train_steps = None

    if args.do_train:
        # 读取数据 evaluate 参数判断不同的数据集
        train_dataset = load_and_cache_examples(args.train_file, args, tokenizer, evaluate=False, output_examples=False,
                                                infix='train')

        if args.semi_file:
            semi_dataset = load_and_cache_examples(args.semi_file, args, tokenizer, evaluate=True,
                                                   output_examples=False, infix='semi')

        num_train_steps = int(len(train_dataset) // args.train_batch_size) * args.num_train_epochs

    if args.do_predict:
        eval_dataset, eval_examples, eval_features = load_and_cache_examples(args.predict_file, args, tokenizer,
                                                                             evaluate=True, output_examples=True,
                                                                             infix='dev')

    # Build Model and load checkpoint
    # 加载预训练模型，如果存在 checkpoint 则从 checkpoint 加载
    if args.model_type == 'roberta':
        model_config = RobertaConfig.from_json_file(args.config_file)
        ModelForQA = RobertaForQuestionAnswering
    elif args.model_type == 'bert':
        model_config = BertConfig.from_json_file(args.config_file)
        ModelForQA = BertForQuestionAnswering
    if args.init_checkpoint is not None:
        state_dict = torch.load(args.init_checkpoint, map_location='cpu')
        model = ModelForQA.from_pretrained(
            pretrained_model_name_or_path=None, config=model_config, state_dict=state_dict)
    else:
        logger.info("Model is randomly initialized.")
    model.to(device)

    if args.do_train:
        # parameters
        params = list(model.named_parameters())  # 所有参数
        all_trainable_params = divide_parameters(params, lr=args.learning_rate)  # ？
        logger.info("Length of all_trainable_params: %d", len(all_trainable_params))
        try:
            assert sum(map(lambda x: len(x['params']), all_trainable_params)) == len(list(model.parameters()))  # - 2
        except:
            logger.info(f"{sum(map(lambda x: len(x['params']), all_trainable_params))}")
            logger.info(f"{len(list(model.parameters()))}")
            raise AssertionError

        # 优化器采用 BERTAdam
        optimizer = BERTAdam(
            all_trainable_params, lr=args.learning_rate,
            warmup=args.warmup_proportion, t_total=num_train_steps, schedule=args.schedule,
            s_opt1=args.s_opt1, s_opt2=args.s_opt2, s_opt3=args.s_opt3)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)  # ,output_device=n_gpu-1)

    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num split examples = %d", len(train_dataset))
        if args.semi_file:
            logger.info("  Num split semi examples = %d", len(semi_dataset))
            logger.info("  forward semi batch size = %d", args.forward_semi_batch_size)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  forward batch size = %d", args.forward_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        model.train()
        train(model, train_dataset, device, logger, optimizer, n_gpu, eval_dataset, eval_examples, eval_features,
              semi_dataset=semi_dataset, tokenizer=tokenizer)
    if args.do_predict and not args.do_train:
        predict(model, eval_dataset, eval_examples, eval_features, step=0, device=device, logger=logger,
                tokenizer=tokenizer)


if __name__ == "__main__":
    main()
