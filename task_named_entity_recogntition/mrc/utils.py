import pickle
import os
import config
import logging
import torch
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor
from transformers import squad_convert_examples_to_features

logger = logging.getLogger(__name__)


def load_and_cache_examples(filename, args, tokenizer, evaluate=False, output_examples=False, infix=''):
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    cached_features_file = filename + ".cached_{}_{}_{}".format(
        infix,
        list(filter(None, args.model_name_or_path.split("/"))).pop(),
        str(args.max_seq_length),
    )

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        logger.info("Creating features from dataset file at %s", cached_features_file)

        if not args.data_dir and ((evaluate and not args.predict_file) or (not evaluate and not args.train_file)):
            try:
                import tensorflow_datasets as tfds
            except ImportError:
                raise ImportError("If not data_dir is specified, tensorflow_datasets needs to be installed.")

            if args.version_2_with_negative:
                logger.warning("tensorflow_datasets does not handle version 2 of SQuAD.")

            tfds_examples = tfds.load("squad")
            examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate)
        else:
            processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
            if evaluate:
                examples = processor.get_dev_examples(args.data_dir, filename=filename)
            else:
                examples = processor.get_train_examples(args.data_dir, filename=filename)

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
            return_dataset="pt",
            threads=args.threads,
        )

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    if output_examples:
        return dataset, examples, features
    return dataset


def read_and_convert(fn, is_training, read_fn, convert_fn):
    data_dirname, data_basename = os.path.split(fn)
    cased = '' if config.args.do_lower_case else 'cased'
    if config.args.max_seq_length != 416:
        data_pklname = data_basename + '%s%d_l%d_cHA.pkl' % (cased, config.args.doc_stride, config.args.max_seq_length)
    else:
        data_pklname = data_basename + '%s%d_cHA.pkl' % (cased, config.args.doc_stride)
    full_pklname = os.path.join(data_dirname, data_pklname)
    if os.path.exists(full_pklname):
        logger.info("Loading dataset %s " % data_pklname)
        with open(full_pklname, 'rb') as f:
            examples, features = pickle.load(f)
    else:
        logger.info("Building dataset %s " % data_pklname)
        examples = read_fn(input_file=fn, is_training=is_training)
        logger.info(f"Size: {len(examples)}")
        features = convert_fn(examples=examples, is_training=is_training)
        try:
            with open(full_pklname, 'wb') as f:
                pickle.dump((examples, features), f)
        except:
            logger.info("Can't save train data file.")
    return examples, features


def divide_parameters(named_parameters, lr=None):
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    decay_parameters_names = list(zip(*[(p, n) for n, p in named_parameters if not any((di in n) for di in no_decay)]))
    no_decay_parameters_names = list(zip(*[(p, n) for n, p in named_parameters if any((di in n) for di in no_decay)]))
    param_group = []
    if len(decay_parameters_names) > 0:
        decay_parameters, decay_names = decay_parameters_names
        # print ("decay:",decay_names)
        if lr is not None:
            decay_group = {'params': decay_parameters, 'weight_decay_rate': config.args.weight_decay_rate, 'lr': lr}
        else:
            decay_group = {'params': decay_parameters, 'weight_decay_rate': config.args.weight_decay_rate}
        param_group.append(decay_group)

    if len(no_decay_parameters_names) > 0:
        no_decay_parameters, no_decay_names = no_decay_parameters_names
        # print ("no decay:", no_decay_names)
        if lr is not None:
            no_decay_group = {'params': no_decay_parameters, 'weight_decay_rate': 0.0, 'lr': lr}
        else:
            no_decay_group = {'params': no_decay_parameters, 'weight_decay_rate': 0.0}
        param_group.append(no_decay_group)

    assert len(param_group) > 0
    return param_group
