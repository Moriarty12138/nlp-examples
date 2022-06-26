import torch
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="../../models/bert-base-chinese")
    parser.add_argument('--train_dataset_path', type=str, default="../../datasets/Event_Competition/train_7000.json")
    parser.add_argument('--valid_dataset_path', type=str, default="../../datasets/Event_Competition/valid_1500.json")
    parser.add_argument('--test_dataset_path', type=str, default="../../datasets/Event_Competition/valid_1500.json")
    parser.add_argument('--model_save_path', type=str, default="./output")
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--seed', type=int, default=1202)
    parser.add_argument('--no_cuda', type=bool, default=False)
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--valid_batch_size', type=int, default=32)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    parser.add_argument('--do_train', type=bool, default=True)
    parser.add_argument('--do_test', type=bool, default=True)
    parser.add_argument('--do_eval', type=bool, default=True)

    args = parser.parse_args()

    args.device = torch.device('cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu')

    return args
