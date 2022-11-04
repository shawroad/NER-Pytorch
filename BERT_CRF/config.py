"""
@file   : config.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-10-26
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", default='./data/train.txt', type=str, required=False, help='训练集')
    parser.add_argument("--dev_data_path", default='./data/dev.txt', type=str, required=False, help='验证集')
    parser.add_argument("--pretrain_model_path", default='../GlobalPointer/mengzi_pretrain', type=str, required=False, help='预训练模型')
    parser.add_argument("--label2id", default='./data/label2id.json', type=str, required=False, help='标签')

    parser.add_argument("--output_dir", default='./outputs', type=str, required=False, help='this output directory')

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="For distant debugging.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help='the initial learning rate for Adam')
    parser.add_argument("--crf_learning_rate", default=5e-5, type=float, help='the initial learning rate for '
                                                                              'crf and linear layer')
    parser.add_argument("--epochs", type=int, default=10, help="For distant debugging.")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")

    parser.add_argument("--warmup_proportion", default=0.05, type=float, help='the initial learning rate for Adam')
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help='Epsilon for Adam optimizer')
    parser.add_argument("--weight_decay", default=0.01, type=float, help='Weight decay if we apply some')
    return parser.parse_args()
