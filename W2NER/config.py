"""
@file   : config.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-10-26
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2022, help='随机种子')
    parser.add_argument('--train_data_path', type=str, default='./data/train.json', help='训练数据')
    parser.add_argument('--dev_data_path', type=str, default='./data/dev.json', help='验证数据')
    parser.add_argument('--test_data_path', type=str, default='./data/dev.json', help='测试数据')
    parser.add_argument('--pretrained_model_path', type=str, default='../GlobalPointer/mengzi_pretrain', help='预训练模型路径')
    parser.add_argument('--output_dir', type=str, default='./output', help='输出文件')
    parser.add_argument('--batch_size', type=int, default=16, help='训练批次大小')
    parser.add_argument('--epochs', type=int, default=10, help='训练多少轮')

    parser.add_argument('--bert_learning_rate', type=float, default=1e-5, help='bert模型的学习率')
    parser.add_argument('--weight_decay', type=float, default=0, help='权重衰减')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='其他模型的学习率')
    parser.add_argument('--warm_factor', type=float, default=0.1, help='模型预热的步数占比')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='梯度裁剪的大小')

    parser.add_argument('--use_bert_last_4_layers', type=int, default=1, help="1: true, 0: false  是否使用bert最后四层")
    parser.add_argument('--lstm_hid_size', type=int, default=768)
    parser.add_argument('--conv_hid_size', type=int, default=96)
    parser.add_argument('--dist_emb_size', type=int, default=20)
    parser.add_argument('--type_emb_size', type=int, default=20)
    parser.add_argument('--dilation', type=str, default='123', help="e.g. 1,2,3")
    parser.add_argument('--conv_dropout', type=float, default=0.5)
    parser.add_argument('--emb_dropout', type=float, default=0.5)
    parser.add_argument('--ffnn_hid_size', type=int, default=768)
    parser.add_argument('--biaffine_size', type=int, default=128)
    parser.add_argument('--out_dropout', type=float, default=0.33)
    parser.add_argument('--bert_hid_size', type=int, default=768)
    args = parser.parse_args()
    return args