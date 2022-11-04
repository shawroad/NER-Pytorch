"""
@file   : config.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-11-03
"""
from argparse import ArgumentParser


def set_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=43, help="random seed")
    parser.add_argument("--pretrained_model_path", type=str, default='./bert_pretrain', help="pretrained model")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--log_dir", type=str, default='./logs', help="tensorboard log_dir")
    parser.add_argument("--path_to_train_data", type=str, default='./data/conll2003/train.bio', help="path to train data")
    parser.add_argument("--path_to_test_data", type=str, default='./data/conll2003/test.bio', help="path to test data")
    parser.add_argument("--path_to_prompt_mapper", type=str, default='./data/conll2003/prompt_mapper.json', help="path to prompt mapper")

    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    # output_dir
    parser.add_argument("--output_dir", type=str, default='./output', help="输出文件")
    return parser.parse_args()


def set_infer_args():
    parser = ArgumentParser()
    parser.add_argument("--context", type=str, required=True, help="sentence to extract entities from",)
    parser.add_argument("--question", type=str, required=True, help="question prompt with entity name to extract",)
    parser.add_argument("--pretrained_model_path", type=str, default='./bert_pretrain', help="pretrained model")
    parser.add_argument("--path_to_prompt_mapper", type=str, default='./data/conll2003/prompt_mapper.json', help="path to prompt mapper",)

    parser.add_argument("--path_to_trained_model", type=str, required=True, help="path to trained QaNER model",)
    parser.add_argument("--n_best_size", type=int, required=True, help="number of best QA answers to consider",)
    parser.add_argument("--max_answer_length", type=int, required=False, default=100, help="entity max length",)
    parser.add_argument("--seed", type=int, required=False, default=43, help="seed for reproducibility",)
    parser.add_argument("--save_model_path", type=str, default='./output/epoch_1.bin', help="path to prompt mapper",)
    args = parser.parse_args()
    return args
