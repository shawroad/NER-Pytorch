"""
@file   : data_helper.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-10-26
"""
import torch
import json
import numpy as np
from torch.utils.data import Dataset
from utils import convert_index_to_text
from torch.nn.utils.rnn import pad_sequence
from transformers.models.bert import BertTokenizer


dis2idx = np.zeros((1000), dtype='int64')
dis2idx[1] = 1
dis2idx[2:] = 2
dis2idx[4:] = 3
dis2idx[8:] = 4
dis2idx[16:] = 5
dis2idx[32:] = 6
dis2idx[64:] = 7
dis2idx[128:] = 8
dis2idx[256:] = 9


class Vocabulary(object):
    # 主要是做实体类型统计
    PAD = '<pad>'
    UNK = '<unk>'
    SUC = '<suc>'

    def __init__(self):
        self.label2id = {self.PAD: 0, self.SUC: 1}
        self.id2label = {0: self.PAD, 1: self.SUC}

    def add_label(self, label):
        label = label.lower()
        if label not in self.label2id:
            self.label2id[label] = len(self.label2id)
            self.id2label[self.label2id[label]] = label
        assert label == self.id2label[self.label2id[label]]

    def label_to_id(self, label):
        label = label.lower()
        return self.label2id[label]

    def id_to_label(self, i):
        return self.id2label[i]


def fill_vocab(vocab, dataset):
    entity_num = 0
    for instance in dataset:
        for entity in instance["ner"]:
            vocab.add_label(entity["type"])   # 加入实体类型 并转为id
        entity_num += len(instance["ner"])   # 统计实体个数
    return entity_num


class RelationDataset(Dataset):
    def __init__(self, bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text):
        self.bert_inputs = bert_inputs
        self.grid_labels = grid_labels
        self.grid_mask2d = grid_mask2d
        self.pieces2word = pieces2word
        self.dist_inputs = dist_inputs
        self.sent_length = sent_length
        self.entity_text = entity_text

    def __getitem__(self, item):
        return torch.LongTensor(self.bert_inputs[item]), \
               torch.LongTensor(self.grid_labels[item]), \
               torch.LongTensor(self.grid_mask2d[item]), \
               torch.LongTensor(self.pieces2word[item]), \
               torch.LongTensor(self.dist_inputs[item]), \
               self.sent_length[item], \
               self.entity_text[item]

    def __len__(self):
        return len(self.bert_inputs)


def process_bert(data, tokenizer, vocab):
    bert_inputs = []
    grid_labels = []
    grid_mask2d = []
    dist_inputs = []
    entity_text = []
    pieces2word = []
    sent_length = []
    for index, instance in enumerate(data):
        if len(instance['sentence']) == 0:
            continue
        # 遍历每个词  用bert针对每个词进行分词
        tokens = [tokenizer.tokenize(word) for word in instance['sentence']]
        # print(tokens)
        # [['e', '##u'], ['re', '##ject', '##s'], ['ge', '##rman'], ['call'], ['to'],
        # 以字符为单位分词
        pieces = [piece for pieces in tokens for piece in pieces]
        # print(pieces)
        # ['e', '##u', 're', '##ject', '##s', 'ge', '##rman', 'call', 'to', 'boy', '##co', '##

        _bert_inputs = tokenizer.convert_tokens_to_ids(pieces)
        _bert_inputs = np.array([tokenizer.cls_token_id] + _bert_inputs + [tokenizer.sep_token_id])

        length = len(instance['sentence'])
        _grid_labels = np.zeros((length, length), dtype=np.int)
        _pieces2word = np.zeros((length, len(_bert_inputs)), dtype=np.bool)   # 每行代表一个词在bert的输入的位置 有多少个词  就有多少行
        _dist_inputs = np.zeros((length, length), dtype=np.int)
        _grid_mask2d = np.ones((length, length), dtype=np.bool)

        if tokenizer is not None:
            start = 0
            for i, pieces in enumerate(tokens):
                if len(pieces) == 0:
                    continue
                pieces = list(range(start, start + len(pieces)))   # 0, 0+2
                _pieces2word[i, pieces[0] + 1:pieces[-1] + 2] = 1  # 这里+1 cls   这里最后+2  cls sep
                start += len(pieces)
        # print(_pieces2word)
        '''
        [[False  True  True False False False False False False False False False False False False False False False False False]
         [False False False  True  True  True False False False False False False False False False False False False False False]
         [False False False False False False  True  True False False False False False False False False False False False False]
         [False False False False False False False False  True False False False False False False False False False False False]
         [False False False False False False False False False  True False False False False False False False False False False]
         [False False False False False False False False False False  True  True True False False False False False False False]
         [False False False False False False False False False False False False False  True  True  True False False False False]
         [False False False False False False False False False False False False False False False False  True  True False False]
         [False False False False False False False False False False False False False False False False False False  True False]]
        '''

        for k in range(length):
            _dist_inputs[k, :] += k
            _dist_inputs[:, k] -= k
        '''
        [[ 0 -1 -2 -3 -4 -5 -6 -7 -8]
         [ 1  0 -1 -2 -3 -4 -5 -6 -7]
         [ 2  1  0 -1 -2 -3 -4 -5 -6]
         [ 3  2  1  0 -1 -2 -3 -4 -5]
         [ 4  3  2  1  0 -1 -2 -3 -4]
         [ 5  4  3  2  1  0 -1 -2 -3]
         [ 6  5  4  3  2  1  0 -1 -2]
         [ 7  6  5  4  3  2  1  0 -1]
         [ 8  7  6  5  4  3  2  1  0]]
        '''
        for i in range(length):
            for j in range(length):
                if _dist_inputs[i, j] < 0:
                    _dist_inputs[i, j] = dis2idx[-_dist_inputs[i, j]] + 9
                else:
                    _dist_inputs[i, j] = dis2idx[_dist_inputs[i, j]]

        # print(_dist_inputs)
        '''
        [[ 0 10 11 11 12 12 12 12 13]
         [ 1  0 10 11 11 12 12 12 12]
         [ 2  1  0 10 11 11 12 12 12]
         [ 2  2  1  0 10 11 11 12 12]
         [ 3  2  2  1  0 10 11 11 12]
         [ 3  3  2  2  1  0 10 11 11]
         [ 3  3  3  2  2  1  0 10 11]
         [ 3  3  3  3  2  2  1  0 10]
         [ 4  3  3  3  3  2  2  1  0]]
        '''
        _dist_inputs[_dist_inputs == 0] = 19

        # "ner": [{"index": [0, 1], "type": "PER"}]}
        for entity in instance["ner"]:
            index = entity["index"]
            for i in range(len(index)):
                if i + 1 >= len(index):
                    break
                _grid_labels[index[i], index[i + 1]] = 1
            _grid_labels[index[-1], index[0]] = vocab.label_to_id(entity["type"])
        # print(_grid_labels)
        '''
        [[2 0 0 0 0 0 0 0 0]
         [0 0 0 0 0 0 0 0 0]
         [0 0 3 0 0 0 0 0 0]
         [0 0 0 0 0 0 0 0 0]
         [0 0 0 0 0 0 0 0 0]
         [0 0 0 0 0 0 0 0 0]
         [0 0 0 0 0 0 3 0 0]
         [0 0 0 0 0 0 0 0 0]
         [0 0 0 0 0 0 0 0 0]]
        '''
        _entity_text = set([convert_index_to_text(e["index"], vocab.label_to_id(e["type"])) for e in instance["ner"]])
        sent_length.append(length)
        bert_inputs.append(_bert_inputs)
        grid_labels.append(_grid_labels)
        grid_mask2d.append(_grid_mask2d)
        dist_inputs.append(_dist_inputs)
        pieces2word.append(_pieces2word)
        entity_text.append(_entity_text)
    return bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text


def load_data(args):
    # 加载数据集
    with open(args.train_data_path, 'r', encoding='utf8') as f:
        train_data = json.load(f)
    with open(args.dev_data_path, 'r', encoding='utf8') as f:
        dev_data = json.load(f)

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path)

    vocab = Vocabulary()
    train_ent_num = fill_vocab(vocab, train_data)   # 返回的是实体的个数
    dev_ent_num = fill_vocab(vocab, dev_data)
    print('训练集实体个数:', train_ent_num)
    print('验证集实体个数:', dev_ent_num)

    label_num = len(vocab.label2id)
    train_dataset = RelationDataset(*process_bert(train_data, tokenizer, vocab))
    dev_dataset = RelationDataset(*process_bert(dev_data, tokenizer, vocab))
    return train_dataset, dev_dataset, label_num


def collate_fn(data):
    bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text = map(list, zip(*data))
    max_tok = np.max(sent_length)
    sent_length = torch.LongTensor(sent_length)
    max_pie = np.max([x.shape[0] for x in bert_inputs])
    bert_inputs = pad_sequence(bert_inputs, True)
    batch_size = bert_inputs.size(0)

    def fill(data, new_data):
        for j, x in enumerate(data):
            new_data[j, :x.shape[0], :x.shape[1]] = x
        return new_data

    dis_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    dist_inputs = fill(dist_inputs, dis_mat)
    labels_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    grid_labels = fill(grid_labels, labels_mat)
    mask2d_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.bool)
    grid_mask2d = fill(grid_mask2d, mask2d_mat)
    sub_mat = torch.zeros((batch_size, max_tok, max_pie), dtype=torch.bool)
    pieces2word = fill(pieces2word, sub_mat)
    return bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text
