"""
@file   : data_helper.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-10-26
"""
import torch
from torch.utils.data import Dataset


def load_data(path):
    # lines.append({"words": words, "labels": labels})
    result = []
    with open(path, 'r', encoding='utf8') as f:
        words, labels = [], []
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                result.append({"words": words, "labels": labels})
                words, labels = [], []
            else:
                w, lab = line.split(' ')
                words.append(w)
                labels.append(lab)
    return result


class NERDataset(Dataset):
    def __init__(self, data, label2id, tokenizer):
        super(NERDataset, self).__init__()
        self.data = data
        self.label2id = label2id
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        words = self.data[item]['words']
        labels = self.data[item]['labels']

        # 将label转为id序列
        labels_ids = []
        for lab in labels:
            labels_ids.append(self.label2id[lab])

        input_ids = self.tokenizer.convert_tokens_to_ids(words)

        # 加cls 和 sep
        input_ids.insert(0, self.tokenizer.cls_token_id)
        input_ids.append(self.tokenizer.sep_token_id)
        labels_ids.insert(0, self.label2id['O'])
        labels_ids.append(self.label2id['O'])
        assert len(input_ids) == len(labels_ids)
        return {'input_ids': input_ids, 'label': labels_ids}


def pad_to_maxlen(input_ids, max_len, pad_value=0):
    if len(input_ids) >= max_len:
        input_ids = input_ids[:max_len]
    else:
        input_ids = input_ids + [pad_value] * (max_len - len(input_ids))
    return input_ids


def collate_fn(batch):
    max_len = max([len(d['input_ids']) for d in batch])
    if max_len > 512:
        max_len = 512

    input_ids, labels = [], []
    for item in batch:
        input_ids.append(pad_to_maxlen(item['input_ids'], max_len=max_len))
        labels.append(pad_to_maxlen(item['label'], max_len=max_len))

    all_input_ids = torch.tensor(input_ids, dtype=torch.long)
    all_label_ids = torch.tensor(labels, dtype=torch.long)
    return all_input_ids, all_label_ids
