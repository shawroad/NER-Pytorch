"""
@file   : gen_label2id.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-10-26
"""
import json


if __name__ == '__main__':
    label = set()
    with open('train.txt', 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                # 空行
                continue
            line = line.split(' ')
            label.add(line[1])
    print(len(label))

    with open('dev.txt', 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            line = line.split(' ')
            label.add(line[1])
    print(len(label))

    id2label = {i: lab for i, lab in enumerate(list(label))}
    label2id = {lab: i for i, lab in enumerate(list(label))}
    print(id2label)
    print(label2id)
    json.dump(id2label, open('id2label.json', 'w', encoding='utf8'), ensure_ascii=False)
    json.dump(label2id, open('label2id.json', 'w', encoding='utf8'), ensure_ascii=False)











