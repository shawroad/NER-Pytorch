"""
@file   : convert_data_format.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-10-26
"""
import jieba
import json
from tqdm import tqdm


def convert_data(path):
    result = []
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line = json.loads(line.strip())
            text = line['text']
            label = line["label"]

            # 1. 找出分词后的索引
            words = jieba.lcut(text)
            word_index = []
            start = 0
            for w in words:
                word_index.append([i for i in range(start, start + len(w))])
                start += len(w)

            # 2. 找出实体
            ner = []
            # print(label)   # {'name': {'叶老桂': [[9, 11]]}, 'company': {'浙商银行': [[0, 3]]}}
            for name, entity in label.items():
                for ent, index in entity.items():
                    for ind in index:
                        s = [ind[0] + i for i in range(len(ent))]
                        ner.append({'index': s, 'type': name})
            result.append({'sentence': list(text), 'ner': ner, 'word': word_index})
    return result


if __name__ == '__main__':
    train_path = '../data/train.json'
    train_data = convert_data(train_path)
    json.dump(train_data, open('./data/train.json', 'w', encoding='utf8'), ensure_ascii=False)
    dev_path = '../data/dev.json'
    dev_data = convert_data(dev_path)
    json.dump(dev_data, open('./data/dev.json', 'w', encoding='utf8'), ensure_ascii=False)






