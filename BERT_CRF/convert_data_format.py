"""
@file   : convert_data_format.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-10-26
"""
import json


def convert_data_SBMEO(path):
    # 这种应用的时S B M E O
    result = ''
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line.strip())
            temp_dict = {}
            for i, w in enumerate(line['text']):
                temp_dict[i] = [w]
            # print(line['label'])   # {'name': {'叶老桂': [[9, 11]]}, 'company': {'浙商银行': [[0, 3]]}}
            label = line['label']
            for name, entity_dict in label.items():
                for ent, index in entity_dict.items():
                    for ind in index:
                        # 单实体
                        if len(ind) == 0:
                            temp_dict[ind[0]].append('S-{}'.format(name))
                            continue

                        # 两个字
                        temp_dict[ind[0]].append('B-{}'.format(name))
                        temp_dict[ind[-1]].append('E-{}'.format(name))

                        # 如果有三个字
                        if ind[-1] - ind[0] > 1:
                            for j in range(ind[0]+1, ind[-1]):
                                temp_dict[j].append('M-{}'.format(name))

            ss = ''
            for i, j in temp_dict.items():
                if len(j) == 1:
                    ss += j[0] + ' ' + 'O' + '\n'
                    continue
                ss += j[0] + ' ' + j[1] + '\n'
            result += ss + '\n'
    return result


def convert_data_SBIO(path):
    # 这种应用的时B I O
    result = ''
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line.strip())
            temp_dict = {}
            for i, w in enumerate(line['text']):
                temp_dict[i] = [w]
            # print(line['label'])   # {'name': {'叶老桂': [[9, 11]]}, 'company': {'浙商银行': [[0, 3]]}}
            label = line['label']
            for name, entity_dict in label.items():
                for ent, index in entity_dict.items():
                    for ind in index:
                        # 单实体
                        if len(ind) == 0:
                            temp_dict[ind[0]].append('S-{}'.format(name))
                            continue

                        # 两个字以上
                        temp_dict[ind[0]].append('B-{}'.format(name))
                        for j in range(ind[0]+1, ind[-1]+1):
                            temp_dict[j].append('I-{}'.format(name))

            ss = ''
            for i, j in temp_dict.items():
                if len(j) == 1:
                    ss += j[0] + ' ' + 'O' + '\n'
                    continue
                ss += j[0] + ' ' + j[1] + '\n'
            result += ss + '\n'
    return result


if __name__ == '__main__':
    train_data_path = '../data/train.json'
    train_data = convert_data_SBIO(train_data_path)
    # train_data = convert_data_SBMEO(train_data_path)
    with open('./data/train.txt', 'w', encoding='utf8') as f:
        f.write(train_data)

    dev_data_path = '../data/dev.json'
    dev_data = convert_data_SBIO(dev_data_path)
    # dev_data = convert_data_SBMEO(dev_data_path)
    with open('./data/dev.txt', 'w', encoding='utf8') as f:
        f.write(dev_data)
