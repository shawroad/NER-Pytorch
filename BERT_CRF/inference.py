"""
@file   : inference.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-10-26
"""
import json
import torch
from tqdm import tqdm
from config import set_args
from model import BertCrfForNer
from torch.utils.data import Dataset, DataLoader
from transformers.models.bert import BertTokenizer


class NERDatasetTest(Dataset):
    def __init__(self, data, tokenizer):
        super(NERDatasetTest, self).__init__()
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        words = self.data[item]
        input_ids = self.tokenizer.convert_tokens_to_ids(words)

        # 加cls 和 sep
        input_ids.insert(0, self.tokenizer.cls_token_id)
        input_ids.append(self.tokenizer.sep_token_id)
        return {'input_ids': input_ids}


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
    all_input_ids = torch.tensor(input_ids, dtype=torch.long)
    return all_input_ids


def infer():
    eval_loss, eval_total_steps = 0, 0
    for batch in tqdm(test_dataloader):
        if torch.cuda.is_available():
            batch = (t.cuda() for t in batch)
        input_ids = batch
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs[0]
            attention_mask = torch.ne(input_ids, 0)
            if torch.cuda.is_available():
                attention_mask = attention_mask.cuda()
            tags = model.crf.decode(logits, attention_mask)
        tags = tags.squeeze(0).cpu().detach().numpy().tolist()
        label = [[id2label[i] for i in tag] for tag in tags]
        print(label)
        exit()


if __name__ == '__main__':
    args = set_args()
    label2id = json.load(open(args.label2id, 'r', encoding='utf8'))
    id2label = {ids: lab for lab, ids in label2id.items()}
    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path)

    model = BertCrfForNer(len(label2id))

    model.load_state_dict(torch.load('./outputs/base_model_epoch_0.bin'))
    if torch.cuda.is_available():
        model.cuda()

    model.eval()

    texts = ['金山软件和小米公司是从属关系吗', '国家领导人是多长时间换一次']
    corpus = []
    for text in texts:
        corpus.append(list(text))

    test_dataset = NERDatasetTest(corpus, tokenizer)

    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=2, collate_fn=collate_fn)
    infer()










