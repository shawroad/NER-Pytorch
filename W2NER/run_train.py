"""
@file   : run_train.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-10-26
"""
import os
import torch
import random
import numpy as np
from torch import nn
from tqdm import tqdm
from model import Model
from config import set_args
from utils import decode, cal_f1
from torch.optim import AdamW
from torch.utils.data import DataLoader
from data_helper import load_data, collate_fn
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support, f1_score


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def evaluate(data_loader):
    model.eval()
    pred_result, label_result = [], []
    total_ent_r, total_ent_p, total_ent_c = 0, 0, 0
    with torch.no_grad():
        for batch in tqdm(data_loader):
            entity_text = batch[-1]
            batch = batch[:-1]
            if torch.cuda.is_available():
                batch = [t.cuda() for t in batch]
            bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = batch
            outputs = model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
            length = sent_length
            grid_mask2d = grid_mask2d.clone()

            outputs = torch.argmax(outputs, -1)
            ent_c, ent_p, ent_r, _ = decode(outputs.cpu().numpy(), entity_text, length.cpu().numpy())
            total_ent_r += ent_r
            total_ent_p += ent_p
            total_ent_c += ent_c
            grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
            outputs = outputs[grid_mask2d].contiguous().view(-1)
            label_result.append(grid_labels.cpu())
            pred_result.append(outputs.cpu())

    label_result = torch.cat(label_result)
    pred_result = torch.cat(pred_result)

    p, r, f1, _ = precision_recall_fscore_support(label_result.numpy(),
                                                  pred_result.numpy(),
                                                  average="macro")
    e_f1, e_p, e_r = cal_f1(total_ent_c, total_ent_p, total_ent_r)
    return e_f1


if __name__ == '__main__':
    args = set_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    train_dataset, dev_dataset, label_num = load_data(args)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True, drop_last=True)
    dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False, drop_last=False)

    total_num = len(train_dataset) // args.batch_size * args.epochs

    model = Model(label_num)
    if torch.cuda.is_available():
        model.cuda()

    loss_func = nn.CrossEntropyLoss()

    bert_params = set(model.bert.parameters())
    other_params = list(set(model.parameters()) - bert_params)
    no_decay = ['bias', 'LayerNorm.weight']
    params = [
        {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
         'lr': args.bert_learning_rate,
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
         'lr': args.bert_learning_rate,
         'weight_decay': 0.0},
        {'params': other_params,
         'lr': args.learning_rate,
         'weight_decay': args.weight_decay},
    ]

    optimizer = AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warm_factor * total_num,
                                                num_training_steps=total_num)
    best_f1 = 0
    best_test_f1 = 0
    for epoch in range(args.epochs):
        model.train()
        loss_list = []
        pred_result = []
        label_result = []
        for step, batch in enumerate(train_dataloader):
            batch = batch[:-1]
            if torch.cuda.is_available():
                batch = (t.cuda() for t in batch)
            bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = batch
            # print(bert_inputs.size())   # torch.Size([16, 59])
            # print(grid_labels.size())   # torch.Size([16, 26, 26])
            # print(grid_mask2d.size())   # torch.Size([16, 26, 26])
            # print(pieces2word.size())   # torch.Size([16, 26, 59])
            # print(dist_inputs.size())   # torch.Size([16, 26, 26])
            # print(sent_length.size())   # torch.Size([16])
            outputs = model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
            grid_mask2d = grid_mask2d.clone()
            loss = loss_func(outputs[grid_mask2d], grid_labels[grid_mask2d])
            print('epoch:{}, step:{}, loss:{:10f}'.format(epoch, step, loss))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)  # 梯度裁剪
            optimizer.step()
            optimizer.zero_grad()
            loss_list.append(loss.cpu().item())
            outputs = torch.argmax(outputs, -1)
            grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
            outputs = outputs[grid_mask2d].contiguous().view(-1)

            label_result.append(grid_labels.cpu())
            pred_result.append(outputs.cpu())
            scheduler.step()

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)

        p, r, f1, _ = precision_recall_fscore_support(label_result.numpy(), pred_result.numpy(), average="macro")

        val_f1 = evaluate(dev_dataloader)
        log_path = os.path.join(args.output_dir, 'logs.txt')
        with open(log_path, 'a+', encoding='utf8') as f:
            s = 'train_f1:{:10f}, val_f1:{:10f}'.format(f1, val_f1)
            f.write(s+'\n')

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.bin'))