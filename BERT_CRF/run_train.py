"""
@file   : run_train.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-10-26
"""
import os
import json
import torch
from tqdm import tqdm
from config import set_args
from model import BertCrfForNer
from torch.optim import AdamW
from torch.utils.data import DataLoader
from data_helper import load_data, NERDataset, collate_fn
from transformers.models.bert import BertTokenizer
from transformers import get_linear_schedule_with_warmup


def evaluate():
    model.eval()
    eval_loss, eval_total_steps = 0, 0
    for batch in tqdm(dev_dataloader):
        if torch.cuda.is_available():
            batch = (t.cuda() for t in batch)
        input_ids, label_ids = batch
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=label_ids)
            tmp_eval_loss, logits = outputs[:2]
            attention_mask = torch.ne(input_ids, 0)
            if torch.cuda.is_available():
                attention_mask = attention_mask.cuda()
            tags = model.crf.decode(logits, attention_mask)

        eval_loss += tmp_eval_loss.item()
        eval_total_steps += 1
        true_label_list = label_ids.cpu().detach().numpy().tolist()
        tags = tags.squeeze(0).cpu().detach().numpy().tolist()
    # f1_score
    eval_avg_loss = eval_loss / eval_total_steps
    return eval_avg_loss


if __name__ == '__main__':
    args = set_args()
    os.makedirs(args.output_dir)
    train_data = load_data(args.train_data_path)
    dev_data = load_data(args.dev_data_path)
    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path)

    label2id = json.load(open(args.label2id, 'r', encoding='utf8'))
    train_dataset = NERDataset(train_data, label2id, tokenizer)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=collate_fn)

    dev_dataset = NERDataset(dev_data, label2id, tokenizer)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=collate_fn)

    t_total = len(train_dataset) // args.gradient_accumulation_steps * args.epochs

    model = BertCrfForNer(num_labels=len(label2id))
    if torch.cuda.is_available():
        model.cuda()

    # 对一些参数指定学习
    no_decay = ["bias", "LayerNorm.weight"]
    bert_param_optimizer = list(model.bert.named_parameters())
    crf_param_optimizer = list(model.crf.named_parameters())
    linear_param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.learning_rate},

        {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
        {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.crf_learning_rate},

        {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
        {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.crf_learning_rate}
    ]

    # 定义优化器
    warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)

    print("***** Running training *****")
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    for epoch in range(int(args.epochs)):
        model.train()
        for step, batch in enumerate(train_dataloader):
            if torch.cuda.is_available():
                batch = (t.cuda() for t in batch)
            input_ids, label_ids = batch
            outputs = model(input_ids=input_ids, labels=label_ids)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            print('epoch:{}, step:{}, loss:{:10f}'.format(epoch, step, loss))

            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        eval_avg_loss = evaluate()
        ss = 'Epoch: {} | Eval_Loss: {:10f}'.format(epoch, eval_avg_loss)
        logs_path = os.path.join(args.output_dir, 'logs.txt')
        with open(logs_path, 'a+') as f:
            ss += '\n'
            f.write(ss)
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, "base_model_epoch_{}.bin".format(epoch))
        torch.save(model_to_save.state_dict(), output_model_file)











