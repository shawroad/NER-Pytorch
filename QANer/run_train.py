"""
@file   : run_train.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-11-03
"""
import json
import os
import torch
import random
import numpy as np
from model import Model
from config import set_args
from collections import defaultdict
from inference import get_top_valid_spans
from utils import compute_metrics
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from data_helper import MyDataset, Collator, Span
from torch.utils.tensorboard import SummaryWriter
from data_helper import load_data, prepare_sentences_and_spans


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    args = set_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    # tensorboard --logdir=./logs
    writer = SummaryWriter(log_dir=args.log_dir)
    # 1. 分词器
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)
    tokenizer_kwargs = {
        "max_length": 512,
        "truncation": "only_second",
        "padding": True,
        "return_tensors": "pt",
        "return_offsets_mapping": True,
    }
    # 2. 将ner的tag转为nl
    with open(args.path_to_prompt_mapper, mode="r", encoding="utf-8") as fp:
        prompt_mapper = json.load(fp)

    # 训练集
    train_token_seq, train_label_seq = load_data(args.path_to_train_data)
    # print(train_token_seq[:2])
    # [['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.'], ['Peter', 'Blackburn']]
    # print(train_label_seq[:2])
    # [['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'O', 'O'], ['B-PER', 'I-PER']]
    train_qa_sentences, train_qa_labels = prepare_sentences_and_spans(token_seq=train_token_seq, label_seq=train_label_seq)
    train_dataset = MyDataset(qa_sentences=train_qa_sentences, qa_labels=train_qa_labels, prompt_mapper=prompt_mapper)

    # 测试集
    test_token_seq, test_label_seq = load_data(args.path_to_test_data)
    test_qa_sentences, test_qa_labels = prepare_sentences_and_spans(token_seq=test_token_seq, label_seq=test_label_seq)
    test_dataset = MyDataset(qa_sentences=test_qa_sentences, qa_labels=test_qa_labels, prompt_mapper=prompt_mapper)

    collator = Collator(tokenizer=tokenizer, tokenizer_kwargs=tokenizer_kwargs)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    model = Model()
    if torch.cuda.is_available():
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    epoch_loss = []

    batch_metrics_list = defaultdict(list)
    for epoch in range(10):
        model.train()
        optimizer.zero_grad()

        for step, inputs in enumerate(train_dataloader):
            instances_batch = inputs.pop("instances")

            context_list, question_list = [], []
            for instance in instances_batch:
                context_list.append(instance.context)
                question_list.append(instance.question)

            if torch.cuda.is_available():
                inputs = inputs.cuda()

            offset_mapping_batch = inputs.pop("offset_mapping")
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            print('epoch:{}, step:{}, loss:{:10f}'.format(epoch, step, loss))
            epoch_loss.append(loss.item())
            writer.add_scalar(
                "batch loss / train", loss.item(), epoch * len(train_dataloader) + step
            )

            model.eval()
            with torch.no_grad():
                outputs_inference = model(**inputs)
                # loss, start_logits, end_logits
                # print(outputs_inference.start_logits.size())   # torch.Size([2, 52])

            model.train()
            # print(offset_mapping_batch.size())   # batch_size, 52, 2
            spans_pred_batch_top_1 = get_top_valid_spans(
                context_list=context_list,
                question_list=question_list,
                prompt_mapper=train_dataloader.dataset.prompt_mapper,
                inputs=inputs,
                outputs=outputs_inference,
                offset_mapping_batch=offset_mapping_batch,
                n_best_size=1,
                max_answer_length=100
            )
            # print(spans_pred_batch_top_1)
            # [[Span(token='in', label='ORG', start_context_char_pos=43, end_context_char_pos=45)], []]

            # 预测的top1
            for idx in range(len(spans_pred_batch_top_1)):
                if not spans_pred_batch_top_1[idx]:
                    # 没有预测位置 全部为"O"标签
                    empty_span = Span(
                        token='',
                        label='O',
                        start_context_char_pos=0,
                        end_context_char_pos=0
                    )
                    spans_pred_batch_top_1[idx] = [empty_span]
            spans_true_batch = [instance.answer for instance in instances_batch]
            # [[Span(token='in', label='ORG', start_context_char_pos=43, end_context_char_pos=45)], []]
            # print(spans_true_batch)

            # [Span(token='PSC', label='ORG', start_context_char_pos=32, end_context_char_pos=35),
            #  Span(token='Washington', label='LOC', start_context_char_pos=153, end_context_char_pos=163)]

            batch_metrics = compute_metrics(
                spans_true_batch=spans_true_batch,
                spans_pred_batch_top_1=spans_pred_batch_top_1,
                prompt_mapper=train_dataloader.dataset.prompt_mapper
            )
            # print(batch_metrics)
            # {'accuracy': 0.0, 'precision_tag_O': 0.0, 'recall_tag_O': 0.0, 'f1_tag_O': 0.0, 'precision_tag_LOC': 0.0,
            # 'recall_tag_LOC': 0.0, 'f1_tag_LOC': 0.0, 'precision_tag_PER': 0.0, 'recall_tag_PER': 0.0,
            # 'f1_tag_PER': 0.0, 'precision_tag_ORG': 0.0, 'recall_tag_ORG': 0.0, 'f1_tag_ORG': 0.0,
            # 'precision_tag_MISC': 0.0, 'recall_tag_MISC': 0.0, 'f1_tag_MISC': 0.0, 'precision_macro': 0.0,
            # 'recall_macro': 0.0, 'f1_macro': 0.0, 'precision_weighted': 0.0, 'recall_weighted': 0.0,
            # 'f1_weighted': 0.0}
            for metric_name, metric_value in batch_metrics.items():
                batch_metrics_list[metric_name].append(metric_value)
                writer.add_scalar(
                    f"batch {metric_name} / train",
                    metric_value,
                    epoch * len(train_dataloader) + step,
                )
        avg_loss = np.mean(epoch_loss)
        print("Train Loss:{:10f}".format(avg_loss))
        writer.add_scalar("loss / train", avg_loss, epoch)

        for metric_name, metric_value_list in batch_metrics_list.items():
            metric_value = np.mean(metric_value_list)
            print(f"Train {metric_name}: {metric_value}\n")
            writer.add_scalar(f"{metric_name} / train", metric_value, epoch)

        # 保存模型
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, "epoch{}_ckpt.bin".format(epoch))
        torch.save(model_to_save.state_dict(), output_model_file)
