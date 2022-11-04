"""
@file   : model.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-10-26
"""
import torch
from torch import nn
from crf import CRF
from config import set_args
from transformers import BertModel, BertConfig


args = set_args()


class BertCrfForNer(nn.Module):
    def __init__(self, num_labels):
        super(BertCrfForNer, self).__init__()
        # bert模型
        self.config = BertConfig.from_pretrained(args.pretrain_model_path)
        self.bert = BertModel.from_pretrained(args.pretrain_model_path)

        # 每个token进行分类
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

        # 送入CRF进行预测
        self.crf = CRF(num_tags=num_labels, batch_first=True)

    def forward(self, input_ids, labels=None):
        attention_mask = torch.ne(input_ids, 0)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]   # B, L, H
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
            outputs = (-1*loss,) + outputs
        return outputs # (loss), scores