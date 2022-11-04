"""
@file   : model.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-11-03
"""
from torch import nn
from config import set_args
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from transformers import BertForQuestionAnswering
args = set_args()


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = AutoModelForQuestionAnswering.from_pretrained(args.pretrained_model_path)

    def forward(self, input_ids, token_type_ids, attention_mask, start_positions, end_positions):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, start_positions=start_positions, end_positions=end_positions)
        # loss, start_logits, end_logits
        return output
