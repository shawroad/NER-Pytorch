"""
@file   : data_helper.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-11-03
"""
import torch
from tqdm import tqdm, trange
from collections import namedtuple
from torch.utils.data import Dataset


def load_data(path):
    token_seq, label_seq = [], []
    with open(path, 'r', encoding='utf8') as f:
        tokens, labels = [], []
        for line in tqdm(f):
            if line != '\n':
                label, token = line.strip().split('\t')
                tokens.append(token)
                labels.append(label)
            else:
                if len(tokens) > 0:
                    token_seq.append(tokens)
                    label_seq.append(labels)
                tokens, labels = [], []
    return token_seq, label_seq


Span = namedtuple(
    "Span", ["token", "label", "start_context_char_pos", "end_context_char_pos"]
)
Instance = namedtuple("Instance", ["context", "question", "answer"])


def prepare_sentences_and_spans(token_seq, label_seq):
    qa_sentences, qa_labels = [], []
    for i in trange(len(token_seq)):
        current_char_pos = 0
        qa_sent, qa_lab = [], []

        for token, label in zip(token_seq[i], label_seq[i]):
            # 取出一条数据 遍历每个token
            qa_sent.append(token)
            if label != "O":
                span = Span(
                    token=token,
                    label=label,
                    start_context_char_pos=current_char_pos,    # 统计的是按字符去数的位置
                    end_context_char_pos=current_char_pos + len(token),
                )
                qa_lab.append(span)
            current_char_pos += len(token) + 1
        # print(" ".join(qa_sent))    # EU rejects German call to boycott British lamb .
        # print(qa_lab)
        # [Span(token='EU', label='B-ORG', start_context_char_pos=0, end_context_char_pos=2),
        #  Span(token='German', label='B-MISC', start_context_char_pos=11, end_context_char_pos=17),
        #  Span(token='British', label='B-MISC', start_context_char_pos=34, end_context_char_pos=41)]
        qa_sentences.append(" ".join(qa_sent))
        qa_labels.append(qa_lab)

    qa_labels_v2 = []
    for qa_lab in qa_labels:
        qa_lab_v2 = []
        for span in qa_lab:
            if span.label.startswith("B-"):   # 如果是B开头的标签
                span_v2 = Span(
                    token=span.token,
                    label=span.label.split("-")[-1],   # 把 label前面的B去掉
                    start_context_char_pos=span.start_context_char_pos,
                    end_context_char_pos=span.end_context_char_pos,
                )
                qa_lab_v2.append(span_v2)
            elif span.label.startswith("I-"):   # 如果是I开头的标签
                # TODO: remove duplicates and optimize
                span_v2 = Span(  # TODO: maybe use Span as dataclass
                    token=f"{span_v2.token} {span.token}",
                    label=span_v2.label,
                    start_context_char_pos=span_v2.start_context_char_pos,
                    end_context_char_pos=span.end_context_char_pos,
                )
                try:
                    qa_lab_v2[-1] = span_v2
                except:
                    continue
            else:
                raise ValueError(f"Unknown label: {span.label}")
        # print(qa_lab_v2)
        # [Span(token='EU', label='ORG', start_context_char_pos=0, end_context_char_pos=2),
        #  Span(token='German', label='MISC', start_context_char_pos=11, end_context_char_pos=17),
        #  Span(token='British', label='MISC', start_context_char_pos=34, end_context_char_pos=41)]
        qa_labels_v2.append(qa_lab_v2)
    return qa_sentences, qa_labels_v2


class MyDataset(Dataset):
    def __init__(self, qa_sentences, qa_labels, prompt_mapper):
        super(MyDataset, self).__init__()
        self.prompt_mapper = prompt_mapper
        self.dataset = self._prepare_dataset(
            qa_sentences=qa_sentences,
            qa_labels=qa_labels,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Instance:
        return self.dataset[idx]

    def _prepare_dataset(self, qa_sentences, qa_labels,):
        dataset = []
        # 对所有的数据处理
        for sentence, labels in zip(qa_sentences, qa_labels):
            for label_tag, label_name in self.prompt_mapper.items():
                question_prompt = f"What is the {label_name}?"
                answer_list = []
                for span in labels:
                    if span.label == label_tag:
                        answer_list.append(span)

                if len(answer_list) == 0:
                    empty_span = Span(
                        token="",
                        label="O",
                        start_context_char_pos=0,
                        end_context_char_pos=0,
                    )
                    instance = Instance(
                        context=sentence,
                        question=question_prompt,
                        answer=empty_span,
                    )
                    dataset.append(instance)
                else:
                    for answer in answer_list:
                        instance = Instance(
                            context=sentence,
                            question=question_prompt,
                            answer=answer,
                        )
                        dataset.append(instance)
            # print(dataset)
            # [Instance(context='EU rejects German call to boycott British lamb .', question='What is the location?',
            #           answer=Span(token='', label='O', start_context_char_pos=0, end_context_char_pos=0)),
            #  Instance(context='EU rejects German call to boycott British lamb .', question='What is the person?',
            #           answer=Span(token='', label='O', start_context_char_pos=0, end_context_char_pos=0)),
            #  Instance(context='EU rejects German call to boycott British lamb .',question='What is the organization?',
            #           answer=Span(token='EU', label='ORG', start_context_char_pos=0, end_context_char_pos=2)),
            #  Instance(context='EU rejects German call to boycott British lamb .',
            #           question='What is the miscellaneous entity?',
            #           answer=Span(token='German', label='MISC', start_context_char_pos=11, end_context_char_pos=17)),
            #  Instance(context='EU rejects German call to boycott British lamb .',
            #           question='What is the miscellaneous entity?',
            #         answer=Span(token='British', label='MISC', start_context_char_pos=34, end_context_char_pos=41))]
        return dataset


class Collator:
    def __init__(self, tokenizer, tokenizer_kwargs):
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs

    def __call__(self, batch):
        context_list = []
        question_list = []
        start_end_context_char_pos_list = []

        for instance in batch:
            context_list.append(instance.context)
            question_list.append(instance.question)
            start_end_context_char_pos_list.append(
                [
                    instance.answer.start_context_char_pos,
                    instance.answer.end_context_char_pos,
                ]
            )

        # 对问题 和 文本 转为 [CLS] question [SEP] answer [SEP]
        tokenized_batch = self.tokenizer(
            question_list, context_list, **self.tokenizer_kwargs
        )
        # print(tokenized_batch)
        # 单个样本:[  101,  2054,  2003,  1996,  3029,  1029,   102,  1996,  2270,  2326,
        #           3222,  1006,  8827,  2278,  1007,  2056,  1999,  1037,  4861,  2008,
        #           1996,  3667,  1011,  1011,  2164, 11500,  1010,  3502,  7435,  1010,
        #          22294, 24384, 26727,  1010,  8205,  3738,  1998, 21767,  1011,  1011,
        #           2052,  2022, 15605,  2013,  5738,  2037, 16165,  2015,  2006,  6928,
        #           1012,   102]

        offset_mapping_batch = tokenized_batch["offset_mapping"].numpy().tolist()
        # print(offset_mapping_batch)
        # [[  0,   0], [  0,   4], [  5,   7], [  8,  11], [ 12,  24],
        assert len(offset_mapping_batch) == len(start_end_context_char_pos_list)

        # convert char boudaries to token boudaries in batch
        start_token_pos_list, end_token_pos_list = [], []
        for offset_mapping, start_end_context_char_pos in zip(offset_mapping_batch, start_end_context_char_pos_list):
            if start_end_context_char_pos == [0, 0]:
                start_token_pos_list.append(0)
                end_token_pos_list.append(0)
            else:
                (
                    start_token_pos,
                    end_token_pos,
                ) = self.convert_char_boudaries_to_token_boudaries(
                    offset_mapping=offset_mapping,
                    start_end_context_char_pos=start_end_context_char_pos,
                )
                start_token_pos_list.append(start_token_pos)
                end_token_pos_list.append(end_token_pos)

        tokenized_batch["start_positions"] = torch.LongTensor(start_token_pos_list)
        tokenized_batch["end_positions"] = torch.LongTensor(end_token_pos_list)

        # additional
        tokenized_batch["instances"] = batch
        return tokenized_batch

    # TODO: add tests
    @staticmethod
    def convert_char_boudaries_to_token_boudaries(offset_mapping, start_end_context_char_pos):
        start_context_char_pos, end_context_char_pos = start_end_context_char_pos
        assert end_context_char_pos >= start_context_char_pos

        done = False
        special_tokens_cnt = 0
        for i, token_boudaries in enumerate(offset_mapping):

            if token_boudaries == [0, 0]:
                special_tokens_cnt += 1
                continue

            if special_tokens_cnt == 2:
                start_token_pos, end_token_pos = token_boudaries

                if start_token_pos == start_context_char_pos:
                    res_start_token_pos = i

                if end_token_pos == end_context_char_pos:
                    res_end_token_pos = i  # inclusive
                    done = True
                    break
        assert done
        return res_start_token_pos, res_end_token_pos