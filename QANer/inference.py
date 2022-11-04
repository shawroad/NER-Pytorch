"""
@file   : inference.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-11-04
"""
import random
import json
import torch
import numpy as np
from model import Model
from config import set_infer_args
from transformers import AutoTokenizer
from data_helper import Span, Instance


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


def get_top_valid_spans(context_list, question_list, prompt_mapper, inputs, outputs, offset_mapping_batch,
                        n_best_size, max_answer_length):

    batch_size = len(offset_mapping_batch)
    inv_prompt_mapper = {v: k for k, v in prompt_mapper.items()}
    # print(inv_prompt_mapper)
    # {'location': 'LOC', 'person': 'PER', 'organization': 'ORG', 'miscellaneous entity': 'MISC'}

    assert batch_size == len(context_list)
    assert batch_size == len(question_list)
    assert batch_size == len(inputs["input_ids"])
    assert batch_size == len(inputs["token_type_ids"])
    assert batch_size == len(outputs["start_logits"])
    assert batch_size == len(outputs["end_logits"])

    top_valid_spans_batch = []

    # TODO: optimize it
    for i in range(batch_size):
        context = context_list[i]
        # print(context)    # 文本

        offset_mapping = offset_mapping_batch[i].cpu().numpy()
        mask = inputs["token_type_ids"][i].bool().cpu().numpy()
        # print(mask)   # [0, 0, ..., 1, 1, 1]
        offset_mapping[~mask] = [0, 0]
        # print(offset_mapping)   # 将问题部分的offset_mapping全部置为[0,0]

        offset_mapping = [
            (span if span != [0, 0] else None) for span in offset_mapping.tolist()
        ]
        # print(offset_mapping)   # 相当于这里取出了文章的所有的offset_mapping
        start_logits = outputs["start_logits"][i].cpu().numpy()
        end_logits = outputs["end_logits"][i].cpu().numpy()

        start_indexes = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
        end_indexes = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()
        # 找出概率最高的开始索引   以及  概率最高的结束索引

        top_valid_spans = []
        for start_index, end_index in zip(start_indexes, end_indexes):
            if (
                start_index >= len(offset_mapping)
                or end_index >= len(offset_mapping)
                or offset_mapping[start_index] is None
                or offset_mapping[end_index] is None
            ):
                continue

            if (end_index < start_index) or (
                end_index - start_index + 1 > max_answer_length
            ):
                continue
            if start_index <= end_index:
                start_context_char_char, end_context_char_char = offset_mapping[
                    start_index
                ]
                span = Span(
                    token=context[start_context_char_char:end_context_char_char],
                    label=inv_prompt_mapper[  # TODO: add inference exception
                        question_list[i].split(r"What is the ")[-1].rstrip(r"?")
                    ],
                    start_context_char_pos=start_context_char_char,
                    end_context_char_pos=end_context_char_char,
                )
                # print(span)
                # Span(token='in', label='ORG', start_context_char_pos=43, end_context_char_pos=45)

                top_valid_spans.append(span)
        top_valid_spans_batch.append(top_valid_spans)
    return top_valid_spans_batch


def predict(context, question, prompt_mapper, model, tokenizer, tokenizer_kwargs, n_best_size, max_answer_length):
    inputs = tokenizer([question], [context], **tokenizer_kwargs).to(model.device)
    offset_mapping_batch = inputs.pop("offset_mapping")

    with torch.no_grad():
        outputs = model(**inputs)

    spans_pred_batch_top = get_top_valid_spans(
        context_list=[context],
        question_list=[question],
        prompt_mapper=prompt_mapper,
        inputs=inputs,
        outputs=outputs,
        offset_mapping_batch=offset_mapping_batch,
        n_best_size=n_best_size,
        max_answer_length=max_answer_length,
    )[0]

    spans_pred_batch_top = [span for span in spans_pred_batch_top if span]

    for predicted_answer_span in spans_pred_batch_top:
        start_pos = predicted_answer_span.start_context_char_pos
        end_pos = predicted_answer_span.end_context_char_pos
        assert predicted_answer_span.token == context[start_pos:end_pos]

    prediction = Instance(
        context=context,
        question=question,
        answer=spans_pred_batch_top,
    )
    return prediction


if __name__ == '__main__':
    args = set_infer_args()
    set_seed(args.seed)

    # bert model
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)

    model = Model()
    model.load_state_dict(torch.load(args.save_model_path))
    if torch.cuda.is_available():
        model.cuda()

    tokenizer_kwargs = {
        "max_length": 512,
        "truncation": "only_second",
        "padding": True,
        "return_tensors": "pt",
        "return_offsets_mapping": True,
    }

    with open(args.path_to_prompt_mapper, mode="r", encoding="utf-8") as fp:
        prompt_mapper = json.load(fp)

    prediction = predict(
        context=args.context,
        question=args.question,
        prompt_mapper=prompt_mapper,
        model=model,
        tokenizer=tokenizer,
        tokenizer_kwargs=tokenizer_kwargs,
        n_best_size=args.n_best_size,
        max_answer_length=args.max_answer_length,
    )
    print(f"\nquestion: {prediction.question}\n")
    print(f"context: {prediction.context}")
    print(f"\nanswer: {prediction.answer}\n")


