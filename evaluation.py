"""
Adapted and changed from the SQuAD v1.1 official evaluation script.
Usage:
    python official_eval.py dataset_file.jsonl.gz prediction_file.json
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import gzip
import json
import re
import string
import torch
import numpy as np
from torch.utils.data import DataLoader,TensorDataset,SequentialSampler
from collections import Counter
import collections
from iterator import read_squad_examples,convert_examples_to_features,write_predictions


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def read_predictions(prediction_file):
    with open(prediction_file) as f:
        predictions = json.load(f)
    return predictions


def read_answers(gold_file):
    answers = {}
    with gzip.open(gold_file, 'rb') as f:
        for i, line in enumerate(f):

            example = json.loads(line)
            if i == 0 and 'header' in example:
                continue
            for qa in example['qas']:
                answers[qa['qid']] = qa['answers']

    return answers


def evaluate(answers, predictions, skip_no_answer=False):
    f1 = exact_match = total = 0
    for qid, ground_truths in answers.items():
        if qid not in predictions:
            if not skip_no_answer:
                message = 'Unanswered question %s will receive score 0.' % qid
                print(message)
                total += 1
            continue
        total += 1
        prediction = predictions[qid]
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(
            f1_score, prediction, ground_truths)
    if total>0:
        exact_match = 100.0 * exact_match / total
        f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}

def run_eval(model,tokenizer, file_path, prediction_file,is_adv, args):
    #for dann

    eval_examples = read_squad_examples(file_path, debug=False)

    eval_features = convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=args.data_args.max_seq_length,
        max_query_length=args.data_args.max_ques_length,
        doc_stride=128,
        is_training=False
    )
    all_input_ids = torch.tensor(np.array([f.input_ids for f in eval_features]), dtype=torch.long)
    all_input_mask = torch.tensor(np.array([f.input_mask for f in eval_features]), dtype=torch.long)
    all_segment_ids = torch.tensor(np.array([f.segment_ids for f in eval_features]), dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
    sampler = SequentialSampler(eval_data)
    eval_loader = DataLoader(eval_data, sampler=sampler, batch_size=args.data_args.batch_size)

    RawResult = collections.namedtuple("RawResult",
                                       ["unique_id", "start_logits", "end_logits"])

    model.eval()
    all_results = []
    example_index = -1
    for _, batch in enumerate(eval_loader):
        input_ids, input_mask, token_type_ids = batch
        seq_len = torch.sum(torch.sign(input_ids), 1)
        max_len = torch.max(seq_len)
        input_ids = input_ids[:, :max_len].clone()
        input_mask = input_mask[:, :max_len].clone()
        seq_ids = token_type_ids[:, :max_len].clone()

        if args.use_cuda:
            input_ids = input_ids.cuda(None, non_blocking=True)
            input_mask = input_mask.cuda(None, non_blocking=True)
            seq_ids = token_type_ids.cuda(None, non_blocking=True)

        with torch.no_grad():
            if is_adv:
                qa_output, _ = model.forward(input_ids = input_ids, attention_mask=input_mask)
            else:
                qa_output = model.forward(input_ids = input_ids, attention_mask=input_mask)
            batch_start_logits, batch_end_logits = qa_output.start_logits, qa_output.end_logits
            batch_size = batch_start_logits.size(0)
        for i in range(batch_size):
            example_index += 1
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index]
            unique_id = int(eval_feature.unique_id)
            all_results.append(RawResult(unique_id=unique_id,
                                         start_logits=start_logits,
                                         end_logits=end_logits))

    preds = write_predictions(eval_examples, eval_features, all_results,
                              n_best_size=10, max_answer_length=30, do_lower_case=True,
                              output_prediction_file=prediction_file)

    answers = read_answers(file_path)
    preds_dict = json.loads(preds)
    metrics = evaluate(answers, preds_dict)

    return metrics
