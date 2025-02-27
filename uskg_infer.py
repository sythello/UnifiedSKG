from typing import Optional, Dict

import os
import sys
import json
import random
import numpy as np

import argparse
from argparse import ArgumentParser
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu

import time
import torch
import datasets
from transformers import (
    HfArgumentParser,
    set_seed,
    AutoTokenizer
)
from utils.configue import Configure
from utils.training_arguments import WrappedSeq2SeqTrainingArguments

from filelock import FileLock
import nltk
with FileLock(".lock") as lock:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)

from nltk.tokenize.treebank import TreebankWordDetokenizer

from copy import deepcopy

from seq2seq_construction import spider
from third_party.spider.preprocess.get_tables import dump_db_json_schema


import editdistance

from SpeakQL.Allennlp_models.utils.spider import process_sql, evaluation
from SpeakQL.Allennlp_models.utils.misc_utils import EvaluateSQL, EvaluateSQL_full, \
    Postprocess_rewrite_seq, Postprocess_rewrite_seq_freeze_POS, Postprocess_rewrite_seq_modify_POS



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', '--cfg', type=str, required=False, default='configure/Salesforce/T5_base_prefix_spider_with_cell_value.cfg',
                        help='The path to config file ("high-level" cfg file)')
    parser.add_argument('-model_name', '--model_name', type=str, required=False, default='hkunlp/from_all_T5_base_prefix_spider_with_cell_value2',
                        help='The model name on HuggingFace')
    parser.add_argument('-test_dataset_path', '--test_dataset_path', type=str, required=True,
                        help='The path to test dataset')
    parser.add_argument('-orig_dev_path', '--orig_dev_path', type=str, required=True,
                        help='The path to orig dev dataset')
    parser.add_argument('-db_path', '--db_path', type=str, required=True,
                        help='The path to db (e.g. .../spider/database)')
    parser.add_argument('-eval_vers', '--eval_vers', type=str, required=True, nargs='*',
    					help='The versions to evaluate')
    parser.add_argument('-eval_in_dir', '--eval_in_dir', type=str, required=True,
                        help='The directory for prediction files (i.e. Allennlp_models/outputs)')
    parser.add_argument('-eval_out_dir', '--eval_out_dir', type=str, required=False, default=None,
                        help='The directory for output files (dataset json with preds and eval scores, i.e. /vault/SpeakQL/.../uskg-test-save)')
    # parser.add_argument('-result_out_dir', '--result_out_dir', type=str, required=False, default=None,
    #                     help='The directory for output files (dataset json with preds and eval scores, i.e. /vault/SpeakQL/.../uskg-test-save)')

    args = parser.parse_args()
    return args

def play_pred(txt, model, tokenizer):
    tokenized_txt = tokenizer([txt], max_length=1024, padding="max_length", truncation=True)
    pred = tokenizer.batch_decode(
      model.generate(
        torch.LongTensor(tokenized_txt.data['input_ids']),
        torch.LongTensor(tokenized_txt.data['attention_mask']),
        num_beams=1, 
        max_length=256
        ), 
      skip_special_tokens=True 
    )
    return pred

_SCHEMA_CACHE = dict()

def get_uskg_sample(sample, db_path):
    db_id = sample["db_id"]
    if db_id not in _SCHEMA_CACHE:
        _SCHEMA_CACHE[db_id] = dump_db_json_schema(
            db_path + "/" + db_id + "/" + db_id + ".sqlite", db_id)
    schema = _SCHEMA_CACHE[db_id]

    return {
        "query": sample["query"],
        "question": sample["question"],
        "db_id": db_id,
        "db_path": db_path,
        "db_table_names": schema["table_names_original"],
        "db_column_names": {
            "table_id": [table_id for table_id, column_name in schema["column_names_original"]],
            "column_name": [column_name for table_id, column_name in schema["column_names_original"]]
        },
        "db_column_types": schema["column_types"],
        "db_primary_keys": [{"column_id": column_id} for column_id in schema["primary_keys"]],
        "db_foreign_keys": [
            {"column_id": column_id, "other_column_id": other_column_id}
            for column_id, other_column_id in schema["foreign_keys"]
        ],
    }

def get_uskg_struct_in(uskg_sample):
    return spider.serialize_schema(
        question=uskg_sample["question"],
        db_path=uskg_sample["db_path"],
        db_id=uskg_sample["db_id"],
        db_column_names=uskg_sample["db_column_names"],
        db_table_names=uskg_sample["db_table_names"],
        schema_serialization_type="peteshaw",
        schema_serialization_randomized=False,
        schema_serialization_with_db_id=True,
        schema_serialization_with_db_content=True,
        normalize_query=True,
    )


def _Postprocess_rewrite_seq_wrapper(cand_dict, pred_dict):
    _tags = pred_dict['rewriter_tags']
    _rewrite_seq = pred_dict['rewrite_seq_prediction']
    _question_toks = cand_dict['question_toks']
    return Postprocess_rewrite_seq(_tags, _rewrite_seq, _question_toks)

def _detokenize(toks):
    detokenizer = TreebankWordDetokenizer()
    return detokenizer.detokenize(toks)

def Full_evaluate_ILM(model,
                      tokenizer,
                      eval_version,
                      rewriter_ILM_pred_path,
                      test_dataset_path,
                      orig_dev_path,
                      db_path,
                      test_output_path=None,
                      result_output_path=None,
                      ILM_rewrite_func=_Postprocess_rewrite_seq_wrapper):
    
    '''
    eval_version: simply for printing results 
    
    Example paths:
    rewriter_ILM_pred_path = '/Users/mac/Desktop/syt/Deep-Learning/Projects-M/SpeakQL/SpeakQL/Allennlp_models/outputs/output-{}.json'.format(VERSION)
    test_dataset_path = '/Users/mac/Desktop/syt/Deep-Learning/Dataset/spider/my/dev/test_rewriter.json'
    orig_dev_path = '/Users/mac/Desktop/syt/Deep-Learning/Dataset/spider/dev.json'
    
    ILM_rewrite_func: Callable, args: (_tags, _rewrite_seq, _question_toks)
    '''
    
    VERSION = eval_version
    
    with open(rewriter_ILM_pred_path, 'r') as f:
        rewriter_ILM_preds = [json.loads(l) for l in f.readlines()]
    with open(test_dataset_path, 'r') as f:
        test_dataset = json.load(f)
    with open(orig_dev_path, 'r') as f:
        orig_dev_dataset = json.load(f)

    # len(rewriter_ILM_preds), len(test_dataset), sum([len(d) for d in test_dataset]), len(orig_dev_dataset)

    ## Quick evaluation: only using the 1st ASR candidate

    pred_idx = 0

    ref_list = []
    hyp_list = []
    wer_numer = 0
    wer_denom = 0

    for d in tqdm(test_dataset, desc=f'VERSION {VERSION}'):
        if len(d) == 0:
            continue

        c = d[0]

        p = rewriter_ILM_preds[pred_idx]
        _o_idx = c['original_id']
        o = orig_dev_dataset[_o_idx]
        assert ' '.join(c['question_toks']).lower() == p['question'].lower(), (' '.join(c['question_toks']), p['question'])
        assert c['gold_question_toks'] == o['question_toks'], (c['gold_question_toks'], o['question_toks'])

        # Debug 
        # assert c['rewriter_tags'] == p['rewriter_tags'][:len(c['rewriter_tags'])], f"{c['rewriter_tags']}\n{p['rewriter_tags']}\nShould raise"

        _db_id = o['db_id']

        _tags = p['rewriter_tags']
        _rewrite_seq = p['rewrite_seq_prediction']
        _question_toks = c['question_toks']

        # _rewritten_question_toks = Postprocess_rewrite_seq(_tags, _rewrite_seq, _question_toks)
        _rewritten_question_toks = ILM_rewrite_func(c, p)
        # _rewritten_question = ' '.join(_rewritten_question_toks)
        _rewritten_question = _detokenize(_rewritten_question_toks)

        #_pred_sql = Question(_rewritten_question, _db_id, model_dict=model_dicts[rat_sql_run])[0]['inferred_code']
        
        _uskg_sample = get_uskg_sample(c, db_path=db_path)
        _uskg_sample['question'] = _rewritten_question
        _struct_in = get_uskg_struct_in(_uskg_sample)
        _pred_sql = play_pred("{}; structed knowledge: {}".format(_rewritten_question, _struct_in),
                              model, tokenizer)[0]

        _gold_sql = c['query']
        ## verbose=False since too many get_sql() errors...
        _exact, _score, _exec = EvaluateSQL(_pred_sql, _gold_sql, _db_id, verbose=False)

        # Save the taggerILM raw outputs, for later aggregation 
        c['pred_tags'] = p['rewriter_tags']
        c['pred_ILM'] = p['rewrite_seq_prediction']
        c['pred_ILM_cands'] = p['rewrite_seq_prediction_cands']

        # Save prediction results 
        c['rewritten_question'] = p['rewritten_question'] = _rewritten_question
        c['pred_sql'] = p['pred_sql'] = _pred_sql
        p['gold_sql'] = _gold_sql
        c['score'] = p['score'] = _score
        c['exact'] = p['exact'] = _exact
        c['exec'] = p['exec'] = _exec

        _rewritten_question_toks = [_t.lower() for _t in _rewritten_question_toks]
        _gold_question_toks = [_t.lower() for _t in c['gold_question_toks']]

        ref_list.append([_gold_question_toks])
        hyp_list.append(_rewritten_question_toks)
        wer_numer += editdistance.eval(_gold_question_toks, _rewritten_question_toks)
        wer_denom += len(_gold_question_toks)

        pred_idx += len(d)

    # Only using the 1st candidate to rewrite 
    _avg_1st = sum([d[0]['score'] for d in test_dataset]) / len(test_dataset)
    _avg_exact_1st = sum([d[0]['exact'] for d in test_dataset]) / len(test_dataset)
    _avg_exec_1st = sum([d[0]['exec'] for d in test_dataset]) / len(test_dataset)

    ## Std-dev (1st cand only)
    # _std_1st = np.std([d[0]['score'] for d in test_dataset])

    ## BLEU 
    _bleu = corpus_bleu(list_of_references=ref_list,
                        hypotheses=hyp_list)
    _wer = 1.0 * wer_numer / (wer_denom + 1e-9)


    out_msg = ''
    out_msg += '='*20 + f' VERSION: {VERSION} ' + '='*20 + '\n'
    out_msg += f'avg_exact = {_avg_exact_1st:.4f}' + '\n'
    out_msg += f'avg = {_avg_1st:.4f}' + '\n'
    out_msg += f'avg_exec = {_avg_exec_1st:.4f}' + '\n'
    out_msg += f'BLEU = {_bleu:.4f}' + '\n'
    out_msg += f'WER = {_wer:.4f}' + '\n'
    out_msg += '='*55 + '\n'
    
    if test_output_path is not None:
        with open(test_output_path, 'w') as f:
            json.dump(test_dataset, f, indent=4)
    
    if result_output_path is not None:
        _res_d = {
            "avg_exact": _avg_exact_1st,
            "avg": _avg_1st,
            "avg_exec": _avg_exec_1st,
            "BLEU": _bleu,
            "WER": _wer,
        }
        with open(result_output_path, 'w') as f:
            json.dump(_res_d, f, indent=4)

    print(out_msg)
    return out_msg


def main(args):
    print(f"Loading config {args.cfg}")
    model_args = Configure.Get(args.cfg)

    print(f"Loading model {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    from models.unified.prefixtuning import Model
    model = Model(model_args)
    model.load(args.model_name)
    print("done.")

    test_dataset_path = args.test_dataset_path
    orig_dev_path = args.orig_dev_path
    db_path = args.db_path

    out_msgs = []

    print(f"Starting evaluations: {args.eval_vers}")
    for ver in args.eval_vers:
        rewriter_ILM_pred_path = os.path.join(args.eval_in_dir, f'output-{ver}.json')
        if args.eval_out_dir is not None:
            test_output_path = os.path.join(args.eval_out_dir, f'{ver}.json')
            result_output_path = os.path.join(args.eval_out_dir, f'eval-{ver}.json')
        
        out_msg = Full_evaluate_ILM(model=model,
                          tokenizer=tokenizer,
                          eval_version=ver,
                          rewriter_ILM_pred_path=rewriter_ILM_pred_path,
                          test_dataset_path=test_dataset_path,
                          orig_dev_path=orig_dev_path,
                          db_path=db_path,
                          test_output_path=test_output_path,
                          result_output_path=result_output_path,
                          ILM_rewrite_func=_Postprocess_rewrite_seq_wrapper)
        out_msgs.append(out_msg)

    for out_msg in out_msgs:
        print(out_msg)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)




