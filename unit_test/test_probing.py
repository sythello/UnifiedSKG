import sys
import os
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
# from models.unified.prefixtuning import Model
from models.unified import finetune, prefixtuning
from argparse import ArgumentParser

import json
from copy import deepcopy
from collections import Counter, defaultdict
import importlib
import pickle
import random
import types

from seq2seq_construction import spider
from third_party.spider.preprocess.get_tables import dump_db_json_schema

import numpy as np
from tqdm import tqdm
import editdistance
import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize.treebank import TreebankWordDetokenizer

import re
from copy import deepcopy
from typing import List, Dict

from datasets.dataset_dict import DatasetDict
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from third_party.miscs.bridge_content_encoder import get_database_matches

from language.xsp.data_preprocessing import spider_preprocessing, wikisql_preprocessing, michigan_preprocessing

from sdr_analysis.helpers.general_helpers import _random_select_indices, load_pickle_list
from sdra import probing_data_utils as pb_utils
# from sdra.probing_data_collect import _random_select_indices, load_model_and_tokenizer
from sdra.probing_data_utils import play_pred
# from sdra.link_prediction_collectors import collect_link_prediction_samples


def _random_select_indices_TEST():
    orig_len = 12
    k = 5
    train_ids = _random_select_indices(orig_len=orig_len, k=k, ds='train')
    test_ids = _random_select_indices(orig_len=orig_len, k=k, ds='test')

    # do sth else 
    _s = 0
    for i in range(10):
        _s += random.choice(range(i+1))

    test_ids_2 = _random_select_indices(orig_len=orig_len, k=k, ds='test')
    train_ids_2 = _random_select_indices(orig_len=orig_len, k=k, ds='train')

    try:
        assert train_ids == train_ids_2, (train_ids, train_ids_2)
        assert test_ids == test_ids_2, (test_ids, test_ids_2)
        corr = True
    except AssertionError as e:
        print(e)
        corr = False

    return corr


def tokenizer_TEST():
    ## Adapted from load_model_and_tokenizer()
    ## the code is stale. TODO: make this runnable
    save_argv = sys.argv

    # Set args here for runnning on notebook, we make them out here to make it more illustrative.
    sys.argv = ['/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py', # This is the name of your .py launcher when you run this line of code.
                # belows are the parameters we set, take spider for example
                '--cfg', 'Salesforce/T5_large_prefix_spider_with_cell_value.cfg', 
                '--output_dir', './tmp']
    parser = HfArgumentParser((WrappedSeq2SeqTrainingArguments,))
    training_args, = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)
    args = Configure.Get(training_args.cfg)

    sys.argv = save_argv

    # TODO: add model_path to args 
    model_path = 'hkunlp/from_all_T5_large_prefix_spider_with_cell_value2'
    # model_path = 'hkunlp/from_all_T5_large_prefix_spider_with_cell_value2'
    # model_path = '/Users/mac/Desktop/syt/Deep-Learning/Repos/UnifiedSKG/output/server_runs/A-T5_base_prefix_spider_with_cell_value-asr_mixed/checkpoint-79500/'
    # model_path = '/Users/mac/Desktop/syt/Deep-Learning/Repos/UnifiedSKG/output/server_runs/A-T5_base_prefix_spider_with_cell_value-rewritten_mixed/checkpoint-56500/'

    # tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = Model(args)
    model.load(model_path)

    # for word/token/char mapping functions
    # tokenizer_uskg = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    tokenizer_base = AutoTokenizer.from_pretrained('t5-base', use_fast=True)
    tokenizer_large = AutoTokenizer.from_pretrained('t5-large', use_fast=True)

    uskg_schemas_dict = pb_utils.precompute_spider_uskg_schemas_dict(
        orig_tables_path='/home/yshao/Projects/language/language/xsp/data/spider/tables.json',
        db_dir='/home/yshao/Projects/language/language/xsp/data/spider/database')

    dataset_path = os.path.join('/home/yshao/Projects/SDR-analysis/data/spider/dev+ratsql_graph.json')
    with open(dataset_path, 'r') as f:
        orig_dataset = json.load(f)
    for d in orig_dataset:
        d['rat_sql_graph']['relations'] = json.loads(d['rat_sql_graph']['relations'])

    for sid, sample in enumerate(tqdm(orig_dataset, ascii=True)):
        # enc_uskg = get_USKG_node_encodings(sample, model, tokenizer_uskg, uskg_schemas_dict, tokenizer_args=None, pooling_func=None, debug=False)
        enc_base = get_USKG_node_encodings(sample, model, tokenizer_base, uskg_schemas_dict, tokenizer_args=None, pooling_func=None, debug=False)
        enc_large = get_USKG_node_encodings(sample, model, tokenizer_large, uskg_schemas_dict, tokenizer_args=None, pooling_func=None, debug=False)

        if np.allclose(enc_base, enc_large, atol=1e-3):
            pass
        else:
            enc_diff = np.max(enc_base, enc_large) - np.min(enc_base, enc_large) # (seq_len, enc_dim)
            enc_diff = np.amax(enc_diff, axis=-1)   # (seq_len,)
            max_diff_pos = np.argmax(enc_diff)
            print(sid, max_diff_pos, enc_diff[max_diff_pos])
            print(enc_uskg[max_diff_pos][::300], enc_base[max_diff_pos][::300], enc_large[max_diff_pos][::300])
            print()

        # if np.allclose(enc_uskg, enc_base, atol=1e-3) and np.allclose(enc_uskg, enc_large, atol=1e-3):
        #     pass
        # else:
        #     enc_diff = np.amax([enc_uskg, enc_base, enc_large], axis=0) - np.amin([enc_uskg, enc_base, enc_large], axis=0) # (seq_len, enc_dim)
        #     enc_diff = np.amax(enc_diff, axis=-1)   # (seq_len,)
        #     max_diff_pos = np.argmax(enc_diff)
        #     print(sid, max_diff_pos, enc_diff[max_diff_pos])
        #     print(enc_uskg[max_diff_pos][::300], enc_base[max_diff_pos][::300], enc_large[max_diff_pos][::300])
        #     print()


def model_TEST():
    ## the code is stale. TODO: make this runnable
    struct_in = "| concert_singer | stadium : stadium_id , location , name , capacity , highest , lowest , average | singer : singer_id , name , country ( France ) , song_name , song_release_year , age , is_male | concert : concert_id , concert_name , theme , stadium_id , year | singer_in_concert : concert_id , singer_id"
    text_in = "what is the minimum, average, and maximum age of all singers from France?"

    main_args = types.SimpleNamespace()

    # Model1: USKG (T5-base-prefix)
    main_args.model_path = 'hkunlp/from_all_T5_base_prefix_spider_with_cell_value2'
    main_args.uskg_config = 'Salesforce/T5_base_prefix_spider_with_cell_value.cfg'
    model, tokenizer = load_model_and_tokenizer(main_args)

    sql_pred_1 = play_pred("{}; structed knowledge: {}".format(text_in, struct_in), model, tokenizer)

    # Model2: T5-base
    main_args.model_path = 't5-base'
    main_args.uskg_config = 'Salesforce/T5_base_finetune_spider_with_cell_value.cfg'
    model, tokenizer = load_model_and_tokenizer(main_args)

    sql_pred_2 = play_pred("{}; structed knowledge: {}".format(text_in, struct_in), model, tokenizer)

    # Model3: T5-base-random
    main_args.model_path = 't5-base-rd'
    main_args.uskg_config = 'Salesforce/T5_base_finetune_spider_with_cell_value.cfg'
    model, tokenizer = load_model_and_tokenizer(main_args)

    sql_pred_3 = play_pred("{}; structed knowledge: {}".format(text_in, struct_in), model, tokenizer)

    print('USKG (T5-base-prefix):', sql_pred_1)
    print('T5-base:', sql_pred_2)
    print('T5-base-random:', sql_pred_3)


def data_identical_TEST():
    ds1_path = '/home/yshao/Projects/SDR-analysis/data/probing/text2sql/link_prediction/spider/uskg/dev.test.X.pkl'
    ds2_path = '/home/yshao/Projects/SDR-analysis/data/probing/text2sql/link_prediction/spider/uskg-tmp/dev.test.X.pkl'

    # with open(ds1_path, 'rb') as f:
    #     ds1 = pickle.load(f)
    # with open(ds2_path, 'rb') as f:
    #     ds2 = pickle.load(f)
    ds1 = load_pickle_list(ds1_path)
    ds2 = load_pickle_list(ds2_path)
    
    print(f'Load dataset size: {len(ds1)}, {len(ds2)}')

    mismatch_list = []
    for i, (x1, x2) in enumerate(zip(ds1, ds2)):
        if not np.allclose(x1, x2, atol=0.001):
            mismatch_list.append((i, x1.reshape(-1)[:3], x2.reshape(-1)[:3]))
    
    if not mismatch_list:
        print('All match!')
    else:
        print(mismatch_list[:10])
        if len(mismatch_list) > 10:
            print('...')
        print(f'{len(mismatch_list)} mismatch')


if __name__ == '__main__':
    # print(_random_select_indices_TEST())
    # model_TEST()
    data_identical_TEST()





