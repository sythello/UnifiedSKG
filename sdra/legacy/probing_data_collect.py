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

import nltk

import json
from copy import deepcopy
from collections import Counter, defaultdict
import importlib
import pickle
import random

from seq2seq_construction import spider
from third_party.spider.preprocess.get_tables import dump_db_json_schema

import numpy as np
from tqdm import tqdm
import editdistance
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

from sdra.legacy import probing_data_utils as pb_utils


def _random_select_indices(orig_len, k, ds, seed=42):
    """
    Args:
        k: how many to select
        ds: the output dataset split (train/test)
    """

    assert orig_len >= k * 2, (orig_len, k)
    assert ds in ['train', 'test'], ds

    # Always generate 2*k with the same seed, so train/test are consistent and won't overlap
    random.seed(seed)
    sel_ids_both = random.sample(range(orig_len), k=k*2)

    if ds == 'train':
        sel_ids = sel_ids_both[:k]
    elif ds == 'test':
        sel_ids = sel_ids_both[k:]
    else:
        raise ValueError(ds)

    return sel_ids


def load_model_and_tokenizer(main_args):
    save_argv = sys.argv

    # Set args here for runnning on notebook, we make them out here to make it more illustrative.
    sys.argv = ['/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py', # This is the name of your .py launcher when you run this line of code.
                # belows are the parameters we set, take spider for example
                # '--cfg', 'Salesforce/T5_large_prefix_spider_with_cell_value.cfg', 
                '--cfg', main_args.uskg_config,
                '--output_dir', './tmp']
    parser = HfArgumentParser((WrappedSeq2SeqTrainingArguments,))
    training_args, = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)
    model_args = Configure.Get(training_args.cfg)

    sys.argv = save_argv

    ## Tokenizer: 'fast' for word/token/char mapping functions
    # tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    print('Using tokenizer:', model_args.bert.location)
    tokenizer_fast = AutoTokenizer.from_pretrained(model_args.bert.location, use_fast=True)

    ## Model: for model_path, now support: USKG (hkunlp/xxx); original T5 (t5-xxx); random T5 (t5-xxx-rd)
    model_path = main_args.model_path
    if model_path.startswith('hkunlp'):
        ## USKG
        if 'prefix' in model_path:
            assert 'prefix' in main_args.uskg_config, ('Mismatch', model_path, uskg_config)
            model = prefixtuning.Model(model_args)
        elif 'finetune' in model_path:
            assert 'finetune' in main_args.uskg_config, ('Mismatch', model_path, uskg_config)
            model = finetune.Model(model_args)
        else:
            raise ValueError(model_path)

        model.load(model_path)

    elif model_path.startswith('t5'):
        model = finetune.Model(model_args)
        assert model_path.startswith(model_args.bert.location), ('Mismatch', model_path, model_args.bert.location)  # check USKG & T5 version consistency
        if model_path.endswith('rd'):
            ## random T5
            model.pretrain_model.init_weights()
        else:
            ## original T5, already loaded
            pass
    else:
        raise ValueError(model_path)

    return model, tokenizer_fast


def collect_probing_dataset(args,
        model,
        tokenizer_fast,
        uskg_schemas_dict,
        orig_dataset,
        orig_ds,
        prob_ds,
        **extra_kwargs):
    """
    The main process of data collection (for a single orig_ds and prob_ds, e.g. for dev.test)
    """

    if args.probing_data_in_dir is not None:
        pos_file_path = os.path.join(args.probing_data_in_dir, f'{orig_ds}.{prob_ds}.pos.txt')
    else:
        pos_file_path = None

    if pos_file_path is not None:
        with open(pos_file_path, 'r') as f:
            lines = f.read().strip().split('\n')
            all_pos_triplets = [tuple([int(s) for s in l.split('\t')]) for l in lines]
        # len(all_pos_triplets), all_pos_triplets[0]

        sample_ds_indices = []               # [ds_idx], based on occurring order 
        pos_per_sample = defaultdict(list)   # key = ds_idx, value = pos_list: List[(i, j)]

        for ds_idx, i, j in all_pos_triplets:
            if not sample_ds_indices or sample_ds_indices[-1] != ds_idx:
                sample_ds_indices.append(ds_idx)
            pos_per_sample[ds_idx].append((i, j))
        # len(sample_ds_indices), len(pos_per_sample)
        print(f'Loaded pos file from {pos_file_path}: {len(all_pos_triplets)} triplets, {len(sample_ds_indices)} orig samples')
    else:
        pos_per_sample = defaultdict(lambda: None)
        sample_ds_indices = _random_select_indices(orig_len=len(orig_dataset), k=args.ds_size, ds=prob_ds, seed=42)
        print(f'Generated pos: {len(sample_ds_indices)} orig samples')


    all_X = []
    all_y = []
    all_pos = []

    for sample_ds_idx in tqdm(sample_ds_indices, ascii=True, desc=f'{orig_ds}.{prob_ds}'):
        dataset_sample = orig_dataset[sample_ds_idx]
        pos_list = pos_per_sample[sample_ds_idx]

        try:
            X, y, pos = pb_utils.extract_probing_samples_link_prediction_uskg(
                dataset_sample=dataset_sample,
                db_schemas_dict=uskg_schemas_dict,
                model=model,
                tokenizer=tokenizer_fast,
                pos=pos_list,
                max_rel_occ=args.max_occ,  # when given pos, this is not needed 
                debug=False)
            
            all_X.extend(X)
            all_y.extend(y)
            pos = [(sample_ds_idx, i, j) for i, j in pos]   # add sample idx 
            all_pos.extend(pos)
        except TypeError as e:
            # input too long will cause this error
            print(f'WARNING: sample {sample_ds_idx} txt too long, skipped')
            continue
        except Exception as e:
            raise e

        # time.sleep(0.2)
    # len(all_X), len(all_y), len(all_pos)

    # probing_data_out_dir = "/Users/mac/Desktop/syt/Deep-Learning/Projects-M/SDR-analysis/data/probing/text2sql/link_prediction/spider/uskg"
    probing_data_out_dir = args.probing_data_out_dir
    os.makedirs(probing_data_out_dir, exist_ok=True)

    output_path_test_X = os.path.join(probing_data_out_dir, f'{orig_ds}.{prob_ds}.X.pkl')
    output_path_test_y = os.path.join(probing_data_out_dir, f'{orig_ds}.{prob_ds}.y.pkl')
    output_path_test_pos = os.path.join(probing_data_out_dir, f'{orig_ds}.{prob_ds}.pos.txt')

    with open(output_path_test_X, 'wb') as f:
        pickle.dump(all_X, f)
    with open(output_path_test_y, 'wb') as f:
        pickle.dump(all_y, f)
    with open(output_path_test_pos, 'w') as f:
        for idx, i, j in all_pos:
            f.write(f'{idx}\t{i}\t{j}\n')



def main(args):
    model, tokenizer_fast = load_model_and_tokenizer(args)

    # probing_data_dir = "/Users/mac/Desktop/syt/Deep-Learning/Projects-M/SDR-analysis/data/probing/text2sql/link_prediction/spider/ratsql"
    # probing_data_dir = args.probing_data_dir

    # xsp_data_dir = "/Users/mac/Desktop/syt/Deep-Learning/Repos/Google-Research-Language/language/language/xsp/data"
    # spider_tables_path = os.path.join(xsp_data_dir, 'spider', 'tables.json')

    # TODO: extend to wikisql 
    uskg_schemas_dict = pb_utils.precompute_spider_uskg_schemas_dict(
        orig_tables_path=args.in_tables_path,
        db_dir=args.input_database_dir)

    kwargs = {
        'model': model,
        'tokenizer_fast': tokenizer_fast,
        'uskg_schemas_dict': uskg_schemas_dict,
    }

    orig_ds_list = ['train', 'dev']
    prob_ds_list = ['train', 'test']
    for orig_ds in orig_ds_list:
        # dataset_path = f"/Users/mac/Desktop/syt/Deep-Learning/Projects-M/SDR-analysis/data/spider/{orig_ds}+ratsql_graph.json"
        dataset_path = os.path.join(args.input_spider_dir, f'{orig_ds}+ratsql_graph.json')
        with open(dataset_path, 'r') as f:
            orig_dataset = json.load(f)
        for d in orig_dataset:
            d['rat_sql_graph']['relations'] = json.loads(d['rat_sql_graph']['relations'])

        kwargs['orig_ds'] = orig_ds
        kwargs['orig_dataset'] = orig_dataset

        for prob_ds in prob_ds_list:
            kwargs['prob_ds'] = prob_ds

            collect_probing_dataset(args, **kwargs)



if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('-in_spider_dir', '--input_spider_dir', type=str, required=True,
        help="Dir with input spider dataset files (xxx+rat_sql_graph.json)")
    parser.add_argument('-in_tables', '--in_tables_path', type=str, required=True,
        help="Input spider tables file (tables.json)")
    parser.add_argument('-in_dbs_dir', '--input_database_dir', type=str, required=True,
        help="Dir with spider databases")
    parser.add_argument('-pb_in_dir', '--probing_data_in_dir', type=str, required=False,
        help="The directory with input probing data files (e.g. from rat-sql)")
    parser.add_argument('-sz', '--ds_size', type=int, required=False, default=500,
        help="Only used when no 'pb_in_dir' given. Use X samples from original dataset to collect probing samples.")
    parser.add_argument('-mo', '--max_occ', type=int, required=False, default=1,
        help="Only used when no 'pb_in_dir' given. For each spider sample, include at most X probing samples per relation type.")

    parser.add_argument('-model', '--model_path', type=str, required=False, default='hkunlp/from_all_T5_large_prefix_spider_with_cell_value2',
        help="The (T5) model to use. Now support: USKG (hkunlp/xxx); original T5 (t5-xxx); random T5 (t5-xxx-rd)")
    parser.add_argument('-cfg', '--uskg_config', type=str, required=False, default='Salesforce/T5_large_prefix_spider_with_cell_value.cfg',
        help="The USKG config file (Salesforce/xxx) to use.")
    
    parser.add_argument('-pb_out_dir', '--probing_data_out_dir', type=str, required=True,
        help="The directory to have output probing data files (for uskg)")

    args = parser.parse_args()

    main(args)





