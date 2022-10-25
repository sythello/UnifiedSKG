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
from models.unified.prefixtuning import Model
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

from sdra import probing_data_utils as pb_utils


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



def main(args):
    # probing_data_dir = "/Users/mac/Desktop/syt/Deep-Learning/Projects-M/SDR-analysis/data/probing/text2sql/link_prediction/spider/ratsql"
    probing_data_dir = args.probing_data_dir

    orig_ds = 'dev'
    prob_ds = 'test'
    # dataset_path = f"/Users/mac/Desktop/syt/Deep-Learning/Projects-M/SDR-analysis/data/spider/{orig_ds}+ratsql_graph.json"
    dataset_path = os.path.join(args.input_spider_dir, f'{orig_ds}+ratsql_graph.json')

    # pos_file_path = os.path.join(probing_data_dir, f'{orig_ds}.{prob_ds}.pos.txt')
    pos_file_path = args.in_pos_path

    # xsp_data_dir = "/Users/mac/Desktop/syt/Deep-Learning/Repos/Google-Research-Language/language/language/xsp/data"
    # spider_tables_path = os.path.join(xsp_data_dir, 'spider', 'tables.json')
    spider_tables_path = args.in_tables_path

    uskg_schemas_dict = pb_utils.precompute_spider_uskg_schemas_dict(spider_tables_path)


    with open(dataset_path, 'r') as f:
        orig_dataset = json.load(f)
        
    for d in orig_dataset:
        d['rat_sql_graph']['relations'] = json.loads(d['rat_sql_graph']['relations'])

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
    else:
        pos_per_sample = defaultdict(lambda: None)

        # TODO: test this implementation!


        sample_ds_indices


    all_X = []
    all_y = []
    all_pos = []

    for sample_ds_idx in tqdm(sample_ds_indices):
        dataset_sample = orig_dataset[sample_ds_idx]
        pos_list = pos_per_sample[sample_ds_idx]

        X, y, pos = extract_probing_samples_link_prediction_uskg(dataset_sample=dataset_sample,
                                                                 db_schemas_dict=None,
                                                                 model=model,
                                                                 tokenizer=tokenizer_fast,
                                                                 pos=pos_list,
                                                                 max_rel_occ=None,  # when given pos, this is not needed 
                                                                 debug=False)
        
        all_X.extend(X)
        all_y.extend(y)
        pos = [(sample_ds_idx, i, j) for i, j in pos]   # add sample idx 
        all_pos.extend(pos)
        time.sleep(0.2)
    # len(all_X), len(all_y), len(all_pos)

    probing_data_out_dir = "/Users/mac/Desktop/syt/Deep-Learning/Projects-M/SDR-analysis/data/probing/text2sql/link_prediction/spider/uskg"
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






if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('-in_spider', '--input_spider_dir', type=str, required=True,
        desc="Dir with input spider dataset files (xxx+rat_sql_graph.json)")
    parser.add_argument('-in_tables', '--in_tables_path', type=str, required=True,
        desc="Input spider tables file (tables.json)")
    parser.add_argument('-in_pos', '--in_pos_path', type=str, required=False, default=None,
        desc="Input pos file (xxx.xxx.pos.txt), optional; if not given, will randomly generate")
    parser.add_argument('-mo', '--max_occ', type=int, required=False, default=1,
        desc="Only used when no 'in_pos' given. For each spider sample, include at most M probing samples per relation type.")
    parser.add_argument('-pb_dir', '--probing_data_dir', type=str, required=True,
        desc="The directory with probing data files")

    args = parser.parse_args()

    main(args)





