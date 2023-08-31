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

from sdr_analysis.helpers import general_helpers
from sdr_analysis.helpers.general_helpers import SDRASampleError
from sdr_analysis.helpers.base_graph_data_collector import BaseGraphDataCollector, BaseGraphDataCollector_spider, BaseGraphDataCollector_wikisql
from sdr_analysis.helpers.link_prediction_collector import LinkPredictionDataCollector
# from sdr_analysis.helpers.general_helpers import db_dict_to_general_fmt, collect_link_prediction_samples, LinkPredictionDataCollector



RAT_SQL_RELATION_ID2NAME = {
    0: ('qq_dist', -2),
    1: ('qq_dist', -1),
    2: ('qq_dist', 0),
    3: ('qq_dist', 1),
    4: ('qq_dist', 2),
    5: 'qc_default',
    6: 'qt_default',
    7: 'cq_default',
    8: 'cc_default',
    9: 'cc_foreign_key_forward',
    10: 'cc_foreign_key_backward',
    11: 'cc_table_match',
    12: ('cc_dist', -2),
    13: ('cc_dist', -1),
    14: ('cc_dist', 0),
    15: ('cc_dist', 1),
    16: ('cc_dist', 2),
    17: 'ct_default',
    18: 'ct_foreign_key',
    19: 'ct_primary_key',
    20: 'ct_table_match',
    21: 'ct_any_table',
    22: 'tq_default',
    23: 'tc_default',
    24: 'tc_primary_key',
    25: 'tc_table_match',
    26: 'tc_any_table',
    27: 'tc_foreign_key',
    28: 'tt_default',
    29: 'tt_foreign_key_forward',
    30: 'tt_foreign_key_backward',
    31: 'tt_foreign_key_both',
    32: ('tt_dist', -2),
    33: ('tt_dist', -1),
    34: ('tt_dist', 0),
    35: ('tt_dist', 1),
    36: ('tt_dist', 2),
    37: 'qcCEM',
    38: 'cqCEM',
    39: 'qtTEM',
    40: 'tqTEM',
    41: 'qcCPM',
    42: 'cqCPM',
    43: 'qtTPM',
    44: 'tqTPM',
    45: 'qcNUMBER',
    46: 'cqNUMBER',
    47: 'qcTIME',
    48: 'cqTIME',
    49: 'qcCELLMATCH',
    50: 'cqCELLMATCH'
}



class StructCharRangesCollector:
    def __init__(self):
        self.initialize()

    def initialize(self):
        self.db_id2char_ranges = dict()
        self.table2char_ranges = dict()
        self.column2char_ranges = dict()
        
        # Due to rat-sql stemming tokens, rat-sql nodes and uskg text may mismatch,
        # so we save a list and use the order instead of name for indexing 
        self.db_id_char_ranges_list = []
        self.column_char_ranges_list = []
        self.table_char_ranges_list = []

        self.bar_cnt = 0
        self.curr_table = None
        self.curr_node_type = None   # [None, 'db_id', 'table', 'column']
        self.curr_node_toks = []
        self.curr_node_char_start = None
        self.curr_node_char_end = None
        self.open_bracket = False
        
    def _register_curr_node(self):
        curr_node_name = ' '.join(self.curr_node_toks)
        curr_range = (self.curr_node_char_start, self.curr_node_char_end)

        if None in curr_range:
            print('- StructCharRangesCollector::_register_curr_node():')
            print(f'* WARNING: invalid char span: name = {curr_node_name}, span = {curr_range}')
            raise SDRASampleError('Invalid char span')
        
        if self.curr_node_type == 'db_id':
            self.db_id2char_ranges[curr_node_name] = curr_range
            self.db_id_char_ranges_list.append(curr_range)   
        elif self.curr_node_type == 'table':
            self.table2char_ranges[curr_node_name] = curr_range
            self.table_char_ranges_list.append(curr_range)
            self.curr_table = curr_node_name
        elif self.curr_node_type == 'column':
            self.column2char_ranges[(self.curr_table, curr_node_name)] = curr_range
            self.column_char_ranges_list.append(curr_range)
        else:
            raise ValueError(self.curr_node_type)

        self.curr_node_toks = []
        self.curr_node_char_start = None
        self.curr_node_char_end = None
    
    def collect(self, struct_in, tokenized_txt, _n_words_before_struct):
        # struct_words = struct_in.strip().split(' ')
        struct_words = struct_in.strip().split()
        
        for sw_id, sw in enumerate(struct_words):
            char_range = tokenized_txt.word_to_chars(sw_id + _n_words_before_struct)

            # print(sw_id, char_range, sw, self.curr_node_type, self.open_bracket)

            if sw == '(':
                self.open_bracket = True
                continue

            if sw == ')':
                self.open_bracket = False
                self.curr_node_char_end = char_range[1]
                continue

            if self.open_bracket:
                # in the list of cells, do not add tokens here to name 
                continue

            if sw == '|':
                if self.curr_node_type is not None:
                    self._register_curr_node()
                self.bar_cnt += 1
                if self.bar_cnt == 1:
                    self.curr_node_type = 'db_id'
                if self.bar_cnt > 1:
                    self.curr_node_type = 'table'
                continue

            if sw == ':':
                assert self.curr_node_type == 'table'
                self._register_curr_node()
                self.curr_node_type = 'column'
                continue

            if sw == ',':
                assert self.curr_node_type == 'column'
                self._register_curr_node()
                self.curr_node_type = 'column'
                continue

            self.curr_node_toks.append(sw)
            if self.curr_node_char_start is None:
                self.curr_node_char_start = char_range[0]
            self.curr_node_char_end = char_range[1]

        self._register_curr_node()



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

