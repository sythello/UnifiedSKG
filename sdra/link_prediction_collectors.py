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
from sdr_analysis.helpers.link_prediction_collector import LinkPredictionDataCollector, collect_link_prediction_samples
from sdra.probing_data_utils import RAT_SQL_RELATION_ID2NAME, StructCharRangesCollector, play_pred
from sdra.probing_data_collectors import BaseGraphDataCollector_USKG, BaseGraphDataCollector_USKG_spider, BaseGraphDataCollector_USKG_wikisql


class LinkPredictionDataCollector_USKG(LinkPredictionDataCollector, BaseGraphDataCollector_USKG):
    def extract_probing_samples_link_prediction(self, dataset_samples, pos_list=None):
        """
        Args:
            dataset_samples (List[Dict]): a batch of sample dict from spider dataset
            pos_list (List[List[int]]): the positions (node-ids) to use. If none, will randomly generate
        Return:
            X (List[np.array]): input features, "shape" = (n, (dim,))
            y (List[int]): output labels, "shape" = (n,)
            pos (List[(int, int, int)]): actual positions (in_batch_idx, i, j)
        """

        # TODO (later): add a batched version

        all_enc_repr, valid_in_batch_ids = self.get_node_encodings(samples=dataset_samples, pooling_func=None)      # default pooling is avg

        all_X = []
        all_y = []
        all_pos = []

        for in_batch_idx, enc_repr in zip(valid_in_batch_ids, all_enc_repr):
            d = dataset_samples[in_batch_idx]

            # get relation matrix (relation_id2name not available as it needs rat-sql model)
            graph_dict = d['rat_sql_graph']
            sample_pos = None if pos_list is None else pos_list[in_batch_idx]

            # If sample_pos is not given (None), will do sampling and return in out_pos;
            # otherwise, out_pos is identical to sample_pos
            X, y, out_pos = collect_link_prediction_samples(
                graph_dict,
                enc_repr,
                pos=sample_pos,
                max_rel_occ=self.max_label_occ,
                debug=self.debug)

            all_X.extend(X)
            all_y.extend(y)
            all_pos.extend([(in_batch_idx, i, j) for i, j in out_pos])

        # with open(output_path_test_X, 'wb') as f:
        #     pickle.dump(all_X, f)
        # with open(output_path_test_y, 'w') as f:
        #     for y_toks in all_y:
        #         f.write(' '.join(y_toks) + '\n')
        # with open(output_path_test_pos, 'w') as f:
        #     for ds_idx, node_idx in all_pos:
        #         f.write(f'{ds_idx}\t{node_idx}\n')

        return all_X, all_y, all_pos
    

class LinkPredictionDataCollector_USKG_spider(LinkPredictionDataCollector_USKG, BaseGraphDataCollector_USKG_spider):
    pass


class LinkPredictionDataCollector_USKG_wikisql(LinkPredictionDataCollector_USKG, BaseGraphDataCollector_USKG_wikisql):
    pass

