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
    def extract_probing_samples_link_prediction(self, dataset_sample, pos=None):
        """
        Args:
            dataset_sample (Dict): a sample dict from spider dataset
            pos (List[Tuple]): the position pairs to use. If none, will randomly generate        
        Return:
            X (List[np.array]): input features, "shape" = (n, dim)
            y (List): output labels, "shape" = (n,)
            pos (List[Tuple]): actual position (node-id) pairs for X and y
        """

        # TODO (later): add a batched version

        d = dataset_sample

        # db_id = d['db_id']
        # db_schema = db_schemas_dict[db_id]
        # question = d['question']

        # get relation matrix (relation_id2name not available as it needs rat-sql model)
        graph_dict = d['rat_sql_graph']
        # graph_dict['relation_id2name'] = {v : k for k, v in model.encoder.encs_update.relation_ids.items()}

        # get encodings
        # rat_sql_encoder_state = get_rat_sql_encoder_state(question=question, db_schema=db_schema, model=model)
        # enc_repr = rat_sql_encoder_state.memory.squeeze(0).detach().cpu().numpy()
        enc_repr = self.get_node_encodings([d])[0][0]      # return: all_enc_repr, valid_in_batch_ids

        X, y, pos = collect_link_prediction_samples(
            graph_dict,
            enc_repr,
            pos=pos,
            max_rel_occ=self.max_label_occ,
            debug=self.debug)

        return X, y, pos
    

class LinkPredictionDataCollector_USKG_spider(LinkPredictionDataCollector_USKG, BaseGraphDataCollector_USKG_spider):
    pass


class LinkPredictionDataCollector_USKG_wikisql(LinkPredictionDataCollector_USKG, BaseGraphDataCollector_USKG_wikisql):
    pass

