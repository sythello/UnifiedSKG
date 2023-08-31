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
from sdr_analysis.helpers.single_node_reconstruction_collector import SingleNodeReconstructionDataCollector, collect_single_node_reconstruction_samples
from sdra.probing_data_utils import RAT_SQL_RELATION_ID2NAME, StructCharRangesCollector, play_pred
from sdra.probing_data_collectors import BaseGraphDataCollector_USKG, BaseGraphDataCollector_USKG_spider, BaseGraphDataCollector_USKG_wikisql


class SingleNodeReconstructionDataCollector_USKG(SingleNodeReconstructionDataCollector, BaseGraphDataCollector_USKG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_args['pooling']  = 'list'

    # Moved to LinkPredictionDataCollector
    # def extract_probing_samples_single_node_reconstruction(self, dataset_samples, pos_list=None):
    #     ...


class SingleNodeReconstructionDataCollector_USKG_spider(SingleNodeReconstructionDataCollector_USKG, BaseGraphDataCollector_USKG_spider):
    pass


class SingleNodeReconstructionDataCollector_USKG_wikisql(SingleNodeReconstructionDataCollector_USKG, BaseGraphDataCollector_USKG_wikisql):
    pass

