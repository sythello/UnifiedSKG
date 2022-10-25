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

from sdra.probing_data_collect import _random_select_indices


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


if __name__ == '__main__':
	print(_random_select_indices_TEST())





