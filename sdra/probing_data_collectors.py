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
from sdr_analysis.helpers.general_helpers import SDRASampleError, _wikisql_db_id_to_table_name, POOLING_FUNC_DICT
from sdr_analysis.helpers.base_graph_data_collector import BaseGraphDataCollector, BaseGraphDataCollector_spider, BaseGraphDataCollector_wikisql
from sdr_analysis.helpers.link_prediction_collector import LinkPredictionDataCollector
# from sdr_analysis.helpers.general_helpers import db_dict_to_general_fmt, collect_link_prediction_samples, LinkPredictionDataCollector
from sdra.probing_data_utils import RAT_SQL_RELATION_ID2NAME, StructCharRangesCollector, play_pred


class BaseGraphDataCollector_USKG(BaseGraphDataCollector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_args['need_loading_schemas'] = True

    def load_model(self, main_args):
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
                assert 'prefix' in main_args.uskg_config, ('Mismatch', model_path, main_args.uskg_config)
                model = prefixtuning.Model(model_args)
            elif 'finetune' in model_path:
                assert 'finetune' in main_args.uskg_config, ('Mismatch', model_path, main_args.uskg_config)
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
            
        model.to(self.device_name)

        self.model = model
        self.tokenizer_fast = tokenizer_fast

        return model, tokenizer_fast

    def general_fmt_dict_to_schema(self, general_fmt_dict):
        """
        Args:
            general_fmt_dict (Dict): {
                "db_id": str
                "table_names_original": List[str], original table name (concert_singer)
                "table_names_clean": List[str], clean table names (concert_singer)
                "column_names_original": List[str], original column name (singer_id)
                "column_names_clean": List[str], clean columns names (singer id)
                "column_db_full_names": List[str], name of table::column in DB (may differ from column_names) (singer::singer_id)
                "column_table_ids": List[int], for each column, the corresponding table index
                "column_types": List[str], column types
                "primary_keys": List[int], the columns indices that are primary key
                "foreign_keys": List[[int, int]], the f-p column index pairs (fk_id, pk_id)
                "sqlite_path": str
                "sqlite_conn": sqlite3.Connection
            }
        
        Output:
            uskg_schema = (for reference) {
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
        """

        db_id = general_fmt_dict["db_id"]
        db_table_orig_names = general_fmt_dict["table_names_original"]
        db_table_clean_names = general_fmt_dict["table_names_clean"]
        db_column_orig_names = general_fmt_dict["column_names_original"]
        db_column_clean_names = general_fmt_dict["column_names_clean"]
        col_db_full_names = general_fmt_dict["column_db_full_names"]
        db_column_table_ids = general_fmt_dict["column_table_ids"]
        db_column_types = general_fmt_dict["column_types"]
        db_primary_keys = general_fmt_dict["primary_keys"]
        db_foreign_keys = general_fmt_dict["foreign_keys"]
        sqlite_path = general_fmt_dict["sqlite_path"]
        sqlite_conn = general_fmt_dict["sqlite_conn"]
        
        # USKG specific
        uskg_primary_keys = [{"column_id": col_idx} for col_idx in db_primary_keys]
        uskg_foreign_keys = [{"column_id": fk_idx, "other_column_id": pk_idx} for fk_idx, pk_idx in db_foreign_keys]

        uskg_schema = {
            "db_id": db_id,
            "db_path": sqlite_path,
            "db_table_names": db_table_orig_names,
            "db_column_names": {
                "table_id": db_column_table_ids,
                "column_name": db_column_orig_names,
            },
            "db_column_types": db_column_types,
            "db_primary_keys": [{"column_id": column_id} for column_id in db_primary_keys],
            "db_foreign_keys": [
                {"column_id": column_id, "other_column_id": other_column_id}
                for column_id, other_column_id in db_foreign_keys
            ],
        }
        
        return uskg_schema

    def get_node_encodings(self, samples):
        """
        Args:
            samples (List[Dict]): each sample must have 'question' for user input and 'rat_sql_graph' for graph info
            pooling_func (Callable): np.array(n_pieces, dim) ==> np.array(dim,); default is np.mean
        Extra args from self.extra_args:
            tokenizer_max_length (int) -> tokenizer(max_length=max_length)
            tokenizer_padding (str) -> tokenizer(padding=padding)
            tokenizer_truncation (bool) -> tokenizer(truncation=truncation)
            pooling (str): None / 'avg' -> avg pooling; 'list' -> no pooling; 'max' -> max pooling; 'first' -> 1st pooling
        """

        if isinstance(samples, dict):
            # back compatible: passing a single dict
            samples = [samples]

        # print('- get_node_encodings() samples:')
        # for d in samples:
        #     print('-- table_id:', d['table_id'])
        #     print('-- question:', d['question'])
        #     print()

        bsz = len(samples)

        # if tokenizer_args is None:
        #     tokenizer_args = {
        #         "max_length": 1024,
        #         "padding": "max_length",
        #         "truncation": True
        #     }
        tokenizer_args = {
            "max_length": self.extra_args.get("tokenizer_max_length", 1024),
            "padding": self.extra_args.get("tokenizer_padding", "max_length"),
            "truncation": self.extra_args.get("tokenizer_truncation", True)
        }

        # if pooling_func is None:
        #     def pooling_func(l): return np.mean(l, axis=0)
        pooling = self.extra_args.get("pooling", "avg")
        pooling_func = POOLING_FUNC_DICT[pooling]


        for sample in samples:
            text_in = sample['question'].strip()
            struct_in = self.sample_to_struct_input(sample)

            _splitter = "; structed knowledge: "
            txt = "{}{}{}".format(text_in, _splitter, struct_in)

            tokenized_txt = self.tokenizer_fast([txt], **tokenizer_args)

            # YS: added
            sample['txt_pieces'] = {
                'text_in': text_in,
                'struct_in': struct_in,
                'splitter': _splitter,
                'txt': txt,
                'tokenized_txt': tokenized_txt,
            }

        txt_list = [sample['txt_pieces']['txt'] for sample in samples]
        tokenized_txt_list = self.tokenizer_fast(txt_list, **tokenizer_args)

        ## Split the BatchEncoding object into different elements in the batch (might cause segfault...?)
        # for in_batch_idx, sample in enumerate(samples):
        #     tokenized_txt = deepcopy(tokenized_txt_list)
        #     tokenized_txt_data = {k : [v[in_batch_idx]] for k, v in tokenized_txt.data.items()}
        #     tokenized_txt_encodings = [tokenized_txt.encodings[in_batch_idx]]
        #     tokenized_txt.__setstate__({
        #         'data': tokenized_txt_data,
        #         'encodings': tokenized_txt_encodings,
        #     })
        #     sample['txt_pieces']['tokenized_txt'] = tokenized_txt

        # breakpoint()

        # Get encoding tensor
        with torch.no_grad():
            if isinstance(self.model, prefixtuning.Model):
                past_prompt = self.model.get_prompt(
                    bsz=bsz,            # bsz = input_ids.shape[0]
                    sample_size=1,      # sample_size=kwargs['num_beams']
                    description=None,
                    knowledge=None,
                )
                encoder_outputs = self.model.pretrain_model.encoder(
                    input_ids=torch.LongTensor(tokenized_txt_list.data['input_ids']).to(self.device_name),
                    attention_mask=torch.LongTensor(tokenized_txt_list.data['attention_mask']).to(self.device_name),
                    past_prompt=past_prompt,
                )
            else:
                # model: finetune.Model
                encoder_outputs = self.model.pretrain_model.encoder(
                    input_ids=torch.LongTensor(tokenized_txt_list.data['input_ids']).to(self.device_name),
                    attention_mask=torch.LongTensor(tokenized_txt_list.data['attention_mask']).to(self.device_name),
                )

        encoder_output_hidden_states = encoder_outputs.last_hidden_state.detach().cpu().numpy()
        if self.debug:
            print('encoder_output_hidden_states:',
                encoder_output_hidden_states.shape)

        # breakpoint()

        batch_node_encodings = []
        valid_in_batch_ids = []
        for in_batch_idx, sample in enumerate(samples):
            txt = sample['txt_pieces']['txt']
            tokenized_txt = sample['txt_pieces']['tokenized_txt']

            # Get node-pieces mapping via char ranges (can throw error)
            try:
                char_ranges_dict = self.collect_node_char_ranges(sample, txt=txt, tokenized_txt=tokenized_txt)
                node_char_ranges = char_ranges_dict['q_node_chars'] + \
                    char_ranges_dict['c_node_chars'] + char_ranges_dict['t_node_chars']
            except SDRASampleError as e:
                # Currently, can be input too long / having empty column name
                print(f'* WARNING: a sample (in_batch_idx = {in_batch_idx}) raised SDRASampleError:')
                print(repr(e))
                continue
            except Exception as e:
                raise e

            # some chars can be mapped to multiple tokens (e.g. 'i' => '▁', 'i' )
            char_to_tokens_dict = defaultdict(list)
            for token_idx, tok in enumerate(tokenized_txt.tokens()):
                if tok == '</s>':
                    break
                char_span = tokenized_txt.token_to_chars(token_idx)
                for char_idx in range(char_span[0], char_span[1]):
                    char_to_tokens_dict[char_idx].append(token_idx)

            node_pieces_ranges = []
            for st, ed in node_char_ranges:
                piece_ids = []
                for char_idx in range(st, ed):
                    _piece_ids = char_to_tokens_dict[char_idx]
                    piece_ids.extend(_piece_ids)

                piece_st = piece_ids[0]
                piece_ed = piece_ids[-1] + 1
                # the collected piece_ids should be continuous
                # ^ not true... some chars can be mapped to multiple tokens (started by ▁ )
                # re-collect a char-to-tokens dict, allow char-to-many-tokens
                assert set(range(piece_st, piece_ed)) == set(piece_ids), piece_ids

                node_pieces_ranges.append((piece_st, piece_ed))

            if self.debug:
                print('node_pieces_ranges:', node_pieces_ranges)

            # Pool the encodings per node
            node_encodings = []
            for piece_st, piece_ed in node_pieces_ranges:
                enc_vecs = encoder_output_hidden_states[in_batch_idx][piece_st : piece_ed]
                enc_pooled = pooling_func(enc_vecs)
                node_encodings.append(enc_pooled)
            
            batch_node_encodings.append(node_encodings)
            valid_in_batch_ids.append(in_batch_idx)

        return batch_node_encodings, valid_in_batch_ids

    def collect_node_char_ranges(self, sample, tokenizer_args=None, txt=None, tokenized_txt=None, uskg_schemas_dict=None):
        """
        Return the char ranges in the txt corresponding to each node. Help building the node-token mapping.
        """
        if 'txt_pieces' in sample:
            # use precomputed (ignoring input params)
            text_in = sample['txt_pieces']['text_in']
            struct_in = sample['txt_pieces']['struct_in']
            _splitter = sample['txt_pieces']['splitter']
            txt = sample['txt_pieces']['txt']
            tokenized_txt = sample['txt_pieces']['tokenized_txt']
        else:
            # use input params or recompute
            text_in = sample['question'].strip()
            struct_in = self.sample_to_struct_input(sample, uskg_schemas_dict)
            _splitter = "; structed knowledge: "
        
            if tokenizer_args is None:
                tokenizer_args = dict()
                
            if txt is None:
                txt = "{}{}{}".format(text_in, _splitter, struct_in)
            
            if tokenized_txt is None:
                # tokenized_txt = tokenizer([txt], max_length=1024, padding="max_length", truncation=True)
                tokenized_txt = self.tokenizer_fast([txt], **tokenizer_args)
                ## possible problem: exceeding max length! when happens, throw SDRASampleError
        
        _num_tokens = sum(tokenized_txt.data['attention_mask'][0])
        if _num_tokens > 1000:
            print(f'WARNING: db_id = {sample["db_id"]}, _num_tokens = {_num_tokens}')
        
        ratsql_graph_nodes = sample['rat_sql_graph']['nodes']
    #     question_toks = sample['question_toks']
        question_toks = sample['rat_sql_graph']['q_nodes_orig']

        _q_nodes = []  # [stem token (node name)]
        q_nodes = []  # [(stem token (node name), orig question token)]
        c_nodes = []  # [(orig table name, orig column name)]
        t_nodes = []  # [orig table name]

        for n in ratsql_graph_nodes:
            if n.startswith('<C>'):
                _n = n[3:]
                _t, _c = _n.split('::')
                c_nodes.append((_t, _c))
            elif n.startswith('<T>'):
                _n = n[3:]
                t_nodes.append(_n)
            else:
                _q_nodes.append(n)

        assert len(_q_nodes) == len(question_toks), (_q_nodes, question_toks)
        q_nodes = list(zip(_q_nodes, question_toks))
        
        # Collection char ranges 
        q_node_chars = []   # [(st, ed)]; same below
        c_node_chars = []
        t_node_chars = []
        
        # Text part
        # Assumption: the mismatch between whitespace words (text_words) and question words only come from trailing puncts
        # Currently the code can handle combining question toks into words
        # text_words = text_in.strip().split(' ') + ['<SENTINAL>']
        text_words = text_in.lower().strip().split() + ['<SENTINAL>']
        text_word_char_ranges = [tokenized_txt.word_to_chars(i) for i in range(len(text_words) - 1)] + [(None, None)]  # -1 to remove the sentinal 

        curr_tw_idx = 0
        curr_tw = text_words[0]
        curr_tw_char_range = text_word_char_ranges[0]
        curr_char_ptr = 0
        for stem_tok, orig_tok in q_nodes:
            if curr_tw == orig_tok:
                # finishing current word 
                q_node_chars.append((curr_char_ptr, curr_char_ptr + len(orig_tok)))   # curr pos to curr pos + len 
                curr_tw_idx += 1
                curr_tw = text_words[curr_tw_idx]
                curr_tw_char_range = text_word_char_ranges[curr_tw_idx]
                curr_char_ptr = curr_tw_char_range[0]
            else:
                # not finishing current word 
                if not curr_tw.startswith(orig_tok):
                    print('- LinkPredictionDataCollector_USKG.collect_node_char_ranges():')
                    print(f'* Warning: Word-token mismatch: {curr_tw}  {orig_tok}')
                    print(f'* text_in: {text_in}')
                    raise SDRASampleError('Word-token mismatch')
                q_node_chars.append((curr_char_ptr, curr_char_ptr + len(orig_tok)))   # curr pos to curr pos + len 
                curr_char_ptr += len(orig_tok)     # move ptr forward by len 
                curr_tw = curr_tw[len(orig_tok):]  # get the remaining chars in the word 

        # can have errors from weird tokens 
        # assert [txt[st:ed].lower() for st, ed in q_node_chars] == question_toks, ([txt[st:ed] for st, ed in q_node_chars], question_toks)
        if [txt[st:ed].lower() for st, ed in q_node_chars] != question_toks:
            print('- LinkPredictionDataCollector_USKG.collect_node_char_ranges():')
            print(f'* Warning: weird mismatch (maybe due to python string behaviors)')
            print(f'* question char spans: {[txt[st:ed] for st, ed in q_node_chars]}')
            print(f'* question nodes orig toks: {question_toks}')
            raise SDRASampleError('Word-token mismatch')

        # Struct part 
        _str_before_struct = text_in + _splitter
        _n_words_before_struct = len(_str_before_struct.strip().split())

        struct_ranges_collector = StructCharRangesCollector()
        try:
            struct_ranges_collector.collect(struct_in, tokenized_txt, _n_words_before_struct)
        except TypeError as e:
            # input too long can cause TypeError
            raise SDRASampleError('Input too long')
        except Exception as e:
            raise e
        
        # Due to rat-sql stemming tokens, rat-sql nodes and uskg text may mismatch
    #     for c_node in c_nodes:
    #         if c_node == ('NONE', '*'):
    #             # the special column in spider, using db_id 
    #             c_node_chars.append(list(struct_ranges_collector.db_id2char_ranges.values())[0])   # assuming only 1 db_id, which should be true...
    #         else:
    #             c_node_chars.append(struct_ranges_collector.column2char_ranges[c_node])

    #     for t_node in t_nodes:
    #         t_node_chars.append(struct_ranges_collector.table2char_ranges[t_node])

        c_node_chars.extend(struct_ranges_collector.db_id_char_ranges_list + struct_ranges_collector.column_char_ranges_list)
        t_node_chars.extend(struct_ranges_collector.table_char_ranges_list)

        ## Check all 
        if self.debug:
            print('** LinkPredictionDataCollector_USKG::collect_node_char_ranges()')
            print('* Q nodes')
            for q_node, (st, ed) in zip(q_nodes, q_node_chars):
                print(q_node, st, ed, txt[st:ed])
            print()
            print('* T nodes')
            for t_node, (st, ed) in zip(t_nodes, t_node_chars):
                print(t_node, st, ed, txt[st:ed])
            print()
            print('* C nodes')
            for c_node, (st, ed) in zip(c_nodes, c_node_chars):
                print(c_node, st, ed, txt[st:ed])
            print()
            
        return {
            "q_node_chars": q_node_chars,
            "c_node_chars": c_node_chars,
            "t_node_chars": t_node_chars,
        }


class BaseGraphDataCollector_USKG_spider(BaseGraphDataCollector_USKG, BaseGraphDataCollector_spider):
    def serialize_schema(
            self,
            question: str,
            db_path: str,
            db_id: str,
            db_column_names: Dict[str, str],
            db_table_names: List[str],
            schema_serialization_type: str = "peteshaw",
            schema_serialization_randomized: bool = False,
            schema_serialization_with_db_id: bool = True,
            schema_serialization_with_db_content: bool = False,
            normalize_query: bool = True,
    ) -> str:
        """ Adapted from seq2seq_construction.spider """
        if schema_serialization_type == "verbose":
            db_id_str = "Database: {db_id}. "
            table_sep = ". "
            table_str = "Table: {table}. Columns: {columns}"
            column_sep = ", "
            column_str_with_values = "{column} ({values})"
            column_str_without_values = "{column}"
            value_sep = ", "
        elif schema_serialization_type == "peteshaw":
            # see https://github.com/google-research/language/blob/master/language/nqg/tasks/spider/append_schema.py#L42
            db_id_str = " | {db_id}"
            table_sep = ""
            table_str = " | {table} : {columns}"
            column_sep = " , "
            column_str_with_values = "{column} ( {values} )"
            column_str_without_values = "{column}"
            value_sep = " , "
        else:
            raise NotImplementedError

        def get_column_str(table_name: str, column_name: str) -> str:
            column_name_str = column_name.lower() if normalize_query else column_name
            if schema_serialization_with_db_content:
                matches = get_database_matches(
                    question=question,
                    table_name=table_name,
                    column_name=column_name,
                    # db_path=(db_path + "/" + db_id + "/" + db_id + ".sqlite"),
                    db_path=db_path,
                )
                if matches:
                    return column_str_with_values.format(
                        column=column_name_str, values=value_sep.join(matches)
                    )
                else:
                    return column_str_without_values.format(column=column_name_str)
            else:
                return column_str_without_values.format(column=column_name_str)

        tables = [
            table_str.format(
                table=table_name.lower() if normalize_query else table_name,
                columns=column_sep.join(
                    map(
                        lambda y: get_column_str(table_name=table_name, column_name=y[1]),
                        filter(
                            lambda y: y[0] == table_id,
                            zip(
                                db_column_names["table_id"],
                                db_column_names["column_name"],
                            ),
                        ),
                    )
                ),
            )
            for table_id, table_name in enumerate(db_table_names)
        ]
        if schema_serialization_randomized:
            random.shuffle(tables)
        if schema_serialization_with_db_id:
            serialized_schema = db_id_str.format(db_id=db_id) + table_sep.join(tables)
        else:
            serialized_schema = table_sep.join(tables)
        return serialized_schema

    def sample_to_struct_input(self, sample):
        """ Build the struct_in for a uskg_sample. """
        db_id = sample["db_id"]
        uskg_schema = self.db_schemas_dict[db_id]
        
        return self.serialize_schema(
            question=sample["question"],
            db_path=uskg_schema["db_path"],
            db_id=db_id,
            db_column_names=uskg_schema["db_column_names"],
            db_table_names=uskg_schema["db_table_names"],
            schema_serialization_type="peteshaw",
            schema_serialization_randomized=False,
            schema_serialization_with_db_id=True,
            schema_serialization_with_db_content=True,
            normalize_query=True,
        )
    

# ## Schema serialization for wikisql 
# def _wikisql_db_id_to_table_name(db_id):
#     return '_'.join(['table'] + db_id.split('-'))

class BaseGraphDataCollector_USKG_wikisql(BaseGraphDataCollector_USKG, BaseGraphDataCollector_wikisql):
    def serialize_schema(
        self,
        question: str,
        db_path: str,
        db_id: str,
        db_column_names: Dict[str, str],
        db_table_names: List[str],
        schema_serialization_type: str = "peteshaw",
        schema_serialization_randomized: bool = False,
        schema_serialization_with_db_id: bool = True,
        schema_serialization_with_db_content: bool = False,
        normalize_query: bool = True,
    ) -> str:
        """ Adapted from seq2seq_construction.spider """
        if schema_serialization_type == "verbose":
            db_id_str = "Database: {db_id}. "
            table_sep = ". "
            table_str = "Table: {table}. Columns: {columns}"
            column_sep = ", "
            column_str_with_values = "{column} ({values})"
            column_str_without_values = "{column}"
            value_sep = ", "
        elif schema_serialization_type == "peteshaw":
            # see https://github.com/google-research/language/blob/master/language/nqg/tasks/spider/append_schema.py#L42
            db_id_str = " | {db_id}"
            table_sep = ""
            table_str = " | {table} : {columns}"
            column_sep = " , "
            column_str_with_values = "{column} ( {values} )"
            column_str_without_values = "{column}"
            value_sep = " , "
        else:
            raise NotImplementedError

        def get_column_str(table_name: str, column_name: str, sqlite_table_name: str, sqlite_column_name: str) -> str:
            """
            Args:
                table_name: the original table name in wikisql dataset (natural langauge)
                column_name: the name is wikisql dataset (natural langauge)
                sqlite_table_name: the table name in .db (table_xxx_xxx)
                sqlite_column_name: the column name in .db (col0, col1, ...)
            """
            column_name_str = column_name.lower() if normalize_query else column_name
            if schema_serialization_with_db_content:
                # print(f'{table_name}::{column_name} ({sqlite_table_name}::{sqlite_column_name})')
                # breakpoint()
                matches = get_database_matches(
                    question=question,
                    table_name=sqlite_table_name,
                    column_name=sqlite_column_name,
                    # db_path=(db_path + "/" + db_id + "/" + db_id + ".sqlite"),
                    db_path=db_path,
                )
                # breakpoint()
                if matches:
                    return column_str_with_values.format(
                        column=column_name_str, values=value_sep.join(matches)
                    )
                else:
                    return column_str_without_values.format(column=column_name_str)
            else:
                return column_str_without_values.format(column=column_name_str)

        tables = []
        for table_id, table_name in enumerate(db_table_names):
            table_col_idx = 0
            column_str_list = []
            for col_table_id, col_name in zip(db_column_names["table_id"], db_column_names["column_name"]):
                if col_table_id != table_id:
                    continue
                column_str_list.append(get_column_str(table_name=table_name,
                                                    column_name=col_name,
                                                    sqlite_table_name=_wikisql_db_id_to_table_name(db_id),
                                                    sqlite_column_name=f'col{table_col_idx}'))
                table_col_idx += 1
                
            tables.append(table_str.format(
                table=table_name.lower() if normalize_query else table_name,
                columns=column_sep.join(column_str_list),
            ))
        # breakpoint()
        if schema_serialization_randomized:
            random.shuffle(tables)
        if schema_serialization_with_db_id:
            serialized_schema = db_id_str.format(db_id=db_id) + table_sep.join(tables)
        else:
            serialized_schema = table_sep.join(tables)
        return serialized_schema

    def sample_to_struct_input(self, sample):
        """ Build the struct_in for a uskg_sample. """
        db_id = sample["table_id"]
        uskg_schema = self.db_schemas_dict[db_id]
        
        return self.serialize_schema(
            question=sample["question"],
            db_path=uskg_schema["db_path"],
            db_id=db_id,
            db_column_names=uskg_schema["db_column_names"],
            db_table_names=uskg_schema["db_table_names"],
            schema_serialization_type="peteshaw",
            schema_serialization_randomized=False,
            schema_serialization_with_db_id=True,
            schema_serialization_with_db_content=True,
            normalize_query=True,
        )

