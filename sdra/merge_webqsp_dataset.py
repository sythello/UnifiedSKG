import os, sys, json
from copy import deepcopy
from collections import defaultdict

from argparse import Namespace

import random

def merge_sample(stf_sample, sr_sample):
    """ Merge stanford sample entities text forms (["entities"]) and logical forms (["s_expression"], ["sparql"]) into SR sample """
    
    if len(stf_sample['entities']) != len(sr_sample['entities']):
        merge_sample._warnings += 1
        print(f"* merge_sample() Warning (#{merge_sample._warnings}): Entity number mismatch!")
        print(f"question: {stf_sample['question']} | {sr_sample['question']}")
        print(f"Stanford entities: {stf_sample['entities']}")
        print(f"SR entities: {sr_sample['entities']}\n")
    
    out_sample = deepcopy(sr_sample)
    out_sample['entities'] = [{'kb_id': kb_id, 'text': text} for text, kb_id in stf_sample['entities']]
    out_sample['s_expression'] = stf_sample['s_expression']
    out_sample['sparql'] = stf_sample['sparql']

    return out_sample

merge_sample._warnings = 0


def _index_to_chars(idx):
    s = ""
    while True:
        rem = idx % 26
        idx = idx // 26
        c = chr(ord('a') + rem)
        s = c + s
        if idx == 0:
            break
    return s


def shorten_sample_ids(in_sample):
    topic_ent_kb_ids = set()
    all_ent_kb_ids = set()
    def _maybe_add_id(kb_id, topic=False):
        if not kb_id.startswith('m.0'):
            return False
        if topic:
            topic_ent_kb_ids.add(kb_id)
        else:
            all_ent_kb_ids.add(kb_id)
        return True

    for d in in_sample['entities']:
        kb_id = d['kb_id']
        _maybe_add_id(kb_id, topic=True)
    for h, _, t in in_sample['subgraph']['tuples']:
        _maybe_add_id(h)
        _maybe_add_id(t)

    _warned = False
    for d in in_sample['answers']:
        kb_id = d['kb_id']
        if kb_id.startswith('m.0') and kb_id not in all_ent_kb_ids:
            if not _warned:
                shorten_sample_ids._warnings += 1
                print(f"* abbrev_sample_ids() Warning (#{shorten_sample_ids._warnings}): answer {kb_id} not in subgraph (sample ID = {in_sample['id']})")
                _warned = True
            _maybe_add_id(kb_id)
    for e in in_sample['subgraph']['entities']:
        if e.startswith('m.0') and e not in all_ent_kb_ids:
            print(f"* abbrev_sample_ids() Warning: entity {e} not in subgraph (sample ID = {in_sample['id']})")
            _maybe_add_id(e)
    for e, _ in in_sample.get('paths', []):
        if e.startswith('m.0') and e not in all_ent_kb_ids:
            print(f"* abbrev_sample_ids() Warning: path entity {e} not in subgraph (sample ID = {in_sample['id']})")
            _maybe_add_id(e)

    # assign new short IDs with some randomization; topic entities always in front
    non_topic_kb_ids = all_ent_kb_ids - topic_ent_kb_ids
    _topic = list(topic_ent_kb_ids)
    _non_topic = list(non_topic_kb_ids)
    random.shuffle(_topic)
    random.shuffle(_non_topic)
    kb_ids = _topic + _non_topic
    kb_id2short_id = {kb_id : _index_to_chars(i) for i, kb_id in enumerate(kb_ids)}
    short_id2kb_id = {v : k for k, v in kb_id2short_id.items()}

    def _get_new_id(kb_id):
        if not kb_id.startswith('m.0'):
            return kb_id
        return kb_id2short_id[kb_id]

    out_sample = deepcopy(in_sample)
    for d in out_sample['entities']:
        d['kb_id'] = _get_new_id(d['kb_id'])
    for trip in out_sample['subgraph']['tuples']:
        trip[0] = _get_new_id(trip[0])
        trip[2] = _get_new_id(trip[2])
    for d in out_sample['answers']:
        d['kb_id'] = _get_new_id(d['kb_id'])
    for i, e in enumerate(out_sample['subgraph']['entities']):
        out_sample['subgraph']['entities'][i] = _get_new_id(e)
    for i, (e, _) in enumerate(out_sample.get('paths', [])):
        out_sample['paths'][i][0] = _get_new_id(e)

    # assert 'm.0' not in json.dumps(out_sample), out_sample

    out_sample['id_mapping'] = short_id2kb_id

    return out_sample

shorten_sample_ids._warnings = 0


def main(args):
    stanford_kgqa_webqsp_dir = args.stanford_kgqa_webqsp_dir
    sr_webqsp_dir = args.sr_webqsp_dir
    output_dir = args.output_dir
    ds_list = ['train', 'dev', 'test']

    stf_paths = [os.path.join(stanford_kgqa_webqsp_dir, f'{ds}.jsonl') for ds in ds_list]

    stf_id2sample = dict()
    for p in stf_paths:
        with open(p, 'r') as f:
            samples = [json.loads(l) for l in f]
        for sample in samples:
            s_id = sample['ID'].split('.')[0]   # stanford dataset ID format: WebQTrn-116.P0; SR id format: WebQTrn-116
            stf_id2sample[s_id] = sample

    for ds in ds_list:
        sr_path = os.path.join(sr_webqsp_dir, f'{ds}_simple.json')
        with open(sr_path, 'r') as f:
            sr_samples = [json.loads(l) for l in f]
        out_samples = []
        for sr_sample in sr_samples:
            s_id = sr_sample['id']
            stf_sample = stf_id2sample[s_id]
            out_sample = merge_sample(stf_sample=stf_sample, sr_sample=sr_sample)
            short_out_sample = shorten_sample_ids(out_sample)
            out_samples.append(short_out_sample)

        out_path = os.path.join(output_dir, f'{ds}_simple_short.jsonl')
        with open(out_path, 'w') as f:
            for out_sample in out_samples:
                f.write(json.dumps(out_sample) + '\n')

if __name__ == "__main__":
    # print(_index_to_chars(7))
    # print(_index_to_chars(77))
    # print(_index_to_chars(777))

    args = Namespace()
    args.stanford_kgqa_webqsp_dir = "/vault5/yshao/USKG/webqsp"
    args.sr_webqsp_dir = "/home/yshao/Projects/SubgraphRetrievalKBQA/src/tmp/reader_data/webqsp"
    args.output_dir = "/home/yshao/Projects/SDR-analysis/data/webqsp"

    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
