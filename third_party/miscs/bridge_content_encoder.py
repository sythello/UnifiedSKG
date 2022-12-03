"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Encode DB content.
"""

import difflib
from typing import List, Optional, Tuple
from rapidfuzz import fuzz
import sqlite3
import functools
import time

# fmt: off
_stopwords = {'who', 'ourselves', 'down', 'only', 'were', 'him', 'at', "weren't", 'has', 'few', "it's", 'm', 'again',
              'd', 'haven', 'been', 'other', 'we', 'an', 'own', 'doing', 'ma', 'hers', 'all', "haven't", 'in', 'but',
              "shouldn't", 'does', 'out', 'aren', 'you', "you'd", 'himself', "isn't", 'most', 'y', 'below', 'is',
              "wasn't", 'hasn', 'them', 'wouldn', 'against', 'this', 'about', 'there', 'don', "that'll", 'a', 'being',
              'with', 'your', 'theirs', 'its', 'any', 'why', 'now', 'during', 'weren', 'if', 'should', 'those', 'be',
              'they', 'o', 't', 'of', 'or', 'me', 'i', 'some', 'her', 'do', 'will', 'yours', 'for', 'mightn', 'nor',
              'needn', 'the', 'until', "couldn't", 'he', 'which', 'yourself', 'to', "needn't", "you're", 'because',
              'their', 'where', 'it', "didn't", 've', 'whom', "should've", 'can', "shan't", 'on', 'had', 'have',
              'myself', 'am', "don't", 'under', 'was', "won't", 'these', 'so', 'as', 'after', 'above', 'each', 'ours',
              'hadn', 'having', 'wasn', 's', 'doesn', "hadn't", 'than', 'by', 'that', 'both', 'herself', 'his',
              "wouldn't", 'into', "doesn't", 'before', 'my', 'won', 'more', 'are', 'through', 'same', 'how', 'what',
              'over', 'll', 'yourselves', 'up', 'mustn', "mustn't", "she's", 're', 'such', 'didn', "you'll", 'shan',
              'when', "you've", 'themselves', "mightn't", 'she', 'from', 'isn', 'ain', 'between', 'once', 'here',
              'shouldn', 'our', 'and', 'not', 'too', 'very', 'further', 'while', 'off', 'couldn', "hasn't", 'itself',
              'then', 'did', 'just', "aren't"}
# fmt: on

_commonwords = {"no", "yes", "many"}

# Raising segfault
# def is_number(s: str) -> bool:
#     try:
#         float(s.replace(",", ""))
#         return True
#     except:
#         return False

# Not working either
# def is_number(s: str) -> bool:
#     print(f'*** is_number(): {s}')  # working
#     # print(f'*{s}')    # not working
#     # print(s)          # not working 
#     ret = True
#     try:
#         # s_repl = s.replace(",", "")
#         # print(f'*** is_number() replaced: {s_repl}')
#         # x = float(s_repl)
#         x = float(s.replace(",", ""))
#         # print(f'*** is_number() float: {x}')
#     except:
#         ret = False
#     # print(f'*** is_number() return: {ret}')
#     return ret

# This one works?
def is_number(s: str) -> bool:
    if not s.isascii():
        return False
    try:
        float(s.replace(",", ""))
        return True
    except:
        return False


def is_stopword(s: str) -> bool:
    return s.strip() in _stopwords


def is_commonword(s: str) -> bool:
    return s.strip() in _commonwords


def is_common_db_term(s: str) -> bool:
    return s.strip() in ["id"]


class Match(object):
    def __init__(self, start: int, size: int) -> None:
        self.start = start
        self.size = size


def is_span_separator(c: str) -> bool:
    return c in "'\"()`,.?! "


def split(s: str) -> List[str]:
    return [c.lower() for c in s.strip()]


def prefix_match(s1: str, s2: str) -> bool:
    i, j = 0, 0
    for i in range(len(s1)):
        if not is_span_separator(s1[i]):
            break
    for j in range(len(s2)):
        if not is_span_separator(s2[j]):
            break
    if i < len(s1) and j < len(s2):
        return s1[i] == s2[j]
    elif i >= len(s1) and j >= len(s2):
        return True
    else:
        return False


def get_effective_match_source(s: str, start: int, end: int) -> Match:
    _start = -1

    for i in range(start, start - 2, -1):
        if i < 0:
            _start = i + 1
            break
        if is_span_separator(s[i]):
            _start = i
            break

    if _start < 0:
        return None

    _end = -1
    for i in range(end - 1, end + 3):
        if i >= len(s):
            _end = i - 1
            break
        if is_span_separator(s[i]):
            _end = i
            break

    if _end < 0:
        return None

    while _start < len(s) and is_span_separator(s[_start]):
        _start += 1
    while _end >= 0 and is_span_separator(s[_end]):
        _end -= 1

    return Match(_start, _end - _start + 1)


def get_matched_entries(
    s: str, field_values: List[str], m_theta: float = 0.85, s_theta: float = 0.85
) -> Optional[List[Tuple[str, Tuple[str, str, float, float, int]]]]:
    if not field_values:
        return None

    if isinstance(s, str):
        n_grams = split(s)
    else:
        n_grams = s
    
    # print('** n_grams:', n_grams)
    # print('** field_values:', field_values)
    # breakpoint()

    matched = dict()
    for field_value in field_values:
        if not isinstance(field_value, str):
            continue
        fv_tokens = split(field_value)
        # print(f'** ({field_value}) fv_tokens:', fv_tokens)
        sm = difflib.SequenceMatcher(None, n_grams, fv_tokens)
        match = sm.find_longest_match(0, len(n_grams), 0, len(fv_tokens))
        # print('** get_matched_entries:', field_value, match)
        if match.size > 0:
            # this avoids segfault
            # if match.size / len(fv_tokens) > m_theta:
            #     match_str = field_value[match.b: match.b + match.size]
            #     matched[match_str] = (field_value, field_value, 1.0, 1.0, match.size)
            # continue

            source_match = get_effective_match_source(
                n_grams, match.a, match.a + match.size
            )
            if source_match and source_match.size > 1:
                match_str = field_value[match.b : match.b + match.size]
                source_match_str = s[
                    source_match.start : source_match.start + source_match.size
                ]
                c_match_str = match_str.lower().strip()
                c_source_match_str = source_match_str.lower().strip()
                c_field_value = field_value.lower().strip()

                # this avoids segfault
                # matched[match_str] = (
                #     field_value,
                #     source_match_str,
                #     match.size / len(fv_tokens),
                #     match.size / len(fv_tokens),
                #     match.size,
                # )
                # continue

                if (
                    c_match_str
                    and not is_number(c_match_str)      # Problem is in this line
                    and not is_common_db_term(c_match_str)
                ):
                    ## CHECKED ok
                    if (
                        is_stopword(c_match_str)
                        or is_stopword(c_source_match_str)
                        or is_stopword(c_field_value)
                    ):
                        continue
                    if c_source_match_str.endswith(c_match_str + "'s"):
                        match_score = 1.0
                    else:
                        if prefix_match(c_field_value, c_source_match_str):
                            match_score = (
                                fuzz.ratio(c_field_value, c_source_match_str) / 100
                            )
                        else:
                            match_score = 0
                    if (
                        is_commonword(c_match_str)
                        or is_commonword(c_source_match_str)
                        or is_commonword(c_field_value)
                    ) and match_score < 1:
                        continue
                    ## END CHECKED

                    s_match_score = match_score
                    if match_score >= m_theta and s_match_score >= s_theta:
                        if field_value.isupper() and match_score * s_match_score < 1:
                            continue
                        matched[match_str] = (
                            field_value,
                            source_match_str,
                            match_score,
                            s_match_score,
                            match.size,
                        )

    if not matched:
        return None
    else:
        return sorted(
            matched.items(),
            key=lambda x: (1e16 * x[1][2] + 1e8 * x[1][3] + x[1][4]),
            reverse=True,
        )


@functools.lru_cache(maxsize=1000, typed=False)
def get_column_picklist(table_name: str, column_name: str, db_path: str) -> list:
    fetch_sql = "SELECT DISTINCT `{}` FROM `{}`".format(column_name, table_name)
    # print('fetch_sql:', fetch_sql)
    try:
        conn = sqlite3.connect(db_path)
        conn.text_factory = bytes
        # conn.text_factory = str
        c = conn.cursor()
        c.execute(fetch_sql)
        picklist = set()
        for x in c.fetchall():
            if isinstance(x[0], str):
                picklist.add(x[0].encode("utf-8"))
            elif isinstance(x[0], bytes):
                try:
                    picklist.add(x[0].decode("utf-8"))
                except UnicodeDecodeError:
                    picklist.add(x[0].decode("latin-1"))
            else:
                picklist.add(x[0])
            # picklist.add("placeholder")
        picklist = list(picklist)
    finally:
        conn.close()
    # print('picklist:', picklist)
    return picklist


def get_database_matches(
    question: str,
    table_name: str,
    column_name: str,
    db_path: str,
    top_k_matches: int = 2,
    match_threshold: float = 0.85,
) -> List[str]:
    # breakpoint()
    picklist = get_column_picklist(
        table_name=table_name, column_name=column_name, db_path=db_path
    )
    # breakpoint()   # this makes the segfault go away?!
    # print(picklist)
    matches = []
    if picklist and isinstance(picklist[0], str):
        matched_entries = get_matched_entries(
            s=question,
            field_values=picklist,
            m_theta=match_threshold,
            s_theta=match_threshold,
        )
        # matched_entries = [(val, (val, val, 1.0, 1.0, 1)) for val in picklist if val in question]     # this won't give segfault 
        # print('matched_entries:', matched_entries)
        # breakpoint()
        if matched_entries:
            num_values_inserted = 0
            for _match_str, (
                field_value,
                _s_match_str,
                match_score,
                s_match_score,
                _match_size,
            ) in matched_entries:
                if "name" in column_name and match_score * s_match_score < 1:
                    continue
                if table_name != "sqlite_sequence":  # Spider database artifact
                    matches.append(field_value)
                    num_values_inserted += 1
                    if num_values_inserted >= top_k_matches:
                        break
    # breakpoint()
    return matches
