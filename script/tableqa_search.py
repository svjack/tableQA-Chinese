#!/usr/bin/env python
# coding: utf-8

import json
import os
import re
import sqlite3
import urllib.request
from ast import literal_eval
from functools import reduce
from time import sleep
from urllib.parse import quote

import jieba
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sqlite_utils
import sqlitefts as fts
from bertopic import BERTopic
from bertopic._embeddings import languages
from icecream import ic
from keybert import KeyBERT

assert 'chinese (simplified)' in languages

data_dir = "../TableQA/TableQA"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
val_dir = os.path.join(data_dir, "val")


def template_extractor(rows_filtered):
    re_words = re.compile(u"[\u4e00-\u9fa5]+")
    nest_collection = pd.DataFrame(rows_filtered).applymap(lambda x: tuple(sorted(set(re.findall(re_words, x))))).drop_duplicates().values.tolist()
    def flatten_collection(c):
        if not c:
            return c
        if type(c[0]) == type(""):
            return c
        else:
            c = list(c)
        return flatten_collection(reduce(lambda a, b: a + b, map(list ,c)))
    return flatten_collection(nest_collection)

def extract_text_info_from_tables_json(path):
    assert os.path.exists(path)
    assert path.endswith(".tables.json")
    tables_json_df = pd.read_json(path, lines = True)
    def single_s_table_extractor(s):
        table_dict = s.to_dict()
        header = table_dict["header"]
        common = table_dict["common"]
        title = table_dict["title"]
        name = table_dict["name"]
        id_ = table_dict["id"]
        need_columns_index_list = []
        for i, type_ in enumerate(table_dict["types"]):
            if type_ == "text":
                need_columns_index_list.append(i)
        need_rows_filtered_by_columns = []
        for row in table_dict["rows"]:
            req = []
            for i, ele in enumerate(row):
                if i in need_columns_index_list:
                    req.append(ele)
            need_rows_filtered_by_columns.append(req)
        assert len(need_rows_filtered_by_columns) == len(table_dict["rows"])
        req_dict = {
            "name": name,
            "id": id_,
            "header": header,
            "common": common,
            "title": title,
            "table_elements": sorted(set(template_extractor(need_rows_filtered_by_columns))),
        }
        return req_dict
    need_df = pd.DataFrame(tables_json_df.apply(single_s_table_extractor, axis = 1).tolist())
    #return need_df
    text_cols = ["header", "common", "title", "table_elements"]
    need_df["all_text_list"] = need_df[text_cols].apply(lambda s: reduce(lambda a, b: a + b ,s.map(lambda ele: [ele] if type(ele) == type("") else list(ele)).tolist())
                                                       , axis = 1).map(set).map(sorted).map(lambda x: [x])
    need_df["all_text_list_elements"] = need_df["all_text_list"].map(
        lambda x: sorted(set(template_extractor(x)))
    )
    return need_df


def eval_df(df):
    columns = df.columns.tolist()
    eval_cols = list(map(lambda tt2: tt2[0] ,filter(lambda t2: t2[1].strip().startswith("[") ,df.iloc[0, :].to_dict().items())))
    for eval_col in eval_cols:
        df[eval_col] = df[eval_col].map(literal_eval)
    return df


#### information extraction
train_table_extract_df = extract_text_info_from_tables_json(os.path.join(train_dir, "train.tables.json"))
train_table_extract_df.to_csv("train_table_search_content.csv", index = False)
train_table_extract_df = pd.read_csv("train_table_search_content.csv")
train_table_extract_df_eval = eval_df(train_table_extract_df)
text_cnt_s = pd.Series(reduce(lambda a, b:a + b ,train_table_extract_df["all_text_list_elements"].tolist())).value_counts()
#### common_words may also contain some info but is a topic level amoung tables
cnt_throshold = 12
common_words = text_cnt_s[text_cnt_s > cnt_throshold].sort_values(ascending = False).index.tolist()
train_table_extract_df["common_words"] = train_table_extract_df["all_text_list_elements"].map(lambda x: list(filter(lambda ele: ele in common_words, x)))

#### topic model fit and predict
topic_model = BERTopic("chinese (simplified)")
#### fit transform tokens
outputs = topic_model.fit_transform(train_table_extract_df.iloc[:10000]["all_text_list_elements"].map(lambda x: " ".join(x)).values.tolist())
#### fir transform sub tokens
outputs = topic_model.fit_transform(train_table_extract_df.iloc[:10000]["all_text_list_elements"].map(lambda x: " ".join(map(lambda y: " ".join(jieba.cut(y)), x))).values.tolist())
train_table_extract_df["all_text_str_elements"] = train_table_extract_df["all_text_list_elements"].map(lambda x: " ".join(map(lambda y: " ".join(jieba.cut(y)), x))).values.tolist()
topic_df = topic_model.get_topic_freq()
topic_df["topic_desc"] = topic_df["Topic"].map(topic_model.get_topic)
topic_df = topic_df.explode("topic_desc")
topic_df["entity"] = topic_df["topic_desc"].map(lambda x: x[0])
topic_df["score"] = topic_df["topic_desc"].map(lambda x: x[1])
del topic_df["topic_desc"]
topic_words_dict = dict(topic_df[["entity", "Topic", "score"]].apply(lambda s: (s.iloc[0], s.iloc[1:].tolist()), axis = 1).values.tolist())
train_table_extract_df["topic_words"] = train_table_extract_df["all_text_list_elements"].map(lambda x: list(map(lambda x: (x, topic_words_dict[x]) ,filter(lambda ele: ele in topic_words_dict, x))))
train_table_extract_df["top_topic"] = train_table_extract_df["topic_words"].map(lambda x: pd.DataFrame(x).iloc[:, 1].value_counts().index.tolist()[0][0] if x else np.nan)
train_table_extract_df["max_topic"] = train_table_extract_df["topic_words"].map(lambda x: pd.DataFrame(pd.DataFrame(x).iloc[:, 1].value_counts().index.tolist()).sort_values(by = 1, ascending = False).iloc[0][0] if x else np.nan)

#### serlize to local
train_table_extract_df.to_csv("train_table_extract_with_topic.csv", index = False)

