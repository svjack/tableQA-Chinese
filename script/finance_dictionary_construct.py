#!/usr/bin/env python
# coding: utf-8

import inspect
import json
import os
import urllib.request
from functools import reduce
from glob import glob
from time import sleep
from urllib.parse import quote

import jieba
import numpy as np
import pandas as pd
import seaborn as sns
from icecream import ic
from snorkel.labeling import PandasLFApplier, labeling_function
from snorkel.labeling.model import LabelModel


pd.set_option("display.max_rows", 200)

def retrieve_entities_data(entities, sleep_time = 2, output_file = None):
    assert type(entities) == type([])
    assert type(sleep_time) == type(0)
    def retrieve_one(entity):
        url='https://api.ownthink.com/kg/knowledge?entity=%s'%(quote(entity.replace(" ", "")))
        resb = urllib.request.urlopen(url).read()
        return resb
    with open(output_file, "w") as f:
        for idx ,entity in enumerate(entities):
            use_time = sleep_time
            sleep(use_time)
            resb = retrieve_one(entity)
            while "异常" in resb.decode():
                use_time *= 2
                ic("sleep ", use_time, "s")
                sleep(use_time)
                resb = retrieve_one(entity)
                
            f.write(resb.decode().strip() + "\n")
            if idx % 10 == 0:
                ic(idx)


def read_fkg(path):
    req = [[]]
    with open(path, "r") as f:
        lines = f.readlines()
        for l in lines:
            if l.startswith("{"):
                req.append([])
            req[-1].append(l)
    #return req
    return list(map(lambda x: json.loads("".join(map(lambda y: y.replace("\n", "").strip(), x)).replace("\n", "").strip()), req[1:]))


def finance_tag_filter(tag):
    if type(tag) != type(""):
        return False
    kws = "公司 组织 股票 经纪 行业 经管".split(" ")
    return True if sum(map(lambda kw: kw in tag, kws)) else False


train_table_extract_df = pd.read_csv("train_table_extract_with_topic.csv")
finance_words_s = pd.Series(reduce(lambda a, b: a + b ,train_table_extract_df[(train_table_extract_df["max_topic"] == 16.0) & (train_table_extract_df["top_topic"] == 16.0)]["all_text_list_elements"].map(lambda x: eval(x)).values.tolist()))
finance_word_df = finance_words_s.value_counts().reset_index()
entities = finance_word_df["index"].tolist()
retrieve_entities_data(entities, sleep_time = 0, output_file = "finance_words.json")


req = read_fkg("finance_words.json")
none_empty_req = list(map(lambda y: y["data"] ,filter(lambda x: x["data"] and type(x["data"]) == type({}), req)))
finance_word_kg_df = pd.DataFrame(none_empty_req)
word_tag_df = finance_word_kg_df[["entity", "tag"]].explode("tag")


tags = word_tag_df["tag"].drop_duplicates()
finance_tags = tags[tags.map(finance_tag_filter)].tolist()
finance_tag_df = word_tag_df[word_tag_df["tag"].isin(finance_tags)].copy()


#### not contain subwords such as "证券"
finance_and_economic_words = finance_tag_df["entity"].unique().tolist()
finance_and_economic_words_df = pd.concat([pd.Series(finance_and_economic_words) ,pd.Series(finance_and_economic_words).map(jieba.cut).map(list)], axis = 1)

finance_and_economic_words_df.columns = ["word", "word_split"]
split_part = finance_and_economic_words_df[finance_and_economic_words_df["word_split"].map(lambda x: len(x) > 1)]
entities = sorted(reduce(lambda a, b: a.union(b) ,split_part["word_split"].map(set).tolist()))


retrieve_entities_data(entities, sleep_time = 0, output_file = "finance_sub_words.json")




