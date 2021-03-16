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
    kws = ["股票"]
    return True if sum(map(lambda kw: kw in tag, kws)) else False


def retrieve_finance_tag_df(json_files):
    def produce_df(req):
        return pd.DataFrame(list(map(lambda y: y["data"] ,filter(lambda x: x["data"] and type(x["data"]) == type({}), req))))[["entity", "tag"]].explode("tag")
    word_tag_df = pd.concat(list(map(lambda p: produce_df(read_fkg(p)), json_files)), axis = 0).drop_duplicates()
    tags = word_tag_df["tag"].drop_duplicates()
    finance_tags = tags[tags.map(finance_tag_filter)].tolist()
    return word_tag_df[word_tag_df["tag"].isin(finance_tags)]

def filter_high_evidence_func(df):
    with_title_part = high_evidence_finance_df[~pd.isna(high_evidence_finance_df["title"])].copy()
    without_title_part = high_evidence_finance_df[pd.isna(high_evidence_finance_df["title"])].copy()
    need_words = ["股", "证券"]
    without_title_part = without_title_part[without_title_part["header"].map(lambda h_list: sum(map(lambda h: sum(map(lambda w: w in h, need_words)), h_list))).astype(bool)]
    return pd.concat([with_title_part, without_title_part], axis = 0)




total_tables = pd.read_csv("train_table_extract_with_topic.csv")

finance_word_tag_df = retrieve_finance_tag_df(glob("finance*.json"))
finance_word_tag_df["entity_clean"] = finance_word_tag_df["entity"].map(lambda x: x[:x.find("[")] if "[" in x else x)
all_finance_tags = finance_word_tag_df["tag"].unique().tolist()
finance_tag_entities_dict = dict(map(lambda t2: (t2[0], t2[1]["entity_clean"].tolist()) ,finance_word_tag_df.groupby("tag")))

total_tables["fkwds"] = total_tables["all_text_str_elements"].map(lambda x: sorted(set(filter(lambda y: y in finance_word_tag_df["entity_clean"].tolist(), x.split(" ")))))


# Define the label mappings for convenience
#### ABSTAIN can not guess if finance
#ABSTAIN = -1
NOT_FINANCE = 0
FINANCE = 1

#### labeling function all take series as input
@labeling_function()
def if_top_topic_id(x):
    return FINANCE if x.loc["top_topic"] in [16.0, 11.0] else NOT_FINANCE 

@labeling_function()
def if_max_topic_id(x):
    return FINANCE if x.loc["max_topic"] in [16.0, 11.0] else NOT_FINANCE 

@labeling_function()
def if_other_topic(x):
    return NOT_FINANCE if "TOP" in (x.loc["title"] if type(x.loc["title"]) == type("") else "") else FINANCE

#### generate labeling func 
for i, t2 in enumerate(sorted(finance_tag_entities_dict.items())):
    func_index = i
    tag, entities_list = t2
    func_def_format = '''@labeling_function()\ndef if_tag_in_entities_{}(x):\n\treturn FINANCE if set(x.loc["all_text_str_elements"].split(" ")).intersection({}) else NOT_FINANCE '''
    func_def = func_def_format.format(func_index, set(entities_list))
    ic(func_def)
    exec(func_def.strip())


if_tag_in_entities_funcs = list(filter(lambda x: callable(eval(x)) and x.startswith("if_tag_in_entities_") ,dir()))
#### if funcs are all labeling function needed.
if_funcs = list(filter(lambda x: callable(eval(x)) and x.startswith("if_") ,dir()))

#### try run them on first record
dict(map(lambda func_name: (func_name, eval(func_name)(total_tables.iloc[0])), if_funcs))

# Define the set of labeling functions (LFs)
lfs = list(map(eval, if_funcs))

# Apply the LFs to the unlabeled training data
applier = PandasLFApplier(lfs)
L_train = applier.apply(total_tables)

total_tables_cp = total_tables.copy()
label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train, n_epochs=500, log_freq=50, seed=123)
total_tables_cp["label"] = label_model.predict(L=L_train)
total_tables_cp["label"].value_counts()

label_fkwds_df = pd.DataFrame(total_tables_cp[["label", "fkwds"]].apply(lambda s: (s["label"], len(s["fkwds"])), axis = 1).values.tolist())
label_fkwds_df.columns = ["label", "fkwds_len"]


total_tables_cp["other_t"] = total_tables.apply(if_other_topic, axis = 1)
low_evidence_finance_df = total_tables_cp[(total_tables_cp["label"] == 1) & (total_tables_cp["other_t"] == 1)].copy()

#### low_part 房地产 和 其它部分("股" as finance)
#### high_part 金融 和 其它部分("票房" as non-finance)
low_part = low_evidence_finance_df[low_evidence_finance_df["fkwds"].map(len) == 1].copy()
high_part = low_evidence_finance_df[low_evidence_finance_df["fkwds"].map(len) > 1].copy()

high_evidence_finance_df = pd.concat([low_part[low_part["title"].map(lambda x: "股" in (x if type(x) == type("") else ""))],
          high_part[high_part["title"].map(lambda x: "票房" not in (x if type(x) == type("") else ""))]
          ], axis = 0)

##### with some tiny noise not upto 10 as 房地产 建筑
##### use this as final require as finance
high_evidence_finance_df_filtered = filter_high_evidence_func(high_evidence_finance_df)

high_evidence_finance_df_filtered.to_csv("finance_table_desc.csv", index = False)

