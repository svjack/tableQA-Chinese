#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import os
import ast
import re
import json
from icecream import ic
from copy import deepcopy
from itertools import product, combinations


import pandas as pd
import os
import sys
from pyarrow.filesystem import LocalFileSystem
from functools import reduce
import nltk
from nltk import pos_tag, word_tokenize
from collections import namedtuple
from ast import literal_eval

from torch.nn import functional
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn.utils import rnn as rnn_utils
import math

from icecream import ic
import seaborn as sns

import matplotlib.pyplot as plt

import shutil

#from keybert import KeyBERT
#from bertopic import BERTopic


import sqlite3
import sqlite_utils
from icecream import ic
import jieba
import pandas as pd
import urllib.request
from urllib.parse import quote
from time import sleep
import json
import os
from collections import defaultdict
import re
from functools import reduce, partial

#### used in this condition extract in training.
op_sql_dict = {0:">", 1:"<", 2:"==", 3:"!="}
#### used by clf for intension inference
agg_sql_dict = {0:"", 1:"AVG", 2:"MAX", 3:"MIN", 4:"COUNT", 5:"SUM"}
#### final to combine them (one for 0, and multi for 1 2)
conn_sql_dict = {0:"", 1:"and", 2:"or"}

#### kws and time pattern defination
and_kws = ("且", "而且", "并且", "和", "当中", "同时")
or_kws = ("或", "或者",)
conn_kws = and_kws + or_kws

pattern_list = [u"[年月\.\-\d]+", u"[年月\d]+", u"[年个月\d]+", u"[年月日\d]+"]

time_kws = ("什么时候", "时间", "时候")

sum_count_high_kws = ('多少个', '有几个', '总共') + ('总和','一共',) + ("总数",)
mean_kws = ('平均数', '均值', '平均值', '平均')
max_kws = ('最大', '最多', '最大值', '最高')
min_kws = ('最少', '最小值', '最小', '最低')
sum_count_low_kws = ('个', '总共') + ('总和','加','总','一共','和',) + ("哪些", "查", "数量", "数") + ("几",) + ('多少', "多大")  + ("总数",)
max_special_kws = ("以上", "大于")
min_special_kws = ("以下", "小于")

qst_kws = ("多少", "什么", "多大", "哪些", "怎么", "情况", "那些", "哪个")

only_kws_columns = {"城市": "=="}

##### jointbert predict model init start
jointbert_path = "../../featurize/JointBERT"
sys.path.append(jointbert_path)


from model.modeling_jointbert import JointBERT
from model.modeling_jointbert import *
from trainer import *
from main import *
from data_loader import *


pred_parser = argparse.ArgumentParser()

pred_parser.add_argument("--input_file", default="conds_pred/seq.in", type=str, help="Input file for prediction")
pred_parser.add_argument("--output_file", default="conds_pred/sample_pred_out.txt", type=str, help="Output file for prediction")
pred_parser.add_argument("--model_dir", default="bert", type=str, help="Path to save, load model")

pred_parser.add_argument("--batch_size", default=32, type=int, help="Batch size for prediction")
pred_parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")


pred_parser_config_dict = dict(map(lambda item:(item.option_strings[0].replace("--", ""), item.default) ,pred_parser.__dict__["_actions"]))
pred_parser_config_dict = dict(filter(lambda t2: t2[0] != "-h", pred_parser_config_dict.items()))

pred_parser_namedtuple = namedtuple("pred_parser_config", pred_parser_config_dict.keys())
for k, v in pred_parser_config_dict.items():
    if type(v) == type(""):
        exec("pred_parser_namedtuple.{}='{}'".format(k, v))
    else:
        exec("pred_parser_namedtuple.{}={}".format(k, v))


from predict import *


pred_config = pred_parser_namedtuple
args = get_args(pred_config)
device = get_device(pred_config)

args_parser_namedtuple = namedtuple("args_config", args.keys())
for k, v in args.items():
    if type(v) == type(""):
        exec("args_parser_namedtuple.{}='{}'".format(k, v))
    else:
        exec("args_parser_namedtuple.{}={}".format(k, v))


args = args_parser_namedtuple
pred_model = MODEL_CLASSES["bert"][1].from_pretrained(args.model_dir,
                                                                  args=args,
                                                            intent_label_lst=get_intent_labels(args),
                                                                  slot_label_lst=get_slot_labels(args))
pred_model.to(device)
pred_model.eval()

intent_label_lst = get_intent_labels(args)
slot_label_lst = get_slot_labels(args)
pad_token_label_id = args.ignore_index
tokenizer = load_tokenizer(args)
## jointbert predict model init end


###### one sent conds decomp start
def predict_single_sent(question):
    text = " ".join(list(question))
    batch = convert_input_file_to_tensor_dataset([text.split(" ")], pred_config, args, tokenizer, pad_token_label_id).tensors
    batch = tuple(t.to(device) for t in batch)
    inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "intent_label_ids": None,
                      "slot_labels_ids": None}
    inputs["token_type_ids"] = batch[2]
    outputs = pred_model(**inputs)
    _, (intent_logits, slot_logits) = outputs[:2]
    intent_preds = intent_logits.detach().cpu().numpy()
    slot_preds = slot_logits.detach().cpu().numpy()
    intent_preds = np.argmax(intent_preds, axis=1)
    slot_preds = np.argmax(slot_preds, axis=2)
    all_slot_label_mask = batch[3].detach().cpu().numpy()
    slot_label_map = {i: label for i, label in enumerate(slot_label_lst)}
    slot_preds_list = [[] for _ in range(slot_preds.shape[0])]
    for i in range(slot_preds.shape[0]):
        for j in range(slot_preds.shape[1]):
            if all_slot_label_mask[i, j] != pad_token_label_id:
                slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])
    pred_l = []
    for words, slot_preds, intent_pred in zip([text.split(" ")], slot_preds_list, intent_preds):
        line = ""
        for word, pred in zip(words, slot_preds):
            if pred == 'O':
                line = line + word + " "
            else:
                line = line + "[{}:{}] ".format(word, pred)
        pred_l.append((line, intent_label_lst[intent_pred]))
    return pred_l[0]


###@@ conn_kws = ["且", "或", "或者", "和"]
'''
and_kws = ("且", "而且", "并且", "和", "当中", "同时")
or_kws = ("或", "或者",)
conn_kws = and_kws + or_kws
'''
#conn_kws = ("且", "或", "或者", "和") + ("而且", "并且", "当中")
#### some algorithm use in it. 
def recurrent_extract(question):
    def filter_relation(text):
        #kws = ["且", "或", "或者", "和"]
        kws = conn_kws
        req = text
        for kw in sorted(kws, key= lambda x: len(x))[::-1]:
            req = req.replace(kw, "")
        return req
    def produce_plain_text(text):
        ##### replace tag string from text
        kws = ["[", "]", " ", ":B-HEADER", ":I-HEADER", ":B-VALUE", ":I-VALUE"]
        plain_text = text
        for kw in kws:
            plain_text = plain_text.replace(kw, "")
        return plain_text
    def find_min_commmon_strings(c):
        ##### {"jack", "ja", "ss", "sss", "ps", ""} -> {"ja", "ss", "ps"}
        common_strings = list(filter(lambda x: type(x) == type("") ,
                                     map(lambda t2: t2[0] 
                                         if t2[0] in t2[1] 
                                             else (t2[1] 
                                                if t2[1] in t2[0] 
                                                   else (t2[0], t2[1])),combinations(c, 2))))
        req = set([])
        while c:
            ele = c.pop()
            if all(map(lambda cc: cc not in ele, common_strings)):
                req.add(ele)
        req = req.union(set(common_strings))
        return set(filter(lambda x: x, req))
    def extract_scope(scope_text):
        def find_max_in(plain_text ,b_chars, i_chars):
            chars = "".join(b_chars + i_chars)
            while chars and chars not in plain_text:
                chars = chars[:-1]
            return chars
        b_header_chars = re.findall(r"([\w\W]):B-HEADER", scope_text)
        i_header_chars = re.findall(r"([\w\W]):I-HEADER", scope_text)
        b_value_chars = re.findall(r"([\w\W]):B-VALUE", scope_text)
        i_value_chars = re.findall(r"([\w\W]):I-VALUE", scope_text)
        if len(b_header_chars) != 1 or len(b_value_chars) != 1:
            return None
        plain_text = produce_plain_text(scope_text)
        header = find_max_in(plain_text, b_header_chars, i_header_chars)
        value = find_max_in(plain_text, b_value_chars, i_value_chars)
        if (not header) or (not value):
            return None
        return (header, value)
    def find_scope(text):
        start_index = text.find("[")
        end_index = text.rfind("]")
        if start_index == -1 or end_index == -1:
            return text
        scope_text = text[start_index: end_index + 1]
        res_text = filter_relation(text.replace(scope_text, "")).replace(" ", "").strip()
        return (scope_text, res_text)
    def produce_all_attribute_remove(req):
        if not req:
            return None
        string_or_t2 = find_scope(req[-1][0])
        assert type(string_or_t2) in [type(""), type((1,))]
        if type(string_or_t2) == type(""):
            return string_or_t2
        else:
            return string_or_t2[-1]
    def extract_all_attribute(req):
        extract_list = list(map(lambda t2: (t2[0][0], t2[1], t2[0][1]) ,
                                filter(lambda x: x[0] ,
                                   map(lambda tt2_t2: (extract_scope(tt2_t2[0][0]), tt2_t2[1]) ,
                                       filter(lambda t2_t2: "HEADER" in t2_t2[0][0] and "VALUE" in t2_t2[0][0] ,
                                           filter(lambda string_or_t2_t2: type(string_or_t2_t2[0]) == type((1,)), 
                                                map(lambda tttt2: (find_scope(tttt2[0]), tttt2[1]), 
                                                    req)))))))
        return extract_list
    def extract_attributes_relation_string(plain_text, all_attributes, res):
        if not all_attributes:
            return plain_text.replace(res if res else "", "")
        def replace_by_one_l_r(text ,t3):
            l, _, r = t3
            ##### produce multi l, r to satisfy string contrain problem 
            l0, l1 = l, l
            r0, r1 = r, r
            while l0 and l0 not in text:
                l0 = l0[:-1]
            while l1 and l1 not in text:
                l1 = l1[1:]
            while r0 and r0 not in text:
                r0 = r0[:-1]
            while r1 and r1 not in text:
                r1 = r1[1:]
            if not l or not r:
                return text

            conclusion = set([])
            for l_, r_ in product([l0, l1], [r0, r1]):
                l_r_conclusion = re.findall("({}.*?{})".format(l_, r_), text)
                r_l_conclusion = re.findall("({}.*?{})".format(r_, l_), text)
                conclusion = conclusion.union(set(l_r_conclusion + r_l_conclusion))
            
            ##### because use produce multi must choose the shortest elements from them
            ## to prevent "relation word" also be replaced.
            conclusion_filtered = find_min_commmon_strings(conclusion)

            conclusion = conclusion_filtered
            req_text = text
            for c in conclusion:
                req_text = req_text.replace(c, "")
            return req_text
        req_text_ = plain_text
        for t3 in all_attributes:
            req_text_ = replace_by_one_l_r(req_text_, t3)
        return req_text_.replace(res, "")
    req = []
    t2 = predict_single_sent(question)
    req.append(t2)
    while "[" in t2[0]:
        scope = find_scope(t2[0])
        if type(scope) == type(""):
            break
        else:
            assert type(scope) == type((1,))
            scope_text, res_text = scope
            #ic(req)
            t2 = predict_single_sent(res_text)
            req.append(t2)
    req = list(filter(lambda tt2: "HEADER" in tt2[0] and "VALUE" in tt2[0] , req))
    res = produce_all_attribute_remove(req)
    #ic(req)
    all_attributes = extract_all_attribute(req)
    # plain_text = produce_plain_text(scope_text)
    
    return all_attributes, res, extract_attributes_relation_string(produce_plain_text(req[0][0] if req else ""), all_attributes, res)


def rec_more_time(decomp):
    assert type(decomp) == type((1,)) and len(decomp) == 3
    assert not decomp[0]
    res, relation_string = decomp[1:]
    new_decomp = recurrent_extract(relation_string)
    #### stop if rec not help by new_decomp[1] != decomp[1]
    if not new_decomp[0] and new_decomp[1] != decomp[1]:
        return rec_more_time(new_decomp)
    return (new_decomp[0], res, new_decomp[1])
### one sent conds decomp end


##### data source start
train_path = "../TableQA/TableQA/train"
def data_loader(table_json_path = os.path.join(train_path ,"train.tables.json"),
                json_path = os.path.join(train_path ,"train.json"),
                req_table_num = 1):
    assert os.path.exists(table_json_path)
    assert os.path.exists(json_path)
    json_df = pd.read_json(json_path, lines = True)
    all_tables = pd.read_json(table_json_path, lines = True)
    if req_table_num is not None:
        assert type(req_table_num) == type(0) and req_table_num > 0 and req_table_num <= all_tables.shape[0]
    else:
        req_table_num = all_tables.shape[0]
    for i in range(req_table_num):
        #one_table = all_tables.iloc[i]["table"]
        #one_table_df = pd.read_sql("select * from `{}`".format(one_table), train_tables_dump_engine)
        one_table_s = all_tables.iloc[i]
        one_table_df = pd.DataFrame(one_table_s["rows"], columns = one_table_s["header"])
        yield one_table_df, json_df[json_df["table_id"] == one_table_s["id"]]
## data source end


###### string toolkit start 
def findMaxSubString(str1, str2):
    """
    """
    maxSub = 0
    maxSubString = ""
 
    str1_len = len(str1)
    str2_len = len(str2)
 
    for i in range(str1_len): 
        str1_pos = i
        for j in range(str2_len):
            str2_pos = j
            str1_pos = i
            if str1[str1_pos] != str2[str2_pos]:
                continue
            else:
                while (str1_pos < str1_len) and (str2_pos < str2_len):
                    if str1[str1_pos] == str2[str2_pos]:
                        str1_pos = str1_pos + 1
                        str2_pos = str2_pos + 1
                    else:
                        break
 
                sub_len = str2_pos - j 
                if maxSub < sub_len:
                    maxSub = sub_len
                    maxSubString = str2[j:str2_pos]
    return maxSubString


def find_min_commmon_strings(c):
    ##### {"jack", "ja", "ss", "sss", "ps", ""} -> {"ja", "ss", "ps"}
    common_strings = list(filter(lambda x: type(x) == type("") ,
                                map(lambda t2: t2[0] 
                                    if t2[0] in t2[1] 
                                        else (t2[1] 
                                            if t2[1] in t2[0] 
                                                else (t2[0], t2[1])),combinations(c, 2))))
    req = set([])
    while c:
        ele = c.pop()
        if all(map(lambda cc: cc not in ele, common_strings)):
            req.add(ele)
    req = req.union(set(common_strings))
    return set(filter(lambda x: x, req))
## string toolkit end 



###### datetime column match start 
#### only use object dtype to extract
def time_template_extractor(rows_filtered, pattern = u"[年月\.\-\d]+"):
    #re_words = re.compile(u"[年月\.\-\d]+")
    re_words = re.compile(pattern)
    nest_collection = pd.DataFrame(rows_filtered).applymap(lambda x: tuple(sorted(list(re.findall(re_words, x))))).values.tolist()
    def flatten_collection(c):
        if not c:
            return c
        if type(c[0]) == type(""):
            return c
        else:
            c = list(c)
        return flatten_collection(reduce(lambda a, b: a + b, map(list ,c)))
    return flatten_collection(nest_collection)

###@@ pattern_list
#pattern_list = [u"[年月\.\-\d]+", u"[年月\d]+", u"[年个月\d]+", u"[年月日\d]+"]

def justify_column_as_datetime(df, threshold = 0.8, time_template_extractor = lambda x: x):
    object_columns = list(map(lambda tt2: tt2[0] ,filter(lambda t2: t2[1].name == "object" ,dict(df.dtypes).items())))
    time_columns = []
    for col in object_columns:
        input_ = df[[col]].applymap(lambda x: "~" if type(x) != type("") else x)
        output_ = time_template_extractor(input_.values.tolist())
        input_ = input_.iloc[:, 0].values.tolist()
        time_evidence_cnt = sum(map(lambda t2: t2[0].strip() == t2[1].strip() and t2[0] and t2[1] and t2[0] != "~" and t2[1] != "~",zip(input_, output_)))
        if time_evidence_cnt > 0 and time_evidence_cnt / df.shape[0] >= threshold:
            #### use evidence ratio because may have some noise in data
            time_columns.append(col)
    return time_columns

def justify_column_as_datetime_reduce(df, threshold = 0.8, time_template_extractor_list = list(map(lambda p: partial(time_template_extractor, pattern = p), pattern_list))):
    return sorted(reduce(lambda a, b: a.union(b) ,map(lambda func: set(justify_column_as_datetime(df, threshold, func)), time_template_extractor_list)))
## datetime column match end

##### choose question column have a reduce function call below (choose_res_by_kws)
##### this is a tiny first version
###@@ time_kws = ("什么时候", "时间", "时候")
#time_kws = ("什么时候", "时间", "时候")
##### 
def choose_question_column(decomp, header, df):
    assert type(decomp) == type((1,)) and type(header) == type([])
    
    time_columns = justify_column_as_datetime_reduce(df)
    _, res, _ = decomp
    
    if type(res) != type(""):
        return None
    
    #ic(res)
    ##### should add time kws to it.
    #time_kws = ("什么时候", "时间", "时候")
    if any(map(lambda t_kw: t_kw in res, time_kws)):
        if len(time_columns) == 1:
            return time_columns[0]
        else:
            '''
            return sorted(map(lambda t_col: (t_col ,len(findMaxSubString(t_col, res)) / len(t_col)), time_columns),
                  key = lambda t2: t2[1])[::-1][0][0]
            '''
            sort_list = sorted(map(lambda t_col: (t_col ,len(findMaxSubString(t_col, res)) / len(t_col)), time_columns),
                  key = lambda t2: t2[1])[::-1]
            if sort_list:
                if sort_list[0]:
                    return sort_list[0][0]
            return None
    
    c_res_common_dict = dict(filter(lambda t2: t2[1] ,map(lambda c: (c ,findMaxSubString(c, res)), header)))
    common_ratio_c_dict = dict(map(lambda t2: (t2[0], len(t2[1]) / len(t2[0])), c_res_common_dict.items()))
    common_ratio_res_dict = dict(map(lambda t2: (t2[0], len(t2[1]) / len(res)), c_res_common_dict.items()))
    #ic(decomp)
    #ic(common_ratio_c_dict)
    #ic(common_ratio_res_dict)
    
    if not common_ratio_c_dict or not common_ratio_res_dict:
        return None
    
    dict_0_max_key = sorted(common_ratio_c_dict.items(), key = lambda t2: t2[1])[::-1][0][0]
    dict_1_max_key = sorted(common_ratio_res_dict.items(), key = lambda t2: t2[1])[::-1][0][0]
    return dict_0_max_key if dict_0_max_key == dict_1_max_key else None


##### agg-classifier start
'''
sum_count_high_kws = ('多少个', '有几个', '总共') + ('总和','一共',) + ("总数",)
mean_kws = ('平均数', '均值', '平均值', '平均')
max_kws = ('最大', '最多', '最大值', '最高')
min_kws = ('最少', '最小值', '最小', '最低')
sum_count_low_kws = ('个', '总共') + ('总和','加','总','一共','和',) + ("哪些", "查", "数量", "数") + ("几",) + ('多少', "多大")  + ("总数",)
max_special_kws = ("以上", "大于")
min_special_kws = ("以下", "小于")
'''

###@@ sum_count_high_kws = ('多少个', '有几个', '总共') + ('总和','一共',) + ("总数",)
###@@ mean_kws = ('平均数', '均值', '平均值', '平均')
###@@ max_kws = ('最大', '最多', '最大值', '最高')
###@@ min_kws = ('最少', '最小值', '最小', '最低')
###@@ sum_count_low_kws = ('个', '总共') + ('总和','加','总','一共','和',) + ("哪些", "查", "数量", "数") + ("几",) + ('多少', "多大")  + ("总数",)
###@@ max_special_kws = ("以上", "大于")
###@@ min_special_kws = ("以下", "小于")

def simple_label_func(s, drop_header = True):
    text_tokens =s.question_cut
    header = list(map(lambda x: x[:x.find("(")] if (not x.startswith("(") and x.endswith(")")) else x ,s.header.split(",")))
    
    #### not contain samples may not match in fuzzy-match, special column mapping in finance,
    ### or "3" to "三"
    '''
    fit_collection = ('多少个', '有几个', '总共') + ('总和','一共',) + ('平均数', '均值', '平均值', '平均') +     ('最大', '最多', '最大值', '最高') + ('最少', '最小值', '最小', '最低')
    
    '''
    fit_collection = sum_count_high_kws + mean_kws + max_kws + min_kws
    fit_header = []
    for c in header:
        for kw in fit_collection:
            if kw in c:
                start_idx = c.find(kw)
                end_idx = start_idx + len(kw)
                fit_header.append(c[start_idx: end_idx])
                
    if not drop_header:
        header = []
        fit_header = []
                
    input_ = "".join(text_tokens)
    for c in header + fit_header:
        if c in fit_collection:
            continue
        input_ = input_.replace(c, "")
        c0, c1 = c, c
        while c0 and c0 not in fit_collection and len(c0) >= 4:
            c0 = c0[1:]
            if c0 in fit_collection:
                break
            input_ = input_.replace(c0, "")
        while c1 and c1 not in fit_collection and len(c1) >= 4:
            c1 = c1[:-1]
            if c1 in fit_collection:
                break
            input_ = input_.replace(c1, "")
    
    #ic(input_)
    text_tokens = list(jieba.cut(input_))
    
    #cat_6_collection_high_level = ('多少个', '有几个', '总共') + ('总和','一共',) + ("哪些", "查", "数量")
    #cat_6_collection_high_level = ('多少个', '有几个', '总共') + ('总和','一共',)
    ##### 高置信度部分 （作为是否构成使用特殊规则的判断标准）
    #### case 2 部分 （高置信度有效匹配）
    #cat_6_collection_high_level =  ('多少个', '有几个', '总共') + ('总和','一共',)
    #cat_6_collection_high_level =  ('多少个', '有几个', '总共') + ('总和','一共',) + ("总数",)
    cat_6_collection_high_level = sum_count_high_kws
    if any(map(lambda high_level_token: high_level_token in "".join(text_tokens), cat_6_collection_high_level)):
        return 6
    
    #### 够深 够宽 规则部分, change order by header, if header have kws in , lower order
    if any(map(lambda kw: kw in text_tokens, mean_kws)):
        return 1
    if any(map(lambda kw: kw in text_tokens, max_kws)):
        return 2
    if any(map(lambda kw: kw in text_tokens, min_kws)):
        return 3
    
    ##### 低置信度部分
    #### case 2 部分  (低置信度尾部匹配)
    cat_6_collection = sum_count_low_kws
    if any(map(lambda kw: kw in text_tokens, cat_6_collection)):
        return 6
    if any(map(lambda token: "几" in token, text_tokens)):
        return 6
    
    #### special case 部分
    if any(map(lambda kw: kw in text_tokens, max_special_kws)):
        return 2
    if any(map(lambda kw: kw in text_tokens, min_special_kws)):
        return 3
    
    #### 无效匹配
    return 0


def simple_special_func(s, drop_header = True):
    text_tokens =s.question_cut
    header = list(map(lambda x: x[:x.find("(")] if (not x.startswith("(") and x.endswith(")")) else x ,s.header.split(",")))
    
    #### not contain samples may not match in fuzzy-match, special column mapping in finance,
    ### or "3" to "三"
    fit_collection = sum_count_high_kws + mean_kws + max_kws + min_kws
    fit_header = []
    for c in header:
        for kw in fit_collection:
            if kw in c:
                start_idx = c.find(kw)
                end_idx = start_idx + len(kw)
                fit_header.append(c[start_idx: end_idx])
                
    input_ = "".join(text_tokens)
    if not drop_header:
        header = []
        fit_header = []
    
    for c in header + fit_header:
        if c in fit_collection:
            continue
        input_ = input_.replace(c, "")
        c0, c1 = c, c
        while c0 and c0 not in fit_collection and len(c0) >= 4:
            c0 = c0[1:]
            if c0 in fit_collection:
                break
            input_ = input_.replace(c0, "")
        while c1 and c1 not in fit_collection and len(c1) >= 4:
            c1 = c1[:-1]
            if c1 in fit_collection:
                break
            input_ = input_.replace(c1, "")
    
    #ic(input_)
    text_tokens = list(jieba.cut(input_))
    #ic(text_tokens)
    
    #cat_6_collection_high_level = ('多少个', '有几个', '总共') + ('总和','一共',) + ("哪些", "查", "数量")
    #cat_6_collection_high_level = ('多少个', '有几个', '总共') + ('总和','一共',)
    #### case 2 部分 （高置信度有效匹配）
    cat_6_collection_high_level =  sum_count_high_kws
    if any(map(lambda high_level_token: high_level_token in "".join(text_tokens), cat_6_collection_high_level)):
        return 6
    
    #### 够深 够宽 规则部分, change order by header, if header have kws in , lower order
    if any(map(lambda kw: kw in text_tokens, mean_kws)):
        return 1
    if any(map(lambda kw: kw in text_tokens, max_kws)):
        return 2
    if any(map(lambda kw: kw in text_tokens, min_kws)):
        return 3
    
    return 0


def simple_total_label_func(s):
    is_special = False if simple_special_func(s) == 0 else True
    if not is_special:
        return 0
    return simple_label_func(s)
## agg-classifier end


##### main block of process start
def split_by_cond(s, extract_return = True):
    def recurrent_extract_cond(text):
        #return np.asarray(recurrent_extract(text)[0])
        #return recurrent_extract(text)[0]
        return np.asarray(list(recurrent_extract(text)[0]))
        
    question = s["question"]
    res =  s["rec_decomp"][1]
    if question is None:
        question = ""
    if res is None:
        res = ""
    
    common_res = findMaxSubString(question, res)
    #cond_kws = ("或", "而且", "并且", "当中")
    #cond_kws = ("或", "而且" "并且" "当中")
    cond_kws = conn_kws
    condition_part = question.replace(common_res, "")
    fit_kws = set([])
    for kw in cond_kws:
        if kw in condition_part and not condition_part.startswith(kw) and not condition_part.endswith(kw):
            fit_kws.add(kw)
    if not fit_kws:
        will_return = ([condition_part.replace(" ", "") + " " + common_res], "")
        if extract_return:
            #return (list(map(recurrent_extract_cond, will_return[0])), will_return[1])
            will_return = (np.asarray(list(map(recurrent_extract_cond, will_return[0]))) , will_return[1])
            #will_return = (np.concatenate(list(filter(lambda x: x.size ,map(np.asarray ,will_return[0].tolist()))), axis = 0), will_return[1])
            #will_return = (np.concatenate(list(map(np.asarray ,will_return[0].tolist())), axis = 0), will_return[1])
            input_ = list(filter(lambda x: x.size ,map(np.asarray ,will_return[0].tolist())))
            if input_:
                will_return = (np.concatenate(input_, axis = 0), will_return[1])
            else:
                will_return = (np.empty((0, 3)), will_return[1])
            
            will_return = will_return[0].reshape((-1, 3)) if will_return[0].size else np.empty((0, 3))
        return will_return
        
    fit_kw = sorted(fit_kws, key = len)[::-1][0]
    condition_part_splits = condition_part.split(fit_kw)
    #if fit_kw in ("或",):
    if fit_kw in or_kws:
        fit_kw = "or"
    #elif fit_kw in ("而且", "并且", "当中",):
    elif fit_kw in and_kws:
        fit_kw = "and"
    else:
        fit_kw = ""
    
    will_return = (list(map(lambda cond_: cond_.replace(" ", "") + " " + common_res, condition_part_splits)), fit_kw)
    if extract_return:
        #return (list(map(recurrent_extract_cond, will_return[0])), will_return[1])
        will_return = (np.asarray(list(map(recurrent_extract_cond, will_return[0]))), will_return[1])
        #ic(will_return[0])
        #will_return = (np.concatenate(list(map(np.asarray ,will_return[0].tolist())), axis = 0), will_return[1])
        input_ = list(filter(lambda x: x.size ,map(np.asarray ,will_return[0].tolist())))
        if input_:
            will_return = (np.concatenate(input_, axis = 0), will_return[1])
        else:
            will_return = (np.empty((0, 3)), will_return[1])
        #ic(will_return[0])
        will_return = will_return[0].reshape((-1, 3)) if will_return[0].size else np.empty((0, 3))
    
    return will_return



def filter_total_conds(s, df, condition_fit_num = 0):
    assert condition_fit_num >= 0 and type(condition_fit_num) == type(0)
    df = df.copy()
    
    #### some col not as float with only "None" as cell, also transform them into float
    df = df.applymap(lambda x: np.nan if x in ["None", None, "/"] else x)
    def justify_column_as_float(s):
        if "float" in str(s.dtype):
            return True
        if all(s.map(type).map(lambda tx: "float" in str(tx))):
            return True
        return False
    
    float_cols = list(map(lambda tt2: tt2[0],filter(lambda t2: t2[1] ,df.apply(justify_column_as_float, axis = 0).to_dict().items())))
    for f_col in float_cols:
        df[f_col] = df[f_col].astype(np.float64)
    ###
    
    header = df.columns.tolist()
    units_cols = filter(lambda c: "(" in c and c.endswith(")"), df.columns.tolist())
    if not float_cols:
        float_discribe_df = pd.DataFrame()
    else:
        float_discribe_df = df[float_cols].describe()
    
    def call_eval(val):
        try:
            return literal_eval(val)
        except:
            return val
    
    #### find condition column same as question_column
    def find_cond_col(res, header):
        #ic(res, header)
        c_res_common_dict = dict(filter(lambda t2: t2[1] ,map(lambda c: (c ,findMaxSubString(c, res)), header)))
        #ic(c_res_common_dict)
        common_ratio_c_dict = dict(map(lambda t2: (t2[0], len(t2[1]) / len(t2[0])), c_res_common_dict.items()))
        common_ratio_res_dict = dict(map(lambda t2: (t2[0], len(t2[1]) / len(res)), c_res_common_dict.items()))
        
        if not common_ratio_c_dict or not common_ratio_res_dict:
            return None
        
        dict_0_max_key = sorted(common_ratio_c_dict.items(), key = lambda t2: t2[1])[::-1][0][0]
        dict_1_max_key = sorted(common_ratio_res_dict.items(), key = lambda t2: t2[1])[::-1][0][0]
        return dict_0_max_key if dict_0_max_key == dict_1_max_key else None
    ###
    
    #### type comptatible in column type and value type, and fit_num match
    def filter_cond_col(cond_t3):
        assert type(cond_t3) == type((1,)) and len(cond_t3) == 3
        col, _, value = cond_t3

        if type(value) == type(""):
            value = call_eval(value)
        
        if col not in df.columns.tolist():
            return False
        
        #### type key value comp
        if col in float_cols and type(value) not in (type(0), type(0.0)):
            return False
        
        if col not in float_cols and type(value) in (type(0), type(0.0)):
            return False
        
        #### string value not in corr column
        if col not in float_cols and type(value) not in (type(0), type(0.0)):
            if type(value) == type(""):
                if value not in df[col].tolist():
                    return False
    
        if type(value) in (type(0), type(0.0)):
            if col in float_discribe_df.columns.tolist():
                if condition_fit_num > 0:
                    if value >= float_discribe_df[col].loc["min"] and value <= float_discribe_df[col].loc["max"]:
                        return True
                    else:
                        return False
                else:
                    assert condition_fit_num == 0
                    return True

        if condition_fit_num > 0:
            if value in df[col].tolist():
                return True
            else:
                return False
        else:
            assert condition_fit_num == 0 
            return True
        
        return True
    ###
    
    #### condtions with same column may have conflict, choose the nearest one by stats in float or 
    ### common_string as find_cond_col do.
    def same_column_cond_filter(cond_list, sort_stats = "mean"):
        #ic(cond_list)
        if len(cond_list) == len(set(map(lambda t3: t3[0] ,cond_list))):
            return cond_list
        
        req = defaultdict(list)
        for t3 in cond_list:
            req[t3[0]].append(t3[1:])
        
        def t2_list_sort(col_name ,t2_list):
            if not t2_list:
                return None
            t2 = None
            if col_name in float_cols:
                t2 = sorted(t2_list, key = lambda t2: np.abs(t2[1] - float_discribe_df[col_name].loc[sort_stats]))[0]
            else:
                if all(map(lambda t2: type(t2[1]) == type("") ,t2_list)):
                    col_val_cnt_df = df[col_name].value_counts().reset_index()
                    col_val_cnt_df.columns = ["val", "cnt"]
                    #col_val_cnt_df["val"].map(lambda x: sorted(filter(lambda tt2: tt2[-1] ,map(lambda t2: (t2 ,len(findMaxSubString(x, t2[1]))), t2_list)), 
                    #                                          key = lambda ttt2: -1 * ttt2[-1])[0])
                    
                    t2_list_map_to_column_val = list(filter(lambda x: x[1] ,map(lambda t2: (t2[0] ,find_cond_col(t2[1], list(set(col_val_cnt_df["val"].values.tolist())))), t2_list)))
                    if t2_list_map_to_column_val:
                        #### return max length fit val in column
                        t2 = sorted(t2_list_map_to_column_val, key = lambda t2: -1 * len(t2[1]))[0]
            if t2 is None and t2_list:
                t2 = t2_list[0]
            return t2
                    
        cond_list_filtered = list(map(lambda ttt2: (ttt2[0], ttt2[1][0], ttt2[1][1]) ,
            filter(lambda tt2: tt2[1] ,map(lambda t2: (t2[0] ,t2_list_sort(t2[0], t2[1])), req.items()))))
        
        return cond_list_filtered
    ###
    
    total_conds_map_to_column = list(map(lambda t3: (find_cond_col(t3[0], header), t3[1], t3[2]), s["total_conds"]))
    total_conds_map_to_column_filtered = list(filter(filter_cond_col, total_conds_map_to_column))
    total_conds_map_to_column_filtered = sorted(set(map(lambda t3: (t3[0], t3[1], call_eval(t3[2]) if type(t3[2]) == type("") else t3[2]), total_conds_map_to_column_filtered)))
    #ic(total_conds_map_to_column_filtered)
    
    cp_cond_list = list(filter(lambda t3: t3[1] in (">", "<"), total_conds_map_to_column_filtered))
    eq_cond_list = list(filter(lambda t3: t3[1] in ("==", "!="), total_conds_map_to_column_filtered))
    
    cp_cond_list_filtered = same_column_cond_filter(cp_cond_list)
    
    #total_conds_map_to_column_filtered = same_column_cond_filter(total_conds_map_to_column_filtered)
    return cp_cond_list_filtered + eq_cond_list
    #return total_conds_map_to_column_filtered

###@@ only_kws_columns = {"城市": "=="}

#### this function only use to cond can not extract by JointBert,
### may because not contain column string in question such as "城市" or difficult to extract kw
### define kw column as all cells in series are string type.
### this function support config relation to column and if future 
### want to auto extract relation, this may can be done by head string or tail string by edit pattern "\w?{}\w?"
### "是" or "不是" can be extract in this manner.
def augment_kw_in_question(question_df, df, only_kws_in_string = []):
    #### keep only_kws_in_string empty to maintain all condition
    question_df = question_df.copy()
    #df = df.copy()

    def call_eval(val):
        try:
            return literal_eval(val)
        except:
            return val
    
    df = df.copy()
    
    df = df.applymap(call_eval)
    
    #### some col not as float with only "None" as cell, also transform them into float
    df = df.applymap(lambda x: np.nan if x in ["None", None, "/"] else x)
    def justify_column_as_float(s):
        if "float" in str(s.dtype):
            return True
        if all(s.map(type).map(lambda tx: "float" in str(tx))):
            return True
        return False
    
    float_cols = list(map(lambda tt2: tt2[0],filter(lambda t2: t2[1] ,df.apply(justify_column_as_float, axis = 0).to_dict().items())))
    #obj_cols = set(df.columns.tolist()).difference(set(float_cols))
    
    def justify_column_as_kw(s):
        if all(s.map(type).map(lambda tx: "str" in str(tx))):
            return True
        return False
    
    obj_cols = list(map(lambda tt2: tt2[0],filter(lambda t2: t2[1] ,df.apply(justify_column_as_kw, axis = 0).to_dict().items())))
    obj_cols = list(set(obj_cols).difference(set(float_cols)))
    if only_kws_columns:
        obj_cols = list(set(obj_cols).intersection(set(only_kws_columns.keys())))
    
    #replace_format = "{}是{}"
    #kw_augmented_df = pd.DataFrame(df[obj_cols].apply(lambda s: list(map(lambda val :(val,replace_format.format(s.name, val)), s.tolist())), axis = 0).values.tolist())
    #kw_augmented_df.columns = obj_cols
    kw_augmented_df = df[obj_cols].copy()
    #ic(kw_augmented_df)
    
    def extract_question_kws(question):
        if not kw_augmented_df.size:
            return []
        req = defaultdict(set)
        for ridx, r in kw_augmented_df.iterrows():
            for k, v in dict(r).items():
                if v in question:
                    pattern = "\w?{}\w?".format(v)
                    all_match = re.findall(pattern, question)
                    #req = req.union(set(all_match))
                    #req[v] = req[v].union(set(all_match))
                    key = "{}~{}".format(k, v)
                    req[key] = req[key].union(set(all_match))
                #ic(k, v)
                #question = question.replace(v[0], v[1])
        #return question.replace(replace_format.format("","") * 2, replace_format.format("",""))
        #req = list(req)
        if only_kws_in_string:
            req = list(map(lambda tt2: tt2[0] ,filter(lambda t2: sum(map(lambda kw: sum(map(lambda t: kw in t ,t2[1])), only_kws_in_string)), req.items())))
        else:
            req = list(set(req.keys()))
        
        def req_to_t3(req_string, relation = "=="):
            assert "~" in req_string
            left, right = req_string.split("~")
            if left in only_kws_columns:
                relation = only_kws_columns[left]
            return (left, relation, right)
        
        if not req:
            return []
        
        return list(map(req_to_t3, req))
        
        #return req
    
    question_df["question_kw_conds"] = question_df["question"].map(extract_question_kws)
    return question_df


def choose_question_column_by_rm_conds(s, df):
    question = s.question
    total_conds_filtered = s.total_conds_filtered
    #cond_kws = ("或", "而且", "并且", "当中")
    cond_kws = conn_kws
    stopwords = ("是", )
    #ic(total_conds_filtered)
    def construct_res(question):
        for k, _, v in total_conds_filtered:
            if "(" in k:
                k = k[:k.find("(")]
                #ic(k)
            question = question.replace(str(k), "")
            question = question.replace(str(v), "")
        for w in cond_kws + stopwords:
            question = question.replace(w, "")
        return question
    
    res = construct_res(question)
    decomp = (None, res, None)
    return choose_question_column(decomp, df.columns.tolist(), df)


def split_qst_by_kw(question, kw = "的"):
    return question.split(kw)

#qst_kws = ("多少", "什么", "多大", "哪些", "怎么", "情况", "那些", "哪个")
###@@ qst_kws = ("多少", "什么", "多大", "哪些", "怎么", "情况", "那些", "哪个")
def choose_res_by_kws(question):
    #kws = ["的","多少", '是']
    question = question.replace(" ", "")
    #kws = ["的","或者","或", "且","并且","同时"]
    kws = ("的",) + conn_kws
    kws = list(kws)
    def qst_kw_filter(text):
        #qst_kws = ("多少", "什么", "多大", "哪些", "怎么", "情况", "那些", "哪个")
        if any(map(lambda kw: kw in text, qst_kws)):
            return True
        return False
    
    kws_cp = deepcopy(kws)
    qst_c = set(question.split("，"))
    while kws:
        kw = kws.pop()
        qst_c = qst_c.union(set(filter(qst_kw_filter ,reduce(lambda a, b: a + b,map(lambda q: split_qst_by_kw(q, kw), qst_c))))
                           )
    #print("-" * 10)
    #print(sorted(filter(lambda x: x and (x not in kws_cp) ,qst_c), key = len))
    #print(sorted(filter(lambda x: x and (x not in kws_cp) and qst_kw_filter(x) ,qst_c), key = len))
    #### final choose if or not
    return sorted(filter(lambda x: x and (x not in kws_cp) and qst_kw_filter(x) ,qst_c), key = len)
    #return sorted(filter(lambda x: x and (x not in kws_cp) and True ,qst_c), key = len)


def cat6_to_45_by_column_type(s, df):
    agg_pred = s.agg_pred
    if agg_pred != 6:
        return agg_pred
    question_column = s.question_column 
    
    def call_eval(val):
        try:
            return literal_eval(val)
        except:
            return val
    
    df = df.copy()
    
    df = df.applymap(call_eval)
    
    #### some col not as float with only "None" as cell, also transform them into float
    df = df.applymap(lambda x: np.nan if x in ["None", None, "/"] else x)
    def justify_column_as_float(s):
        if "float" in str(s.dtype):
            return True
        if all(s.map(type).map(lambda tx: "float" in str(tx))):
            return True
        return False
    
    float_cols = list(map(lambda tt2: tt2[0],filter(lambda t2: t2[1] ,df.apply(justify_column_as_float, axis = 0).to_dict().items())))
    #obj_cols = set(df.columns.tolist()).difference(set(float_cols))
    
    def justify_column_as_kw(s):
        if all(s.map(type).map(lambda tx: "str" in str(tx))):
            return True
        return False
    
    #obj_cols = list(map(lambda tt2: tt2[0],filter(lambda t2: t2[1] ,df.apply(justify_column_as_kw, axis = 0).to_dict().items())))
    obj_cols = df.columns.tolist()
    obj_cols = list(set(obj_cols).difference(set(float_cols)))
    
    #ic(obj_cols, float_cols, df.columns.tolist())
    assert len(obj_cols) + len(float_cols) == df.shape[1]
    
    if question_column in obj_cols:
        return 4
    elif question_column in float_cols:
        return 5
    else:
        return 0


def full_before_cat_decomp(df, question_df, only_req_columns = False):
    df, question_df = df.copy(), question_df.copy()
    first_train_question_extract_df = pd.DataFrame(question_df["question"].map(lambda question: (question, recurrent_extract(question))).tolist())
    first_train_question_extract_df.columns = ["question", "decomp"]
    
    first_train_question_extract_df = augment_kw_in_question(first_train_question_extract_df, df)
    
    first_train_question_extract_df["rec_decomp"] = first_train_question_extract_df["decomp"].map(lambda decomp: decomp if decomp[0] else rec_more_time(decomp))
    #return first_train_question_extract_df.copy()
    first_train_question_extract_df["question_cut"] = first_train_question_extract_df["rec_decomp"].map(lambda t3: jieba.cut(t3[1]) if t3[1] is not None else []).map(list)
    first_train_question_extract_df["header"] = ",".join(df.columns.tolist())
    first_train_question_extract_df["question_column_res"] = first_train_question_extract_df["rec_decomp"].map(lambda decomp: choose_question_column(decomp, df.columns.tolist(), df))
    
    #### agg
    first_train_question_extract_df["agg_res_pred"] = first_train_question_extract_df.apply(simple_total_label_func, axis = 1)
    first_train_question_extract_df["question_cut"] = first_train_question_extract_df["question"].map(jieba.cut).map(list)
    first_train_question_extract_df["agg_qst_pred"] = first_train_question_extract_df.apply(simple_total_label_func, axis = 1)
    ### if agg_res_pred and agg_qst_pred have conflict use max, to prevent from empty agg with errorous question column
    ### but this "max" can also be replaced by measure the performance of decomp part, and choose the best one
    ### or we can use agg_qst_pred with high balanced_score as 0.86+ in imbalanced dataset.
    ### which operation to use need some discussion.
    ### (balanced_accuracy_score(lookup_df["sql"], lookup_df["agg_pred"]), 
    ### balanced_accuracy_score(lookup_df["sql"], lookup_df["agg_res_pred"]),
    ### balanced_accuracy_score(lookup_df["sql"], lookup_df["agg_qst_pred"]))
    ### (0.9444444444444445, 0.861111111111111, 0.9444444444444445) first_train_df conclucion
    ### (1.0, 0.8333333333333333, 1.0) cat6_conclucion
    ### this show that res worse in cat6 situation, but the agg-classifier construct is sufficent to have a 
    ### good conclusion in full-question. (because cat6 is the most accurate part in Tupledire tree sense.) 
    ### so use max as the best one 
    first_train_question_extract_df["agg_pred"] =  first_train_question_extract_df.apply(lambda s: max(s.agg_res_pred, s.agg_qst_pred), axis = 1)
   
    #### conn and conds
    first_train_question_extract_df["conds"] = first_train_question_extract_df["rec_decomp"].map(lambda x: x[0])
    first_train_question_extract_df["split_conds"] = first_train_question_extract_df.apply(split_by_cond, axis = 1).values.tolist()
    first_train_question_extract_df["conn_pred"] = first_train_question_extract_df.apply(lambda s: split_by_cond(s, extract_return=False), axis = 1).map(lambda x: x[-1]).values.tolist()
    #first_train_question_extract_df["total_conds"] = first_train_question_extract_df.apply(lambda s: list(set(map(tuple,s["conds"] + s["split_conds"].tolist()))), axis = 1).values.tolist()
    first_train_question_extract_df["total_conds"] = first_train_question_extract_df.apply(lambda s: list(set(map(tuple,s["question_kw_conds"] + s["conds"] + s["split_conds"].tolist()))), axis = 1).values.tolist()
    first_train_question_extract_df["total_conds_filtered"] = first_train_question_extract_df.apply(lambda s: filter_total_conds(s, df), axis = 1).values.tolist()
    
    #### question_column_res more accurate, if not fit then use full-question question_column_qst to extract
    ### can not fit multi question or fuzzy describe, or question need kw replacement.
    
    #first_train_question_extract_df["question_column_qst"] = first_train_question_extract_df.apply(lambda s: choose_question_column_by_rm_conds(s, df), axis = 1)
    first_train_question_extract_df["question_column_qst"] = first_train_question_extract_df["question"].map(choose_res_by_kws).map(lambda res_list: list(filter(lambda x: x ,map(lambda res: choose_question_column((None, res, None), df.columns.tolist(), df), res_list)))).map(lambda x: x[0] if x else None)
    first_train_question_extract_df["question_column"] = first_train_question_extract_df.apply(lambda s: s.question_column_res if s.question_column_res is not None else s.question_column_qst, axis = 1)
    
    #### predict cat6 to 4 5 based on question_column and column dtype,
    #### this may performance bad if question_column has error,
    #### and almost 100% accurate if question_column truly provide and user is not an idoit (speak ....)
    agg_sql_dict = {0:"", 1:"AVG", 2:"MAX", 3:"MIN", 4:"COUNT", 5:"SUM"}
    first_train_question_extract_df["agg_pred"] =  first_train_question_extract_df.apply(lambda s: cat6_to_45_by_column_type(s, df), axis = 1).map(lambda x: agg_sql_dict[x])
    if only_req_columns:
        return first_train_question_extract_df[["question", 
                                                "total_conds_filtered", 
                                                "conn_pred",
                                                "question_column",
                                                "agg_pred"
                                               ]].copy()
    
    return first_train_question_extract_df.copy()


if __name__ == "__main__":
    ###### valid block
    req = list(data_loader(req_table_num=None))


    train_df, _ = req[2]
    train_df
    question = "哪些股票的收盘价大于20？"
    qs_df = pd.DataFrame([[question]], columns = ["question"])
    ic(question)
    ic(full_before_cat_decomp(train_df, qs_df, only_req_columns=True))


    #### not support select 股票 from table where 市值 = （select max(市值) from table）
    #### this is a nest sql.
    question = "哪个股票代码市值最高？"
    qs_df = pd.DataFrame([[question]], columns = ["question"])
    ic(question)
    ic(full_before_cat_decomp(train_df, qs_df, only_req_columns=True))

    question = "市值的最大值是多少？"
    qs_df = pd.DataFrame([[question]], columns = ["question"])
    ic(question)
    ic(full_before_cat_decomp(train_df, qs_df, only_req_columns=True))


    question = "EPS大于0的股票有哪些？"
    qs_df = pd.DataFrame([[question]], columns = ["question"])
    ic(question)
    ic(full_before_cat_decomp(train_df, qs_df, only_req_columns=True))


    question = "EPS大于0且周涨跌大于5的平均市值是多少？"
    qs_df = pd.DataFrame([[question]], columns = ["question"])
    ic(question)
    ic(full_before_cat_decomp(train_df, qs_df, only_req_columns=True))


    train_df, _ = req[5]
    train_df
    question = "产能小于20、销量大于40而且市场占有率超过1的公司有哪些？"
    qs_df = pd.DataFrame([[question]], columns = ["question"])
    ic(question)
    ic(full_before_cat_decomp(train_df, qs_df, only_req_columns=True))


    #### 特殊符号 "、"
    question = "产能小于20而且销量大于40而且市场占有率超过1的公司有哪些？"
    qs_df = pd.DataFrame([[question]], columns = ["question"])
    ic(question)
    ic(full_before_cat_decomp(train_df, qs_df, only_req_columns=True))


    train_df, _ = req[6]
    train_df
    #### 加入列别名 只需要 复刻列即可
    question = "投资评级为维持的名称有哪些？"
    qs_df = pd.DataFrame([[question]], columns = ["question"])
    ic(question)
    ic(full_before_cat_decomp(train_df, qs_df, only_req_columns=True))


    train_df["公司"] = train_df["名称"]
    question = "投资评级为维持的公司有哪些？"
    qs_df = pd.DataFrame([[question]], columns = ["question"])
    ic(question)
    ic(full_before_cat_decomp(train_df, qs_df, only_req_columns=True))


    question = "投资评级为维持而且变动为增持的公司有哪些？"
    qs_df = pd.DataFrame([[question]], columns = ["question"])
    ic(question)
    ic(full_before_cat_decomp(train_df, qs_df, only_req_columns=True))


    question = "投资评级为维持或者变动为增持的公司有哪些？"
    qs_df = pd.DataFrame([[question]], columns = ["question"])
    ic(question)
    ic(full_before_cat_decomp(train_df, qs_df, only_req_columns=True))


    question = "投资评级为维持或者变动为增持的平均收盘价是多少？"
    qs_df = pd.DataFrame([[question]], columns = ["question"])
    ic(question)
    ic(full_before_cat_decomp(train_df, qs_df, only_req_columns=True))


    train_df, _ = req[7]
    train_df
    question = "宁波的一手房每周交易数据上周成交量是多少？"
    qs_df = pd.DataFrame([[question]], columns = ["question"])
    ic(question)
    ic(full_before_cat_decomp(train_df, qs_df, only_req_columns=True))


    question = "一手房每周交易数据为宁波上周成交量是多少？"
    qs_df = pd.DataFrame([[question]], columns = ["question"])
    ic(question)
    ic(full_before_cat_decomp(train_df, qs_df, only_req_columns=True))

    #### this also can deal with set column as use kw to extract
    ### see function augment_kw_in_question
    train_df["城市"] = train_df["一手房每周交易数据"]
    question = "一手房每周交易数据为宁波上周成交量是多少？"
    qs_df = pd.DataFrame([[question]], columns = ["question"])
    ic(question)
    ic(full_before_cat_decomp(train_df, qs_df, only_req_columns=True))

    question = "王翔知道宁波一手房的当月累计成交量是多少吗？"
    qs_df = pd.DataFrame([[question]], columns = ["question"])
    ic(question)
    ic(full_before_cat_decomp(train_df, qs_df, only_req_columns=True))


    question = "王翔知道上周成交量大于50的最大同比当月是多少吗？"
    qs_df = pd.DataFrame([[question]], columns = ["question"])
    ic(question)
    ic(full_before_cat_decomp(train_df, qs_df, only_req_columns=True))


    train_df, _ = req[9]
    #### the last column should be "周跌幅", can't tackle duplicates columns
    train_df
    cols = train_df.columns.tolist()
    cols[-1] = "周跌幅（%）"
    train_df.columns = cols
    question = "周涨幅大于7的涨股有哪些？"
    qs_df = pd.DataFrame([[question]], columns = ["question"])
    ic(question)
    ic(full_before_cat_decomp(train_df, qs_df, only_req_columns=True))


    #### not recognize as 6 agg-classifier
    question = "周涨幅大于7的涨股总数是多少？"
    qs_df = pd.DataFrame([[question]], columns = ["question"])
    ic(question)
    ic(full_before_cat_decomp(train_df, qs_df, only_req_columns=True))


    question = "周涨幅大于7的涨股总共有多少个？"
    qs_df = pd.DataFrame([[question]], columns = ["question"])
    ic(question)
    ic(full_before_cat_decomp(train_df, qs_df, only_req_columns=True))


