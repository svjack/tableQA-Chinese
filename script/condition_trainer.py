#!/usr/bin/env python
# coding: utf-8

import ast
import json
import math
import os
import re
import shutil
import sys
from ast import literal_eval
from collections import namedtuple
from copy import deepcopy
from functools import reduce
from itertools import combinations, product

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from icecream import ic
from nltk import pos_tag, word_tokenize
from pyarrow.filesystem import LocalFileSystem
from torch import nn
from torch.nn import functional, init
from torch.nn.utils import rnn as rnn_utils

pd.set_option("display.max_rows", 100)

#### used in this condition extract in training.
op_sql_dict = {0:">", 1:"<", 2:"==", 3:"!="}
#### used by clf for intension inference
agg_sql_dict = {0:"", 1:"AVG", 2:"MAX", 3:"MIN", 4:"COUNT", 5:"SUM"}
#### final to combine them (one for 0, and multi for 1 2)
conn_sql_dict = {0:"", 1:"and", 2:"or"}


train_path = "../TableQA/TableQA/train"
val_path = "../TableQA/TableQA/val"

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


def sentence_t3_gen(df, q, 
                    header_process_func = lambda x: x[:x.find("(")] if "(" in x else (x[:x.find("（")] if "（" in x else x),
                   use_in_filter = False):
    ##### with some fuzzy string contain.
    headers = pd.Series(df.columns.tolist()).map(header_process_func).tolist()
    
    total_num = 0
    invalid_num = 0
    for idx, row in q.iterrows():
        question = row["question"]
        values_in_question = re.findall(r"[+-]*\d+\.?\d*", question)
        sql_obj = row["sql"]
        assert sum(map(lambda key: key in sql_obj, ['agg', 'cond_conn_op', 'sel', 'conds'])) == 4
        conds = sql_obj["conds"]
        assert type(conds) == type([])
        req = []
        #### pay more attention to pass
        for cond_t3 in conds:
            header_idx, cond_type_int, candidate = cond_t3
            assert cond_type_int in op_sql_dict.keys()
            #req.append((headers[header_idx], cond_type_int, candidate))
            #ele = (headers[header_idx], cond_type_int, candidate_2_to_close(candidate, values_in_question))
            ele = (headers[header_idx], cond_type_int, candidate)
            if use_in_filter:
                if ele[-1] in question and ele[0] in question:
                    req.append(ele)
            else:
                ele_max_head = findMaxSubString(deepcopy(question), deepcopy(ele[0]))
                ele = list(ele)
                if ele_max_head:
                    ele[0] = ele_max_head
                ele = tuple(ele)
                if ele[-1] and ele[0] and question and ele[-1] in question and ele[0] in question:
                    #if ele[-1] in question:
                    req.append(ele)
            
        if req:
            yield (question, req)
        else:
            invalid_num += 1        
        total_num += 1
        if total_num == q.shape[0] - 1:
            #ic("invalid_ratio ", invalid_num, total_num)
            pass


def explode_q_cond(question, req, min_q_len = 6):
    def string_clean_func(input_):
        return input_.replace(" ", "").strip()
    question = string_clean_func(question)
    req_keys = list(map(lambda x: x[0], req))
    
    if len(req_keys) > len(set(req_keys)):
        return [(question, req[0])]
    
    assert len(req_keys) == len(set(req_keys))
    if len(req_keys) == 1:
        return [(question, req[0])]
    def split_func(q, token):
        return q[:q.find(token)].strip(), q[q.find(token):].strip()
    full_question = question
    nest_list = []
    for idx ,token in enumerate(req_keys[1:]):
        previous_question ,full_question = split_func(full_question, token)
        nest_list.append((previous_question, req[idx]))
    if full_question:
        nest_list.append((full_question, req[len(req_keys) - 1]))
    #### filter out some not valid q
    nest_list = list(filter(lambda x: len(x[0]) >= min_q_len, nest_list))
    return nest_list

def all_t3_iter(data_loader_ext = data_loader(req_table_num = None), times = 5000):
    cnt = 0
    for idx ,(zh_df, zh_q) in enumerate(data_loader_ext):
        s_t3_iter = sentence_t3_gen(zh_df, zh_q)
        for q, req in s_t3_iter:
            for question, req_ele in explode_q_cond(q, req):
                yield (question, req_ele)
                cnt += 1
                if cnt >= times:
                    return
        if idx > 0 and idx % 50 == 0:
            ic("table gen ", idx, cnt)


def q_t3_writer(json_path, q_t3_iter):
    if os.path.exists(json_path):
        os.remove(json_path)
    with open(json_path, "w", encoding = "utf-8") as f:
        for question, cond_t3 in q_t3_iter:
            f.write(json.dumps({
                "question": question,
                "cond_t3": list(cond_t3)
            }) + "\n")


def labeling(question, req_ele, sep_sig = "*", left_slot_flag = None, right_slot_flog = None):
    assert type(question) == type("") and type(req_ele) == type([1,]) and len(req_ele) == 3
    assert type(left_slot_flag) == type("") and type(right_slot_flog) == type("")
    #### replace this for split
    question = question.replace(sep_sig, "")
    question_sep_cat = sep_sig.join(list(question))
    def produce_BI_replacement(cond_str, slot_flag):
        cond_str_sep_cat = sep_sig.join(list(cond_str))
        cond_str_sep_cat_replacement = sep_sig.join(map(lambda idx: "B-{}".format(slot_flag) if idx == 0 else "I-{}".format(slot_flag), range(len(cond_str))))
        return (cond_str_sep_cat, cond_str_sep_cat_replacement)
    left_cond = req_ele[0]
    left_cond_str_sep_cat, left_cond_str_sep_cat_replacement = produce_BI_replacement(left_cond, left_slot_flag)
    right_cond = req_ele[-1]
    right_cond_str_sep_cat, right_cond_str_sep_cat_replacement = produce_BI_replacement(right_cond, right_slot_flog)
    question_sep_cat_trans = question_sep_cat.replace(left_cond_str_sep_cat, left_cond_str_sep_cat_replacement).replace(right_cond_str_sep_cat, right_cond_str_sep_cat_replacement)
    tag_list = list(map(lambda tag: tag if tag.startswith("B-") or tag.startswith("I-") else "O",question_sep_cat_trans.split(sep_sig)))
    token_list = question_sep_cat.split(sep_sig)
    assert len(token_list) == len(tag_list)
    return (token_list, tag_list, req_ele[1])


q_t3_writer("q_t3_train_2.json", all_t3_iter(times = int(1e10)))
q_t3_writer("q_t3_val.json", all_t3_iter(data_loader_ext = data_loader(
    os.path.join(val_path ,"val.tables.json"), os.path.join(val_path ,"val.json")
    ,req_table_num = None),
                                                                                      times = int(1e10)))


conds_path = "conds"
def release_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.mkdir(dir_path)

def df_split(input_df, train_ratio = 0.9):
    req = input_df.sample(frac = 1.0)
    train_num = int(req.shape[0] * train_ratio)
    train_df = req.iloc[:train_num]
    others = req.iloc[train_num:].copy()
    dev_num = int(others.shape[0] / 2)
    dev_df = others.iloc[:dev_num]
    test_df = others.iloc[dev_num:]
    return (train_df, dev_df, test_df)


def dump_dfs_to_dir(input_df, conds_path):
    release_dir(conds_path)
    assert os.path.exists(conds_path)
    intent_label = ["UNK"] + input_df["label"].unique().tolist()
    assert "O" in reduce(lambda a, b: a + b ,input_df["out"].tolist())
    slot_label = ["PAD", "UNK", "O"] + sorted(set(reduce(lambda a, b: a + b ,input_df["out"].tolist())).difference(set(["O"])))
    in_out_train_path = os.path.join(conds_path, "train")
    in_out_dev_path = os.path.join(conds_path, "dev")
    in_out_test_path = os.path.join(conds_path, "test")
    release_dir(in_out_train_path)
    release_dir(in_out_dev_path)
    release_dir(in_out_test_path)
    def write_df_to_path(write_df, path):
        with open(os.path.join(path, "seq.in"), "w") as f:
            f.write("\n".join(write_df["in"].map(lambda x: " ".join(filter(lambda yy: yy ,map(lambda y: y.strip() ,x)))).tolist()))
        with open(os.path.join(path, "seq.out"), "w") as f:
            f.write("\n".join(write_df["out"].map(lambda x: " ".join(filter(lambda yy: yy ,map(lambda y: y.strip() ,x)))).tolist()))
        with open(os.path.join(path, "label"), "w") as f:
            f.write("\n".join(write_df["label"].map(lambda x: x.strip()).tolist()))
    train_df, dev_df, test_df = df_split(input_df)
    write_df_to_path(train_df, in_out_train_path)
    write_df_to_path(dev_df, in_out_dev_path)
    write_df_to_path(test_df, in_out_test_path)
    with open(os.path.join(conds_path, "intent_label.txt"), "w") as f:
        f.write("\n".join(intent_label))
    with open(os.path.join(conds_path, "slot_label.txt"), "w") as f:
        f.write("\n".join(slot_label))


q_t3_train_df = pd.read_json("q_t3_train_2.json", lines = True)
q_t3_train_df["labeling"] = q_t3_train_df.apply(lambda s: labeling(s["question"], s["cond_t3"], left_slot_flag = "HEADER", right_slot_flog = "VALUE"), axis = 1)
q_t3_train_df["in"] = q_t3_train_df["labeling"].map(lambda x: x[0])
q_t3_train_df["out"] = q_t3_train_df["labeling"].map(lambda x: x[1])
q_t3_train_df["label"] = q_t3_train_df["labeling"].map(lambda x: op_sql_dict[x[2]])
q_t3_train_df_filtered = q_t3_train_df[q_t3_train_df["out"].map(lambda x: sum(map(lambda t: t in ['B-HEADER', 'I-HEADER', 'B-VALUE', 'I-VALUE', 'O'], x)) == len(x))]



q_t3_val_df = pd.read_json("q_t3_val.json", lines = True)
q_t3_val_df["labeling"] = q_t3_val_df.apply(lambda s: labeling(s["question"], s["cond_t3"], left_slot_flag = "HEADER", right_slot_flog = "VALUE"), axis = 1)
q_t3_val_df["in"] = q_t3_val_df["labeling"].map(lambda x: x[0])
q_t3_val_df["out"] = q_t3_val_df["labeling"].map(lambda x: x[1])
q_t3_val_df["label"] = q_t3_val_df["labeling"].map(lambda x: op_sql_dict[x[2]])
q_t3_val_df_filtered = q_t3_val_df[q_t3_val_df["out"].map(lambda x: sum(map(lambda t: t in ['B-HEADER', 'I-HEADER', 'B-VALUE', 'I-VALUE', 'O'], x)) == len(x))]
q_t3_val_df_filtered = q_t3_val_df_filtered[q_t3_val_df_filtered["out"].map(lambda x: list(filter(lambda y: y.startswith("B"), x))).map(len) == 2]

ic(q_t3_train_df_filtered.shape ,q_t3_val_df_filtered.shape)

q_t3_total_df = pd.concat([q_t3_train_df_filtered, q_t3_val_df_filtered], axis = 0)
q_t3_total_df["cond_t3"] = q_t3_total_df["cond_t3"].map(tuple)
q_t3_total_df["in"] = q_t3_total_df["in"].map(tuple)
q_t3_total_df["out"] = q_t3_total_df["out"].map(tuple)
dump_dfs_to_dir(q_t3_total_df ,conds_path)


jointbert_path = "../../featurize/JointBERT"
sys.path.append(jointbert_path)

from data_loader import *
from main import *
from model.modeling_jointbert import *
from model.modeling_jointbert import JointBERT
from trainer import *

class JointProcessor_DROP_SOME(JointProcessor):
    """Processor for the JointBERT data set """
    def _create_examples(self, texts, intents, slots, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, (text, intent, slot) in enumerate(zip(texts, intents, slots)):
            guid = "%s-%s" % (set_type, i)
            # 1. input_text
            words = text.split()  # Some are spaced twice
            # 2. intent
            intent_label = self.intent_labels.index(intent) if intent in self.intent_labels else self.intent_labels.index("UNK")
            # 3. slot
            slot_labels = []
            for s in slot.split():
                slot_labels.append(self.slot_labels.index(s) if s in self.slot_labels else self.slot_labels.index("UNK"))

            if len(words) == len(slot_labels):
                examples.append(InputExample(guid=guid, words=words, intent_label=intent_label, slot_labels=slot_labels))
        return examples


class Trainer_DROP_SOME(Trainer):
    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        
        dev_loss_list = []
        for epoch_num ,_ in enumerate(train_iterator):
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            #for step, batch in enumerate(epoch_iterator):
            for step, batch in enumerate(train_dataloader):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'intent_label_ids': batch[3],
                          'slot_labels_ids': batch[4]}
                '''
                inputs = {'input_ids': batch[0],
                  "input_pos_tag": batch[5],
                          'attention_mask': batch[1],
                          'intent_label_ids': batch[3],
                          'slot_labels_ids': batch[4],
                 }
                '''
                if self.args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2]
                outputs = self.model(**inputs)
                loss = outputs[0]

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        dev_results = self.evaluate("dev")
                        dev_loss_list.append(dev_results)

                    if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        self.save_model()

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break
            
            ic("epoch num ", epoch_num)
        
        with open("dev_loss.pkl", "wb") as f:
            import pickle as pkl
            pkl.dump(dev_loss_list, f)
        
        return global_step, tr_loss / global_step

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(self.args.model_dir)

        # Save training arguments together with the trained model
        #torch.save(self.args, os.path.join(self.args.model_dir, 'training_args.bin'))
        torch.save(
            dict(filter(lambda t2: not t2[0].startswith("_") ,trainer.args.__dict__.items())), 
                   os.path.join(self.args.model_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", self.args.model_dir)



parser = argparse.ArgumentParser()

parser.add_argument("--task", default=None, required=True, type=str, help="The name of the task to train")
parser.add_argument("--model_dir", default=None, required=True, type=str, help="Path to save, load model")
parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir")
parser.add_argument("--intent_label_file", default="intent_label.txt", type=str, help="Intent Label file")
parser.add_argument("--slot_label_file", default="slot_label.txt", type=str, help="Slot Label file")

parser.add_argument("--model_type", default="bert", type=str, help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))

parser.add_argument('--seed', type=int, default=1234, help="random seed for initialization")
parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size for training.")
parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size for evaluation.")
parser.add_argument("--max_seq_len", default=50, type=int, help="The maximum total input sequence length after tokenization.")
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--num_train_epochs", default=10.0, type=float, help="Total number of training epochs to perform.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")

parser.add_argument('--logging_steps', type=int, default=200, help="Log every X updates steps.")
parser.add_argument('--save_steps', type=int, default=200, help="Save checkpoint every X updates steps.")

parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

parser.add_argument("--ignore_index", default=0, type=int,
                        help='Specifies a target value that is ignored and does not contribute to the input gradient')

parser.add_argument('--slot_loss_coef', type=float, default=1.0, help='Coefficient for the slot loss.')

# CRF option
parser.add_argument("--use_crf", action="store_true", help="Whether to use CRF")
parser.add_argument("--slot_pad_label", default="PAD", type=str, help="Pad token for slot label pad (to be ignore when calculate loss)")


parser_config_dict = dict(map(lambda item:(item.option_strings[0].replace("--", ""), item.default) ,parser.__dict__["_actions"]))
parser_config_dict = dict(filter(lambda t2: t2[0] != "-h", parser_config_dict.items()))


parser_namedtuple = namedtuple("parser_config", parser_config_dict.keys())
for k, v in parser_config_dict.items():
    if type(v) == type(""):
        exec("parser_namedtuple.{}='{}'".format(k, v))
    else:
        exec("parser_namedtuple.{}={}".format(k, v))


MODEL_PATH_MAP_CP = deepcopy(MODEL_PATH_MAP)
MODEL_PATH_MAP_CP["bert"] = "bert-base-chinese"

parser_namedtuple.task = "conds"
parser_namedtuple.model_dir = "bert"
parser_namedtuple.data_dir = os.getcwd()
parser_namedtuple.model_name_or_path = MODEL_PATH_MAP_CP[parser_namedtuple.model_type]
args = parser_namedtuple

processors["conds"] = JointProcessor_DROP_SOME
tokenizer = load_tokenizer(args)
train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev")
test_dataset = load_and_cache_examples(args, tokenizer, mode="test")

trainer = Trainer_DROP_SOME(args, train_dataset, dev_dataset, test_dataset)
trainer.train()
