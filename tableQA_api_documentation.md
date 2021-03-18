<br />
<p align="center">
  <!--
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>
  -->
  
  <!--
  <h3 align="center">Best-README-Template</h3>
  <h3 align="center">Weighted-W2V-Senti</h3>
  -->
  <h1 align="center">tableQA-Chinese Api Documentation </h1>

  <p align="center">
    <br />
  </p>
</p>

<h2> tableQA components </h2> 

<br />

 <h3> <a href="https://github.com/svjack/tableQA-Chinese/blob/main/script/condition_trainer.py">
   condition_trainer.py
  </a></h3>

&ensp; &ensp; &ensp; Use [JointBERT](https://github.com/monologg/JointBERT) to train condition extraction task

<b>data_loader</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp;tableQA data provider, will yield data table with question and training data.

<b>findMaxSubString</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp;return common string above two text.

<b>sentence_t3_gen</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; function to extract condition tuple (with length 3) according question and training data. (because original training data have some noise in labeling,
findMaxSubString is used to deal with it)

<b>explode_q_cond</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; explode the sample from sentence_t3_gen

<b>all_t3_iter</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; yield all condition extraction training data.

<b>q_t3_writer</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; serialize all_t3_iter data to local

<b>labeling</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; transform condition extraction data into JointBERT NER format

<b>dump_dfs_to_dir</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; save labeling conclusion into JointBERT friendly format

<b>JointProcessor_DROP_SOME</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; JointProcessor in JointBERT

<b>Trainer_DROP_SOME</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; Trainer in JointBERT

<br />

 <h3> <a href="https://github.com/svjack/tableQA-Chinese/blob/main/notebook/agg-classifier.ipynb">
   agg-classifier.ipynb
  </a></h3>

&ensp; &ensp; &ensp; Notebook to explore a good rule for classify which aggregate keyword to use

<b>agg_classifier_data_loader</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; data loader for all useful feature and agg label 

<b>standlize_agg_column</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; keep identical agg label from agg label list and remove multi-agg-label samples

<b>transform_stats_to_df</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; give a token based summary dataframe with evidence score amoung different agg category

<b>kws_dict_after_dict</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; look up the dataframe produced by transform_stats_to_df to find some keywords for different agg category. this dict has a high recall about 99.8% to cover all situations.

<b>kws_tuple_key_dict</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; a key value swap format for kws_dict_after_dict. use it to generate a permutations of order rules to perform rule strategy backtest.

<b>different_rule_product_list</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; generate product of samples from kws_tuple_key_dict with identical agg number this collection with permutation in itertools final produce all sample points (like all strategy generate in [vectorbt strategy space](https://github.com/polakowo/vectorbt/blob/master/examples/MACDVolume.ipynb))

<b>kw_matcher_func_format</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; function string format use to generate single order rule labeling function definition (as a sample point———strategy in [finance strategy](https://en.wikipedia.org/wiki/Trading_strategy) [backtest](https://en.wikipedia.org/wiki/Backtesting))

<b>one_rule_if_else_generate</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; use different_rule_product_list and permutations to truly generate all order rule strategy parameter

<b>kw_matcher_func_list</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; all labeling functions generate by one_rule_if_else_generate i.e. the truly samples’s name in all strategy space.

<b>produce_L_train_iter</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; because the kw_matcher_func_list has length 2880, and train data
have about 40000, this is a big feature matrix. use iterator on [PandasParallelLFApplier](https://snorkel.readthedocs.io/en/v0.9.1/packages/_autosummary/labeling/snorkel.labeling.apply.dask.PandasParallelLFApplier.html) (supported by [dask](https://github.com/dask/dask)) to give parallel apply.

<b>reconstruct_data</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; combine conclusion of produce_L_train_iter in sparse format (many prediction of labeling functions are zero)

<b>acc_score_s</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; measure [balanced_accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html) above different column 
in L_train to choose some gold label functions (label functions to give relatively good score as a good rule strategy)

<b>gold_t5_list</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; reverse map gold strategy (a strategy sub set that can make [snorkel label model](https://snorkel.readthedocs.io/en/v0.9.3/packages/_autosummary/labeling/snorkel.labeling.LabelModel.html) converge) back to parameters. where t5 is a length with 5 tuple in order that prior elements have more superiority to return first in kw_matcher_func_format

<b>gold_rule_trie</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; overwrite common used [Dictire](https://github.com/sufyanAbbasi/dictrie) in nlp to Tupletrie to give a measure friendly format to have a macro-summary above different t5 in gold_t5_list. (like a well organized strategy space for visualize) 

<b>count_all_sub_lines</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; count how many branch in current node in gold_rule_trie

<b>count_all_sub_nodes</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; count how many sub nodes in current node in gold_rule_trie

<b>depth</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; depth of current node in gold_rule_trie

<b>count_distinct_sub_nodes</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; distinct count of count_all_sub_nodes

<b>full_sub_tree_cnt</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; if generate a worst rule_trie in current node with same depth and sub nodes with current node. how many nodes this worst tree will have. the definition of “worst” is the tree which leaves is the permutation of nodes this tree have. and step to step into higher. a node with high num of full_sub_tree_cnt and low num of count_sub_nodes will become a will sub-tree (branch), because the balance_accurate_score “clip” this sub-tree better.

<b>node_stats_summary</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; function to generate a summary by the use of above functions on
single node.

<b>sub_tree_stats_summary</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; function to use node_stats_summary in all first level node in gold_rule_trie_dict (the summary of this function perform like the [result table](https://polakowo.io/vectorbt/docs/returns/index.html) for strategy backtest in [vectorbt](https://github.com/polakowo/vectorbt), simple compare above this summary will give a advice about different order rules) with the discussion in the notebook the final rule is located in simple_label_func

<br />

 <h3> <a href="https://github.com/svjack/tableQA-Chinese/blob/main/script/tableQA_single_table.py">
  tableQA_single_table.py
  </a></h3>

&ensp; &ensp; &ensp; Give conclusion of single table with question input. use this as import module in other script or notebook can simply add or overwrite some *_kws dictionary or pattern_list to satisfy your own context. The usage of this script is located in [tableqa-single-valid.ipynb](https://github.com/svjack/tableQA-Chinese/blob/main/notebook/tableqa-single-valid.ipynb)

<b>*_kws dictionary</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; global dictionary used for aggregate classifier

<b>pattern_list</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; global list for match a column in table as datatime column

<b>predict_single_sent</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; retrieve raw conclusion of condition extraction in [JointBERT](https://github.com/monologg/JointBERT) format.

<b>recurrent_extract</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; use predict_single_sent in recurrent manner to extract condition 、residual (condition remove) components 、conn string (such as ‘and’ ‘or’ lookup string) from question.

<b>rec_more_time</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; use recurrent_extact in recurrent manner to have exhaustive condition extractions from question.

<b>find_min_common_strings</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; find common strings subset with a non-greedy filter

<b>time_template_extractor</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; match pattern_list elements in one column of table data

<b>justify_column_as_datetime</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; use time_template_extractor over dataframe

<b>justify_column_as_datetime_reduce</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; merge conclusion of different elements pattern_list in map reduce manner.

<b>choose_question_column</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; justify which column to ask by firstly look up this question if it is about datetime column
 (use justify_column_as_datetime_reduce)

<b>single_label_func</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; when the question have aggregate kws (i.e. a question about MAX MIN AVG COUNT SUM), decide which aggregate keyword to use. The construction about this function is get from [agg-classifier.ipynb](https://github.com/svjack/tableQA-Chinese/blob/main/notebook/agg-classifier.ipynb) with [snorkel labeling model](https://snorkel.readthedocs.io/en/v0.9.3/packages/_autosummary/labeling/snorkel.labeling.LabelModel.html) and parameters choose methods in strategy [backtest](https://en.wikipedia.org/wiki/Backtesting) (like [vectorbt](https://github.com/polakowo/vectorbt))

<b>simple_special_func</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; decide if the question is a special question.
the definition of “special” is that, the according sql query about the question is a query with aggregate words (MAX MIN AVG COUNT SUM)

<b>simple_total_label_func</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; decide the aggregate kw used in sql query (first use simple_special_func and if is special, use single_label_func to decide which agg kw to use)

<b>split_by_cond</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; function to extract condition and conn string from question and residual components (remove condition) , use fit_kw to retrieve conn string (‘and’ ‘or’)

<b>filter_total_conds</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; filter the condition extracted by split_by_cond with the help of datatype of table

<b>augment_kw_in_question</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; some condition keywords can not extract by  [JointBERT](https://github.com/monologg/JointBERT)  are append by this function by config.
For example:
 &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; [JointBERT](https://github.com/monologg/JointBERT) can extract (“城市”, ==, “宁波”) from “城市是宁波的一手房成交均价是多少？” but can not extract (“城市”, ==, “宁波”) from “宁波的一手房成交均价是多少？”
so this function first justify which column in table is a category column, and find if any category in that column as a condition in question, and add it to “question_kw_conds”

<b>choose_question_column_by_rm_conds</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; same as choose_question_column, but with remove all condition first.

<b>choose_res_by_kws</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; another method (parallel with choose_question_column) to choose question column use the nearest components with question words.

<b>cat6_to_45_column_type</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; above simple_total_label_func take “COUNT” and “SUM” as identical. this function distinguish them by the question column datatype, category column use “COUNT” and numerical column use “SUM”

<b>full_before_cat_decomp</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; function to apply all above functions to question, to have the final prediction of question of tableQA task
when set only_req_columns to True, only return the truly needed prediction of question.
<br/><br/>
“question”: user input question<br/>
“total_conds_filtered”: all extract conditions.<br/>
“conn_pred”: connection string (“and” “or”) among conditions<br/>
“question_column”: which column to ask question<br/>
“agg_pred”: aggregate operator on question column.<br/>
<br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; because there are many components extract from above, the final agg operator use the max value above them, the final question column retrieve by a particular order (from accurate residual components to full question question word nearest prediction)

<br />

<h2> databaseQA components </h2> 

<br />

 <h3> <a href="https://github.com/svjack/tableQA-Chinese/blob/main/script/finance_dictionary_construct.py">
  finance_dictionary_construct.py
  </a></h3>

&ensp; &ensp; &ensp; Construct finance dictionary by open source knowledge graph in [ownthink](https://github.com/ownthink/KnowledgeGraphData).

<b>retrieve_entities_data</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; fetch entity information by ownthink api and save to local.

<b>read_fkg</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; read finance entities data from local in json load format (as python object)

<b>finance_tag_filter</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; justify a entity is a finance entity by tag in [ownthink](https://github.com/ownthink/KnowledgeGraphData) response data.

<br />

<h3> <a href="https://github.com/svjack/tableQA-Chinese/blob/main/script/tableqa_search.py">
 tableqa_search.py
  </a></h3>

&ensp; &ensp; &ensp; Use finance dictionary and [Bertopic](https://github.com/MaartenGr/BERTopic) model to build profile on table.

<b>template_extractor</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; extract all chinese components from table

<b>extract_text_info_from_tables_json</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; extract all text components from table (add header)

<b>eval_df</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; eval df collection column as python object.

<br />

<h3> <a href="https://github.com/svjack/tableQA-Chinese/blob/main/script/tableqa_finance_unsupervised.py">
 tableqa_finance_unsupervised.py
  </a></h3>

&ensp; &ensp; &ensp; use profile build by [tableqa_search.py](https://github.com/svjack/tableQA-Chinese/blob/main/script/tableQA_single_table.py) to filter out  finance tables from table collection.

<b>retrieve_finance_tag_df</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; build finance tag dataframe from json format finance dictionaries.

<b>filter_high_evidence_func</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; filter out high evidence dataframe part by high evidence dataframe with title and with finance keywords(need_words) in header without title.

<b>low_evidence_finance_df</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp;[snorkel labeling function](https://snorkel.readthedocs.io/en/v0.9.3/packages/_autosummary/labeling/snorkel.labeling.LabelingFunction.html) conclusion that predict samples(tables) as finance tables (use labeling function if_top_topic_id if_max_topic_id from label made by [Bertopic](https://github.com/MaartenGr/BERTopic) 
from [tableqa_search.py](https://github.com/svjack/tableQA-Chinese/blob/main/script/tableQA_single_table.py), if_other_topic is a observe rule on title, if_tag_in_entities_* use finance_tag_df in different tag group ) 
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; After this voting (labeling model fit prediction) and some rule based filter, use filter_high_evidence_func to produce a relatively clean finance table subset, and save this subset’s profile to local

<br />

<h3> <a href="https://github.com/svjack/tableQA-Chinese/blob/main/notebook/fine-tune-on-finance.ipynb">
fine-tune-on-finance.ipynb
  </a></h3>

&ensp; &ensp; &ensp; use finance tables’s profile and [tableQA_single_table.py](https://github.com/svjack/tableQA-Chinese/blob/main/script/tableQA_single_table.py) to perform finance databaseQA search on finance database

<b>retrieve_table_in_pd_format</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; instantiation dataframe from finance database with add header on truly data and add alias column to dataframe by config ori_alias_mapping.(this will help table perform better with alias of question column)

<b>search_tables_on_db_iter</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; use [sqlite_utils](https://github.com/simonw/sqlite-utils)’ search function on sqlite FTS5 registered meta_table (without loss of generalizations, meta_table in this notebook always point to finance desc table produce by [tableqa_finance_unsupervised.py](https://github.com/svjack/tableQA-Chinese/blob/main/script/tableqa_finance_unsupervised.py)) to retrieve all tables related with kwords in dataframe format by retrieve_table_in_pd_format

<b>search_question</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; search and sorted table names related with question input and meta_table by [bm25](https://github.com/dorianbrown/rank_bm25) measurement

<b>get_table</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; get table object from a collection of table objects init by sqlite_utils’ [Table](https://github.com/simonw/sqlite-utils/blob/c236894caa976d4ea5c7503e9331a3e9d2fbb1c4/sqlite_utils/db.py#L727) class (by name attribute)

<b>extract_special_header_string</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; use meta_table header column to produce chinese finance special header keywords and english finance special header keywords  
this function will help zh_sp_tks and en_sp_tks to build a common asked column mapping between question 、conditions、question column.

<b>produce_token_dict</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; use *_sp_tks and sp_* to build a two level *_sp_dict as extract_special_header_string prepare.

<b>calculate_sim_score_on_s</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; use zh_sp_dict and en_sp_dict to guess the most match column in question column、conditions with header. Calculate bm25 score as similarity measurement between question and table.

<b>percentile_sort</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; a composite resort above different scores produced by above bm25 (question table similarity) and tableQA conclusion validation measurement defined in sort_search_tableqa_df’s sort_order_list and sort_func_list. Which use numpy’s percentile to produce confirm quality measurement above different scores (this makes quality scores partition as a collection of statistic stairs that search conclusions can walk down)

<b>sort_search_tableqa_df</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; add tableQA conclusion validation measurements by sort_order_list and sort_func_list. and bm25 between question and all_text_str_elements as search_score with many other scores defined by above functions.
t5_order is to control the lexicograpic order in qa_score_tuple, which is useful in conclusion of run_sql_search (in that function, qa_score_tuple is more useful than percentile_sort, because that function have truly remove some invalid search conclusions by run sql query)

<b>single_question_search</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; run question QA above a collection of tables (databaseQA) with the help of percentile_sort on the conclusion of this function, can see the performance of databaseQA. the columns user should care are :
<br/><br/>
"question_column", <br/>
"total_conds_filtered",<br/>
"agg_pred",<br/>
"conn_pred"<br/>
“*score*”,<br/>
“name”<br/>
<br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; firstly four columns are tableQA conclusion, columns match “*score*” format, are search score columns, and “name” refer to table name in the database, user can check the table structure by get_table function

<b>run_sql_query</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; run sql query on conclusion of single_question_search by init sqlite table

<b>run_sql_search</b><br/>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; add run_sql_query conclusion to single_question_search and remove invalid samples, where invalid refer to query that have no records(select count(*) statement) on table
prefer to use “qa_score_tuple” as sort scores.
