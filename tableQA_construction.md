<!--
construction documentation 
components description:
-->

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
  <h3 align="center">tableQA-Chinese Design</h3>

  <p align="center">
    <br />
  </p>
</p>

This project is similar with [tableQA](https://github.com/abhijithneilabraham/tableQA) which is also a unsupervised tableQA project with clear construction.

In order to have a clear introduce about tableQA-chinese, following will provide a brief tour of [abhijithneilabraham’s tableQA](https://github.com/abhijithneilabraham/tableQA).

TableQA task in unsupervised manner can decomposed into condition extraction 、question column extraction 、aggregate operator classification.

[abhijithneilabraham’s tableQA](https://github.com/abhijithneilabraham/tableQA) take condition extraction as a SQUAD predition problem and use [transformers’ SQUAD model](https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad)
to ask condition in user input question. (see qa and slot_fill function in [nlp.py](https://github.com/abhijithneilabraham/tableQA/blob/master/tableqa/nlp.py))

When comes to question column extraction, it use unknown_slot_extractor to remove condition extraction from header and question to justify which column to ask.
And finally, use a embedding classifier based on [universe embedding from google tensorflow](https://tfhub.dev/google/universal-sentence-encoder/4) to predict the aggregate operator (see [clauses.py](https://github.com/abhijithneilabraham/tableQA/blob/master/tableqa/clauses.py))

The above components only use a pretrained SQUAD qa model and self-trained [keras sentence embedding based classifier](https://github.com/abhijithneilabraham/tableQA/blob/master/tableqa/Question_Classifier.h5).
 
If you want to have a try to use abhijithneilabraham’s tableQA   
project on chinese task, you can replace the SQUAD qa model by some multilingual model such as [deepset’s roberta-base-squad2](https://huggingface.co/deepset/roberta-base-squad2)
and pre-tokenize the input by [jieba](https://github.com/fxsjy/jieba) with some keywords replacement. (i also try to use [albert-chinese-large-qa](https://huggingface.co/wptoux/albert-chinese-large-qa), the conclusion can not beat multilingual model)

And when it comes to [aggregate operator classification](https://github.com/abhijithneilabraham/tableQA/blob/master/tableqa/clauses.py), simply use [EasyNMT](https://github.com/UKPLab/EasyNMT) to translate training data used in [clf.py](https://github.com/abhijithneilabraham/tableQA/blob/master/tableqa/clf.py) can have similar prediction.

But i think there are many things need to be improved when use abhijithneilabraham’s tableQA in chinese.

In the condition extraction 、aggregate operator classification tasks.
abhijithneilabraham’s tableQA always use [“number of” and “which”](https://github.com/abhijithneilabraham/tableQA/blob/master/tableqa/nlp.py) to ask the question about the condition information. This input format can’t change and no more info about condition column. And when you change the input format in english or chinese, the conclusion seems different. —— use some kinds of formats in some question context will perform better but other contexts may worse. such as “how many” “how much” in english and “什么” “怎么” “怎么样” in chinese, so which format is the best to choose ? 

A method to improve this is to iterate over many format and combine them, but also can not overcome it essentially.

abhijithneilabraham’s tableQA use [sentence embedding classifier](https://github.com/abhijithneilabraham/tableQA/blob/master/tableqa/clauses.py) to make aggregate operator prediction. this have some limitations on imbalanced training dataset, this is the common problem to many classification tasks. It use full text question to train the classifier without remove condition，so it had extract and maintain useless linguistic components.

They are noise to the classifier. (and the  [clf.py](https://github.com/abhijithneilabraham/tableQA/blob/master/tableqa/clf.py) may always perform worse to distinguish “COUNT” and “SUM”, because without the help of table datatype structure, the classifier can not distinguish them only by question context)

The above two limitations are all about use supervised model to deal with unsupervised task without avoid the shortcoming from supervised.
(feature sensitive and imbalance)

To tackle them. tableQA-chinese replace SQUAD qa model by [Joint Intent Classification and Slot Filling](https://github.com/monologg/JointBERT) on question to extract condition in a no ask unsupervised manner (support by [JointBERT](https://github.com/monologg/JointBERT), train with a NER manner).

And use [snorkel’s](https://github.com/snorkel-team/snorkel) labeling model help to construct a keyword based classifier to predict aggregate operator. (without training, but use rule based order permutations [backtest](https://en.wikipedia.org/wiki/Backtesting) method,  like many [finance strategy](https://en.wikipedia.org/wiki/Trading_strategy) backtest framework used to choose the best strategy——like [vectorbt](https://github.com/polakowo/vectorbt/blob/master/examples/MACDVolume.ipynb))

Finally , To improve the tableQA task into databaseQA, add search components on finance tables to make this project as a toolkit for information aggregation above database (finance tables) with different table structure which is useful for database construction explore and data consolidation from multi-table.

This need merge the text similarity between question and table descriptions with measure scores of tableQA conclusion. tableQA-Chinese handle it as a  
consistent percentages’ ranking problem.
