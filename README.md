<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h3 align="center">tableQA-Chinese</h3>

  <p align="center">
   		无监督TableQA和数据库QA在中文表格数据和金融问题上的应用。
    <br />
  </p>
</p>

[In English](README_EN.md)

<!-- TABLE OF CONTENTS -->


<!-- ABOUT THE PROJECT -->
## 关于这个工程

<!--
[![Product Name Screen Shot][product-screenshot]](https://example.com)
-->


<!--
There are many great README templates available on GitHub, however, I didn't find one that really suit my needs so I created this enhanced one. I want to create a README template so amazing that it'll be the last one you ever need.
-->

通常针对具有数字和类别混合的表格数据，对这些数据进行自然语言查询是实际应用中的常见任务。本项目尝试在中文表格数据上进行无监督的TableQA，并主要针对金融数据进行数据库QA。

<!--
Querying natural language on tabular data is a common task for practical application, typically, for data with mixture of numerical and categories.
This project is a unsupervised tableQA attempts on chinese tabular data and databaseQA mainly on finance data.
-->

<!--
Here's why:
* Your time should be focused on creating something amazing. A project that solves a problem and helps others
* You shouldn't be doing the same tasks over and over like creating a README from scratch
* You should element DRY principles to the rest of your life :smile:

Of course, no one template will serve all projects since your needs may be different. So I'll be adding more in the near future. You may also suggest changes by forking this repo and creating a pull request or opening an issue.

A list of commonly used resources that I find helpful are listed in the acknowledgements.
-->
### 框架组成部分
<!--
This section should list any major frameworks that you built your project using. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.
* [Bootstrap](https://getbootstrap.com)
* [JQuery](https://jquery.com)
* [Laravel](https://laravel.com)

* [Prophet](https://www.prophet.com/)
* [Scikit-Hts](https://github.com/carlomazzaferro/scikit-hts)
* [Hyperopt](https://github.com/hyperopt/hyperopt)

* [Gensim](https://github.com/RaRe-Technologies/gensim)
* [Wikipedia2Vec](https://github.com/wikipedia2vec/wikipedia2vec)
-->
* [JointBERT](https://github.com/monologg/JointBERT)
* [Snorkel](https://github.com/snorkel-team/snorkel)
* [Bertopic](https://github.com/MaartenGr/BERTopic)
* [sqlite-utils](https://github.com/simonw/sqlite-utils)



<!-- GETTING STARTED -->
## 开始
<!--
This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.
-->

<!--
### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
```sh
npm install npm@latest -g
```
-->

### 安装
[Snorkel](https://github.com/snorkel-team/snorkel) 和 [Bertopic](https://github.com/MaartenGr/BERTopic) 可能存在依赖冲突。建议使用 conda 安装不同虚拟环境分别区分 [JointBERT](https://github.com/monologg/JointBERT)、[Snorkel](https://github.com/snorkel-team/snorkel) 和 [Bertopic](https://github.com/MaartenGr/BERTopic)

<!--
[Snorkel](https://github.com/snorkel-team/snorkel) and [Bertopic](https://github.com/MaartenGr/BERTopic) may have some dependency conflict.
Recommend use three different visual environments provide by [conda](https://docs.conda.io/en/latest/) to distinguish [JointBERT](https://github.com/monologg/JointBERT) [Snorkel](https://github.com/snorkel-team/snorkel) [Bertopic](https://github.com/MaartenGr/BERTopic) respectively.
-->

* conda
```sh
conda create -n jointbert_env python=3.8
conda activate jointbert_env
pip install -r jointbert_requirements.txt
conda create -n snorkel_env python=3.8
conda activate snorkel_env
bash snorkel_install.sh
conda create -n topic_env python=3.8
conda activate topic_env
bash topic_install.sh
```

在使用notebook和script之前，请键入 conda activate 命令以初始化特定环境。下面是不同文件及其环境映射：<br/>
<br/>
(jointbert_env)<br/>
[condition_trainer.py](https://github.com/svjack/tableQA-Chinese/blob/main/script/condition_trainer.py)<br/>
[tableQA_single_table.py](https://github.com/svjack/tableQA-Chinese/blob/main/script/tableQA_single_table.py)<br/>
[tableqa-single-valid.ipynb](https://github.com/svjack/tableQA-Chinese/blob/main/notebook/tableqa-single-valid.ipynb)<br/>
[fine-tune-on-finance.ipynb](https://github.com/svjack/tableQA-Chinese/blob/main/notebook/fine-tune-on-finance.ipynb)<br/>
<br/>
(snorkel_env)<br/>
[finance_dictionary_construction.py](https://github.com/svjack/tableQA-Chinese/blob/main/script/finance_dictionary_construct.py)<br/>
[agg-classifier.ipynb](https://github.com/svjack/tableQA-Chinese/blob/main/notebook/agg-classifier.ipynb)<br/>
[tableqa_finance_unsupervised.py](https://github.com/svjack/tableQA-Chinese/blob/main/script/tableqa_finance_unsupervised.py)<br/>
<br/>
(topic_env)<br/>
[tableqa_search.py](https://github.com/svjack/tableQA-Chinese/blob/main/script/tableqa_search.py)<br/>
<br/>
<br/>

<!-- USAGE EXAMPLES -->

<h1><b>完全功能使用</b></h1>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp;包含模型训练方式及使用

## tableQA 使用方式
<!--
Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_
-->

<h4>
<p>
<a href="https://github.com/ZhuiyiTechnology/TableQA">
1. 从 github 上下载tableqa数据</a>
</p>
</h4>

<h4>
<p>
<a href="https://github.com/monologg/JointBERT">
2. 下载 JointBERT 工程</a>
</p>
</h4>

<h4>
<p>
<a href="https://github.com/svjack/tableQA-Chinese/blob/main/script/condition_trainer.py">3. 在tableqa数据上使用 JointBERT 训练实体和条件抽取模型</a>
</p>
</h4>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp; 不要忘记在代码中配置 train_path 和 jointbert_path
<h4>
<p>
<a href="https://github.com/svjack/tableQA-Chinese/blob/main/notebook/tableqa-single-valid.ipynb">4. 在 tableqa-single-valid 中使用 tableQA_single_table.py 在单个数据表数据中执行数据表问答 </a>
</p>
</h4>

## databaseQA 使用方式

<h4>
<p>
<a href="https://github.com/svjack/tableQA-Chinese/blob/main/script/finance_dictionary_construct.py">5. 运行 finance_dictionary_construct 对 databaseQA 建立金融字典</a>
</p>
</h4>

<h4>
<p>
<a href="https://github.com/svjack/tableQA-Chinese/blob/main/script/tableqa_search.py">6. 对 databaseQA 建立金融画像</a>
</p>
</h4>

<h4>
<p>
<a href="https://github.com/svjack/tableQA-Chinese/blob/main/script/tableqa_finance_unsupervised.py">7. 使用 Snorkel 从tableqa中筛选金融数据表格 </a>
</p>
</h4>

<h4>
<p>
<a href="https://github.com/svjack/tableQA-Chinese/blob/main/notebook/fine-tune-on-finance.ipynb">8. 在构建的金融数据库(由一些金融数据表格构建)上执行databaseQA</a>
</p>
</h4>

<br/>

<h1><b>简单使用方式</b></h1>
&ensp; &ensp; &ensp; &ensp; &ensp; &ensp;使用已经构建好的预训练模型进行使用 <br/>
<br/>

这是使用此项目的推荐方法，因为金融词典是通过从 [ownthink](https://github.com/ownthink/KnowledgeGraphData) 调用 API 构建的。如果 API 不稳定或 [Bertopic](https://github.com/MaartenGr/BERTopic) 发生某些随机状态更改，可能会对数据库 QA 中的金融概况产生不良影响。因此，我推荐至少使用 [Google Drive](https://drive.google.com/drive/folders/19NcYWybSBi_44zfcbtstLXk5rB_SymJt?usp=sharing) 中的 data.tar.gz。通过使用 data.tar.gz，您只需要安装 (jointbert_env) 环境即可探索 TableQA 和 DatabaseQA。<br/>

从 [Google Drive](https://drive.google.com/drive/folders/19NcYWybSBi_44zfcbtstLXk5rB_SymJt?usp=sharing) 下载预训练模型和数据。此共享路径包含三个文件：<br/>
<b>1 bert.tar.gz</b> (JointBERT 训练模型) <br/>
<b>2 conds.tar.gz</b> (JointBERT 友好数据集) <br/>
<b>3 data.tar.gz</b> (Bertopic 和 Snorkel 生成的数据，用于执行数据库 QA)<br/>

将它们解压缩到此项目的根路径，并将它们配置到上面的 Full Usage 文件中。您可以通过在 notebook 和 script 中搜索已解压缩的文件名来找到配置位置。<br/>


## tableQA 使用
<!--
Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_
-->

<h4>
<p>
<a href="https://github.com/ZhuiyiTechnology/TableQA">
1. 从 github 上下载中文 tableqa 数据</a>
</p>
</h4>

<h4>
<p>
<a href="https://github.com/monologg/JointBERT">
2. 下载 JointBERT 工程</a>
</p>
</h4>

<h4>
<p>
<a href="https://github.com/svjack/tableQA-Chinese/blob/main/notebook/tableqa-single-valid.ipynb"> 3. 在 tableqa-single-valid 中使用 tableQA_single_table.py 在单个数据表数据中执行数据表问答</a>
</p>
</h4>

![avatar](IMG_0900.jpeg)

## databaseQA 使用方法
<h4>
<p>
<a href="https://github.com/svjack/tableQA-Chinese/blob/main/notebook/fine-tune-on-finance.ipynb">4. 在构建的金融数据库(由一些金融数据表格构建)上执行databaseQA </a>
</p>
</h4>

![avatar](IMG_0901.jpeg)

![avatar](IMG_0907.jpeg)

![avatar](IMG_0904.jpeg)

<br/>

<h1>
<p>
<a href="https://github.com/svjack/tableQA-Chinese/blob/main/tableQA_construction.md"> 设计结构 （英文版）</a>
</p>
</h1>

<h1>
<p>
<a href="https://github.com/svjack/tableQA-Chinese/blob/main/tableQA_api_documentation.md"> API 文档 （英文版）</a>
</p>
</h1>


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

<!--
Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com
-->
svjack - svjackbt@gmail.com

<!--
Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)
-->
Project Link:[https://github.com/svjack/tableQA-Chinese](https://github.com/svjack/tableQA-Chinese)


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
<!--
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Img Shields](https://shields.io)
* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Pages](https://pages.github.com)
* [Animate.css](https://daneden.github.io/animate.css)
* [Loaders.css](https://connoratherton.com/loaders)
* [Slick Carousel](https://kenwheeler.github.io/slick)
* [Smooth Scroll](https://github.com/cferdinandi/smooth-scroll)
* [Sticky Kit](http://leafo.net/sticky-kit)
* [JVectorMap](http://jvectormap.com)
* [Font Awesome](https://fontawesome.com)
-->

* [tableQA](https://github.com/abhijithneilabraham/tableQA)
* [vectorbt](https://github.com/polakowo/vectorbt)
* [zvt](https://github.com/zvtvz/zvt)
* [JointBERT](https://github.com/monologg/JointBERT)
* [Snorkel](https://github.com/snorkel-team/snorkel)
* [Bertopic](https://github.com/MaartenGr/BERTopic)
* [sqlite-utils](https://github.com/simonw/sqlite-utils)
* [bm25](https://github.com/dorianbrown/rank_bm25)
* [TableQA](https://github.com/ZhuiyiTechnology/TableQA)
* [ownthink](https://github.com/ownthink/KnowledgeGraphData)
* [Sentence Transformers](https://github.com/UKPLab/sentence-transformers)
* [EasyNMT](https://github.com/UKPLab/EasyNMT)
* [PyArrowExpressionCastToolkit](https://github.com/svjack/PyArrowExpressionCastToolkit)
* [Sbert-ChineseExample](https://github.com/svjack/Sbert-ChineseExample)

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=flat-square
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=flat-square
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=flat-square
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=flat-square
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=flat-square
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
