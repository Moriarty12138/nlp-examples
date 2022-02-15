# nlp-demos
平时积累的一些NLP代码。



目录结构：

* classification 文本分类
  * Sequence_Classification_with_IMDb_Reviews.py 在 distilbert 上进行 fine-tuning ，数据集采用 IMDb 英文数据集。(Transformers/PyTorch)
  * iflytek_classification.py 在iflytek长文本分类数据集上对BERT模型进行 fine-tuning
* summarization 文本摘要
  * bart 使用开源CPT预训练模型作为bart权重，完成文本摘要训练。  
* hr_analysis.ipynb 员工离职数据分析，包含一些图表的画法
