# 中药材电商评论分析系统

本系统实现了论文中提到的情感分析和关键词提取算法，用于分析中药材电商评论数据。

## 功能特点

1. **情感分析**：
   - 基于词典的情感分析方法
   - 基于机器学习的情感分析方法（SVM、朴素贝叶斯）
   - 基于深度学习的情感分析方法（LSTM）
   - 基于BERT预训练模型的情感分析方法

2. **关键词提取**：
   - 基于TF-IDF的关键词提取
   - 基于TextRank的关键词提取
   - 基于LDA的主题关键词提取

3. **关键词映射**：
   - 将提取的关键词映射到评价指标体系
   - 分析不同评价指标的关注度

4. **可视化**：
   - 算法性能比较图表
   - 关键词词云
   - 评价指标映射结果可视化

## 安装依赖

本系统依赖于多个Python库，可以使用以下命令安装：

```bash
# 激活虚拟环境
source venv/bin/activate

# 安装依赖
pip install pandas numpy matplotlib scikit-learn jieba gensim networkx wordcloud tensorflow torch transformers
```

## 使用方法

### 1. 基本用法

运行主脚本进行分析：

```bash
# 使用部分数据进行分析
python main_analysis.py

# 使用全部数据进行分析
python full_data_analysis.py
```

默认情况下，系统将运行所有分析模块，并将结果保存到`output`目录。

### 2. 命令行参数

#### 基本分析脚本参数

```bash
python main_analysis.py --mode [sentiment|keyword|all] --max_files N --output_dir DIR
```

- `--mode`：分析模式，可选值为`sentiment`（仅情感分析）、`keyword`（仅关键词提取）或`all`（全部）
- `--max_files`：每类评论使用的最大文件数，默认为3
- `--output_dir`：输出目录，默认为`output`

#### 全数据分析脚本参数

```bash
python full_data_analysis.py --mode [sentiment|keyword|all] --sample_size N --output_dir DIR --use_deep_learning --use_bert
```

- `--mode`：分析模式，可选值为`sentiment`（仅情感分析）、`keyword`（仅关键词提取）或`all`（全部）
- `--sample_size`：样本大小，如果指定，将随机采样指定数量的评论进行分析
- `--output_dir`：输出目录，默认为`output`
- `--use_deep_learning`：是否使用深度学习模型（LSTM）
- `--use_bert`：是否使用BERT模型

示例：

```bash
# 仅运行情感分析，使用LSTM模型
python full_data_analysis.py --mode sentiment --use_deep_learning

# 运行所有分析，使用10000条评论样本，包括BERT模型
python full_data_analysis.py --mode all --sample_size 10000 --use_deep_learning --use_bert

# 仅运行关键词提取，使用全部数据
python full_data_analysis.py --mode keyword
```

### 3. 单独运行各模块

也可以单独运行各个模块：

```bash
# 情感分析
python sentiment_analysis.py

# 深度学习情感分析
python deep_learning_sentiment.py

# BERT情感分析
python bert_sentiment_analysis.py

# 关键词提取
python keyword_extraction.py
```

## 输出结果

系统的输出结果包括：

1. **情感分析结果**：
   - `sentiment_analysis_results.csv`：各算法的性能指标
   - `sentiment_analysis_comparison.png`：算法性能比较图表
   - `lstm_training_history.png`：LSTM模型训练历史
   - `bert_training_stats.png`：BERT模型训练历史

2. **关键词提取结果**：
   - `keywords_wordcloud.png`：关键词词云
   - `keyword_extraction_comparison.png`：不同算法提取的关键词比较
   - `keyword_mapping_results.csv`：关键词映射到评价指标的结果
   - `keyword_mapping_comparison.png`：评价指标映射结果可视化

3. **摘要报告**：
   - `summary_report.md`：包含所有分析结果的摘要报告

## 系统架构

本系统由以下几个主要模块组成：

1. **sentiment_analysis.py**：实现基于词典和机器学习的情感分析方法
2. **deep_learning_sentiment.py**：实现基于LSTM的情感分析方法
3. **bert_sentiment_analysis.py**：实现基于BERT的情感分析方法
4. **keyword_extraction.py**：实现关键词提取和映射方法
5. **main_analysis.py**：主脚本，使用部分数据进行分析
6. **full_data_analysis.py**：全数据分析脚本，支持读取全部数据并集成所有模型

## 注意事项

1. 本系统需要处理大量评论数据，可能需要较长时间运行。
2. 深度学习模型训练需要较高的计算资源，请确保您的计算机有足够的内存和处理能力。
3. BERT模型训练尤其需要大量计算资源，建议在GPU环境下运行。
4. 为了加快处理速度，系统默认只使用每类评论的前3个文件，可以通过参数调整。
5. 使用全数据分析时，建议根据可用内存指定合适的样本大小。 