# 中药材电商评论分析系统

本系统实现了论文中提到的情感分析和关键词提取算法，用于分析中药材电商评论数据，支持中药材供应链服务质量评价研究。

## 🎯 功能特点

### 1. 情感分析
- **基础方法**：基于词典的情感分析方法
- **机器学习**：SVM、朴素贝叶斯、逻辑回归
- **深度学习**：LSTM、TextCNN
- **预训练模型**：BERT中文预训练模型
- **图算法**：基于TextRank的情感分析

### 2. 关键词提取
- **TF-IDF**：基于词频-逆文档频率的关键词提取
- **TextRank**：基于图算法的关键词提取
- **LDA**：基于潜在狄利克雷分配的主题关键词提取

### 3. 关键词映射
- 将提取的关键词映射到评价指标体系
- 分析不同评价指标的关注度
- 支持供应链全流程分析（上游、中游、下游）

### 4. 可视化分析
- 算法性能比较图表
- 关键词词云
- 评价指标映射结果可视化
- 训练过程可视化

## 📁 项目结构

```
master-thesis/
├── core/                           # 核心分析模块
│   ├── sentiment_analysis.py       # 基础情感分析
│   ├── deep_learning_sentiment.py  # LSTM情感分析
│   ├── bert_sentiment_analysis.py  # BERT情感分析
│   ├── textcnn_sentiment_analysis.py # TextCNN情感分析
│   ├── textrank_sentiment_analysis.py # TextRank情感分析
│   └── keyword_extraction.py       # 关键词提取
├── scripts/                        # 执行脚本
│   ├── main_analysis.py            # 主分析脚本（部分数据）
│   ├── full_data_analysis.py       # 全数据分析脚本
│   └── train_deep_learning_models.py # 深度学习模型训练
├── utils/                          # 工具模块
│   ├── data_analysis.py            # 数据分析工具
│   ├── visualization.py            # 可视化工具
│   ├── keyword_mapping.py          # 关键词映射
│   ├── read_comments.py            # 评论数据读取
│   ├── generate_figures.py         # 图表生成
│   ├── generate_sentiment_boxplot.py # 情感分析箱线图
│   └── scholar_search.py           # 学术搜索工具
├── docs/                           # 项目文档
│   ├── 1-topics.md                 # 选题依据
│   ├── 2-plans.md                  # 研究方案
│   ├── 3-schedules.md              # 进度安排
│   ├── 4-refers.md                 # 参考文献
│   ├── refqa.md                    # 参考问答
│   ├── 论文-202507.md              # 论文正文
│   └── 参考文献.txt                # 参考文献列表
├── data/                           # 数据目录
│   └── *.xls/*.xlsx                # 评论数据文件
├── output/                         # 输出目录
│   ├── figures/                    # 生成的图表
│   ├── models/                     # 保存的模型
│   └── *.csv/*.json                # 分析结果
├── config/                         # 配置文件
│   ├── requirements.txt            # Python依赖
│   └── install_dependencies.sh     # 安装脚本
└── src/                           # 遗留代码（待清理）
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository_url>
cd master-thesis

# 安装依赖
pip install -r config/requirements.txt

# 或使用安装脚本
bash config/install_dependencies.sh
```

### 2. 基本使用

```bash
# 使用部分数据进行快速分析
python scripts/main_analysis.py

# 使用全部数据进行完整分析
python scripts/full_data_analysis.py --mode all
```

### 3. 高级用法

```bash
# 仅运行情感分析
python scripts/full_data_analysis.py --mode sentiment --sample_size 10000

# 仅运行关键词提取
python scripts/full_data_analysis.py --mode keyword

# 使用深度学习模型
python scripts/full_data_analysis.py --use_deep_learning --use_bert

# 使用所有可用模型
python scripts/full_data_analysis.py --use_deep_learning --use_bert --use_textcnn --use_textrank_sa
```

## 📊 命令行参数

### full_data_analysis.py 参数说明

- `--mode`: 分析模式 (`sentiment` | `keyword` | `all`)
- `--sample_size`: 样本大小，随机采样指定数量的评论
- `--max_comments`: 最大评论数量限制
- `--balanced`: 使用均衡采样（各类别数量相等）
- `--output_dir`: 输出目录（默认：`output`）
- `--use_deep_learning`: 启用LSTM深度学习模型
- `--use_bert`: 启用BERT预训练模型
- `--use_textcnn`: 启用TextCNN模型
- `--use_textrank_sa`: 启用TextRank情感分析
- `--offline_bert`: 使用离线BERT模型
- `--bert_sample_size`: BERT分析样本大小（默认：10000）

### 使用示例

```bash
# 使用10000条评论进行均衡采样的完整分析
python scripts/full_data_analysis.py --mode all --sample_size 10000 --balanced

# 使用BERT和LSTM进行情感分析
python scripts/full_data_analysis.py --mode sentiment --use_deep_learning --use_bert

# 限制最大评论数量并输出到指定目录
python scripts/full_data_analysis.py --max_comments 50000 --output_dir results
```

## 📈 输出结果

### 1. 情感分析结果
- `sentiment_analysis_results.csv`: 各算法的性能指标
- `sentiment_analysis_comparison.png`: 算法性能比较图表
- `lstm_training_history.png`: LSTM模型训练历史
- `bert_training_stats.png`: BERT模型训练统计

### 2. 关键词提取结果
- `keywords_wordcloud.png`: 关键词词云
- `keyword_extraction_comparison.png`: 不同算法提取的关键词比较
- `keyword_mapping_results.csv`: 关键词映射到评价指标的结果
- `keyword_mapping_comparison.png`: 评价指标映射结果可视化

### 3. 综合报告
- `summary_report.md`: 包含所有分析结果的摘要报告
- `deep_learning_report.md`: 深度学习模型分析报告

## 🔧 技术栈

- **编程语言**: Python 3.8+
- **机器学习**: scikit-learn
- **深度学习**: PyTorch, TensorFlow/Keras
- **自然语言处理**: jieba, transformers (Hugging Face)
- **数据处理**: pandas, numpy
- **可视化**: matplotlib, seaborn, wordcloud
- **图算法**: networkx
- **其他**: gensim, openpyxl, tqdm

## 📋 评价指标体系

### 供应链三个维度

1. **上游（原料采购）**
   - 原料质量评分
   - 供应稳定性
   - 原材料可追溯性评分

2. **中游（加工制造）**
   - 生产效率评分
   - 工艺技术评价
   - 质检标准符合度

3. **下游（销售配送）**
   - 库存管理评分
   - 订单准确性
   - 交货速度
   - 包装评分
   - 售后服务质量

## ⚠️ 注意事项

1. **计算资源**: 深度学习模型训练需要较高的计算资源，建议在GPU环境下运行BERT模型
2. **内存需求**: 处理大量评论数据可能需要较大内存，可通过`--sample_size`参数控制数据量
3. **数据格式**: 评论数据需为Excel格式，包含"评论内容"列
4. **模型下载**: 首次使用BERT模型时需要下载预训练模型，请确保网络连接正常
5. **中文支持**: 系统专门针对中文文本优化，使用jieba分词器

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目仅用于学术研究目的。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 项目Issues: [GitHub Issues](链接)
- 邮箱: [your-email@example.com]

---

**注**: 本系统是硕士论文"基于在线评论的中药材企业电商供应链服务质量评价研究"的实现代码。 