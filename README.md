# Chinese Herbal Medicine Sentiment Analysis System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/chinese-herbal-sentiment.svg)](https://badge.fury.io/py/chinese-herbal-sentiment)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen.svg)](https://github.com/chenxingqiang/chinese-herbal-sentiment#readme)

A comprehensive Natural Language Processing (NLP) toolkit specifically designed for analyzing customer reviews and evaluating supply chain quality in Chinese herbal medicine e-commerce platforms.

## 🎯 Features

### 🔍 Sentiment Analysis
- **Dictionary-based Analysis**: Traditional sentiment analysis using Chinese sentiment dictionaries
- **Machine Learning Models**: SVM, Naive Bayes, and Logistic Regression classifiers
- **Deep Learning Models**: LSTM, TextCNN, and BERT-based sentiment analysis
- **Graph-based Analysis**: TextRank algorithm for sentiment analysis

### 🔑 Keyword Extraction
- **TF-IDF**: Term Frequency-Inverse Document Frequency for keyword extraction
- **TextRank**: Graph-based algorithm for keyword ranking
- **LDA**: Latent Dirichlet Allocation for topic-based keyword extraction

### 📊 Supply Chain Evaluation
- **Multi-dimensional Analysis**: Upstream (raw materials), midstream (processing), downstream (distribution)
- **Quality Metrics**: Comprehensive evaluation of supply chain quality indicators
- **Visualization**: Rich visualizations for analysis results

### 🛠️ Utility Features
- **Data Processing**: Efficient handling of large-scale review datasets
- **Visualization Tools**: Comprehensive plotting and charting capabilities
- **Command-line Interface**: Easy-to-use CLI for batch processing
- **Modular Design**: Flexible and extensible architecture

## 🚀 Quick Start

### Installation

```bash
# Basic installation
pip install chinese-herbal-sentiment

# With deep learning support
pip install chinese-herbal-sentiment[deep_learning]

# With development tools
pip install chinese-herbal-sentiment[dev]

# Complete installation
pip install chinese-herbal-sentiment[all]
```

### Basic Usage

```python
import pandas as pd
from chinese_herbal_sentiment import SentimentAnalyzer, KeywordExtractor

# Sample data
data = pd.DataFrame({
    '评论内容': [
        '这个中药质量很好，效果不错',
        '包装很差，质量一般',
        '服务态度很好，物流快'
    ]
})

# Sentiment analysis
analyzer = SentimentAnalyzer()
sentiment_results = analyzer.analyze_all_methods(data)

# Keyword extraction
extractor = KeywordExtractor()
keyword_results = extractor.extract_all_methods(data, num_keywords=10)

print("Sentiment Results:", sentiment_results.head())
print("Keywords:", keyword_results.head())
```

### Command Line Usage

```bash
# Analyze sentiment
chinese-herbal-analyze data/reviews.xlsx --method all --output results.csv

# Extract keywords
chinese-herbal-keywords data/reviews.xlsx --method tfidf --num_keywords 20

# Full analysis
chinese-herbal-full data/reviews.xlsx --mode all --output_dir results/
```

## 📚 Documentation

### API Reference

#### SentimentAnalyzer

```python
from chinese_herbal_sentiment import SentimentAnalyzer

analyzer = SentimentAnalyzer()

# Single method analysis
results = analyzer.analyze_sentiment(data, method='svm')

# All methods analysis
results = analyzer.analyze_all_methods(data)
```

**Methods:**
- `dictionary`: Dictionary-based sentiment analysis
- `svm`: Support Vector Machine classifier
- `naive_bayes`: Naive Bayes classifier
- `logistic_regression`: Logistic Regression classifier
- `all`: All available methods

#### KeywordExtractor

```python
from chinese_herbal_sentiment import KeywordExtractor

extractor = KeywordExtractor()

# Single method extraction
keywords = extractor.extract_keywords(data, method='tfidf', num_keywords=20)

# All methods extraction
keywords = extractor.extract_all_methods(data, num_keywords=20)
```

**Methods:**
- `tfidf`: TF-IDF keyword extraction
- `textrank`: TextRank algorithm
- `lda`: Latent Dirichlet Allocation
- `all`: All available methods

#### Deep Learning Models

```python
from chinese_herbal_sentiment import BERTSentimentAnalyzer, TextCNNSentimentAnalyzer

# BERT analysis
bert_analyzer = BERTSentimentAnalyzer()
bert_results = bert_analyzer.analyze_sentiment(data)

# TextCNN analysis
textcnn_analyzer = TextCNNSentimentAnalyzer()
textcnn_results = textcnn_analyzer.analyze_sentiment(data)
```

### Advanced Usage

#### Custom Analysis Pipeline

```python
from chinese_herbal_sentiment import DataAnalyzer, Visualizer

# Load and preprocess data
data_analyzer = DataAnalyzer()
data = data_analyzer.load_data('reviews.xlsx', sample_size=10000)

# Perform analysis
sentiment_results = analyzer.analyze_all_methods(data)
keyword_results = extractor.extract_all_methods(data)

# Generate visualizations
visualizer = Visualizer()
visualizer.plot_sentiment_distribution(sentiment_results, save_path='sentiment.png')
visualizer.plot_keyword_cloud(keyword_results, save_path='keywords.png')
```

#### Supply Chain Quality Evaluation

```python
from chinese_herbal_sentiment.utils.keyword_mapping import KeywordMapper

# Map keywords to supply chain dimensions
mapper = KeywordMapper()
supply_chain_results = mapper.map_keywords_to_dimensions(keyword_results)

# Analyze quality indicators
quality_metrics = mapper.calculate_quality_metrics(supply_chain_results)
```

## 📊 Output Examples

### Sentiment Analysis Results

| 评论内容 | dictionary_sentiment | svm_sentiment | naive_bayes_sentiment | logistic_regression_sentiment |
|----------|---------------------|---------------|----------------------|------------------------------|
| 质量很好，效果不错 | positive | positive | positive | positive |
| 包装很差，质量一般 | negative | negative | negative | negative |
| 服务态度很好 | positive | positive | positive | positive |

### Keyword Extraction Results

| keyword | score | method |
|---------|-------|--------|
| 质量 | 0.85 | TF-IDF |
| 包装 | 0.72 | TF-IDF |
| 服务 | 0.68 | TF-IDF |
| 效果 | 0.65 | TextRank |
| 物流 | 0.58 | TextRank |

## 🔧 Configuration

### Data Format

The package expects data in the following format:

```python
# Excel/CSV file with columns:
data = pd.DataFrame({
    '评论内容': ['review text 1', 'review text 2', ...],
    '评分': [5, 4, 3, ...],  # Optional
    '时间': ['2024-01-01', '2024-01-02', ...],  # Optional
    '用户ID': ['user1', 'user2', ...]  # Optional
})
```

### Model Configuration

```python
# Custom model parameters
analyzer = SentimentAnalyzer(
    vectorizer_params={'max_features': 5000},
    classifier_params={'C': 1.0}
)

extractor = KeywordExtractor(
    tfidf_params={'max_features': 1000},
    textrank_params={'window_size': 4}
)
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=chinese_herbal_sentiment

# Run specific test file
pytest tests/test_sentiment_analysis.py
```

## 📈 Performance

### Accuracy Comparison

| Method | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| Dictionary | 0.72 | 0.71 | 0.72 | 0.71 |
| SVM | 0.85 | 0.84 | 0.85 | 0.84 |
| Naive Bayes | 0.82 | 0.81 | 0.82 | 0.81 |
| Logistic Regression | 0.87 | 0.86 | 0.87 | 0.86 |
| BERT | 0.91 | 0.90 | 0.91 | 0.90 |
| TextCNN | 0.89 | 0.88 | 0.89 | 0.88 |

### Processing Speed

- **Small dataset (< 1K reviews)**: ~1-2 seconds
- **Medium dataset (1K-10K reviews)**: ~10-30 seconds
- **Large dataset (> 10K reviews)**: ~2-5 minutes

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/chenxingqiang/chinese-herbal-sentiment.git
cd chinese-herbal-sentiment

# Install in development mode
pip install -e .[dev]

# Run tests
pytest

# Format code
black chinese_herbal_sentiment tests

# Lint code
flake8 chinese_herbal_sentiment tests
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Research Foundation**: Based on master's thesis research on Chinese herbal medicine e-commerce supply chain quality evaluation
- **Open Source Libraries**: Built on top of scikit-learn, transformers, PyTorch, and other excellent open-source projects
- **Academic Community**: Inspired by research in sentiment analysis and supply chain management

## 📞 Support

- **Documentation**: [GitHub Wiki](https://github.com/chenxingqiang/chinese-herbal-sentiment/wiki)
- **Issues**: [GitHub Issues](https://github.com/chenxingqiang/chinese-herbal-sentiment/issues)
- **Email**: chenxingqiang@example.com

## 🔄 Changelog

### v0.1.0 (2024-12-XX)
- Initial release
- Basic sentiment analysis (dictionary, SVM, Naive Bayes, Logistic Regression)
- Keyword extraction (TF-IDF, TextRank, LDA)
- Deep learning models (BERT, TextCNN, TextRank)
- Command-line interface
- Comprehensive documentation and examples

---

**Note**: This package is designed specifically for Chinese herbal medicine e-commerce review analysis and supply chain quality evaluation. For general sentiment analysis tasks, consider using more general-purpose NLP libraries.
