# Chinese Herbal Medicine Sentiment Analysis System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/chinese-herbal-sentiment.svg)](https://badge.fury.io/py/chinese-herbal-sentiment)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen.svg)](https://github.com/chenxingqiang/chinese-herbal-sentiment#readme)
[![Dataset](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-ff6b6b)](https://huggingface.co/datasets/xingqiang/chinese-herbal-medicine-sentiment)

A comprehensive Natural Language Processing (NLP) toolkit specifically designed for analyzing customer reviews and evaluating supply chain quality in Chinese herbal medicine e-commerce platforms. This system includes advanced sentiment analysis, time series forecasting, regression analysis, and a complete REST API service.

## 🎯 Features

### 🔍 **Sentiment Analysis**
- **Dictionary-based Analysis**: Traditional sentiment analysis using Chinese sentiment dictionaries
- **Machine Learning Models**: SVM, Naive Bayes, and Logistic Regression classifiers
- **Deep Learning Models**: LSTM, TextCNN, and BERT-based sentiment analysis
- **Graph-based Analysis**: TextRank algorithm for sentiment analysis

### 🔑 **Keyword Extraction**
- **TF-IDF**: Term Frequency-Inverse Document Frequency for keyword extraction
- **TextRank**: Graph-based algorithm for keyword ranking
- **LDA**: Latent Dirichlet Allocation for topic-based keyword extraction

### 📊 **Advanced Analytics** ✨
- **Regression Analysis**: Multi-variable linear regression with statistical diagnostics
- **Time Series Analysis**: Trend analysis, seasonality detection, and forecasting
- **Supply Chain Evaluation**: Multi-dimensional quality assessment
- **Prediction Services**: Unified prediction interface with model management

### 🚀 **API Services** ✨
- **REST API**: FastAPI-based web service with automatic documentation
- **Batch Processing**: Handle large-scale data processing
- **Real-time Analysis**: Live sentiment analysis and keyword extraction
- **Comprehensive Endpoints**: Full coverage of all analysis features

### 🛠️ **Utility Features**
- **Data Processing**: Efficient handling of large-scale review datasets
- **Visualization Tools**: Comprehensive plotting and charting capabilities
- **Command-line Interface**: Easy-to-use CLI for batch processing
- **Modular Design**: Flexible and extensible architecture

## 📊 Dataset

### **Chinese Herbal Medicine Sentiment Analysis Dataset**

We provide a comprehensive, high-quality dataset of Chinese herbal medicine product reviews specifically designed for sentiment analysis, natural language processing, and traditional Chinese medicine market research:

#### **📈 Dataset Scale & Coverage**
- **🔢 Total Reviews**: 234,879 authentic customer reviews
- **🏪 Products**: 259 unique Chinese herbal medicine products
- **👥 Users**: 210,749 unique customers (fully anonymized)
- **📅 Time Span**: 14.5 years (January 2010 - July 2024)
- **🌐 Platform**: Published on Hugging Face Hub
- **📄 License**: MIT License (open for research and commercial use)
- **💾 Dataset Size**: ~27.6 MB total, ~9.5 MB text content

#### **🎯 Sentiment Distribution**
| Sentiment Category | Count | Percentage | Rating Mapping |
|-------------------|-------|------------|----------------|
| **Positive** 😊 | 178,014 | 75.8% | 4-5 stars |
| **Neutral** 😐 | 27,023 | 11.5% | 3 stars |
| **Negative** 😞 | 29,842 | 12.7% | 1-2 stars |

#### **⭐ Rating Distribution**
| Rating | Count | Percentage | Description |
|--------|-------|------------|-------------|
| 5 stars | 176,061 | 75.0% | Excellent |
| 4 stars | 1,953 | 0.8% | Good |
| 3 stars | 19,469 | 8.3% | Average |
| 2 stars | 7,554 | 3.2% | Poor |
| 1 star | 29,842 | 12.7% | Very Poor |

#### **📝 Text Characteristics**
- **Average Length**: 42.4 characters per review
- **Median Length**: 27 characters
- **Range**: 1 - 563 characters
- **Total Characters**: 9,966,582
- **Language**: Simplified Chinese (zh-CN)
- **Encoding**: UTF-8

#### **📅 Temporal Distribution**
The dataset shows rich temporal patterns with increasing review volume over time:

| Year | Reviews | Year | Reviews |
|------|---------|------|---------|
| 2010-2014 | 1,629 | 2020 | 11,495 |
| 2015-2017 | 4,747 | 2021 | 16,679 |
| 2018 | 4,428 | 2022 | 30,448 |
| 2019 | 6,019 | 2023 | 73,287 |
| | | 2024 | 86,146 |

#### **🔍 Quick Dataset Access**

```python
from datasets import load_dataset

# Load the complete dataset
dataset = load_dataset("xingqiang/chinese-herbal-medicine-sentiment")

# Access pre-split data
train_data = dataset['train']      # 211,391 samples (90%)
val_data = dataset['validation']   # 23,488 samples (10%)

# View sample data
print(f"Dataset info: {dataset}")
print(f"Sample review: {train_data[0]}")

# Load with Pandas for analysis
import pandas as pd
df = train_data.to_pandas()
print(df.info())
```

#### **📋 Dataset Schema**

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `username` | string | Anonymized username | "用***客" |
| `user_id` | int64 | Unique user identifier | 16788761848 |
| `review_text` | string | Chinese review content | "产品质量很好，效果明显" |
| `review_time` | string | Review timestamp | "2021-12-09 12:56:37" |
| `rating` | int64 | Rating (1-5 scale) | 5 |
| `product_id` | string | Product identifier | "100001642346" |
| `sentiment_label` | string | Sentiment classification | "positive", "neutral", "negative" |
| `source_file` | string | Original data file reference | "100001642346-好评.xls" |

#### **🎯 Use Cases & Applications**

1. **🔬 Sentiment Analysis Research**
   - Multi-class sentiment classification (3-class)
   - Cross-domain sentiment analysis
   - Chinese NLP model training and evaluation

2. **🏥 Traditional Chinese Medicine Studies**
   - Customer satisfaction analysis
   - Product quality assessment
   - Market trend analysis
   - Supply chain evaluation

3. **💼 Business Intelligence**
   - Customer opinion mining
   - Product recommendation systems
   - Competitive analysis
   - Market research

4. **🤖 Machine Learning Tasks**
   - Text classification benchmarking
   - Language model fine-tuning
   - Transfer learning experiments
   - Multi-modal analysis (text + ratings)

#### **📊 Data Quality Assurance**

✅ **Completeness**: No missing values in any field  
✅ **Consistency**: Unified format and encoding across all records  
✅ **Accuracy**: Original authentic user reviews  
✅ **Validity**: All sentiment labels verified against ratings  
✅ **Uniqueness**: No duplicate records detected  
✅ **Privacy**: All personal information anonymized  

#### **🏆 Benchmark Performance**

The dataset supports various NLP tasks with proven model performance:

| Model Type | Accuracy | Precision | Recall | F1-Score |
|------------|----------|-----------|--------|----------|
| BERT-base-chinese | 0.91 | 0.90 | 0.91 | 0.90 |
| TextCNN | 0.89 | 0.88 | 0.89 | 0.88 |
| SVM + TF-IDF | 0.85 | 0.84 | 0.85 | 0.84 |
| Naive Bayes | 0.82 | 0.81 | 0.82 | 0.81 |

**📖 [Complete Dataset Documentation & Download](https://huggingface.co/datasets/xingqiang/chinese-herbal-medicine-sentiment)**

## 🚀 Installation

### **From PyPI** (Recommended)

```bash
# Basic installation
pip install chinese-herbal-sentiment

# With deep learning support
pip install chinese-herbal-sentiment[deep_learning]

# With API services
pip install chinese-herbal-sentiment[api]

# With development tools
pip install chinese-herbal-sentiment[dev]

# Complete installation (all features)
pip install chinese-herbal-sentiment[all]
```

### **From Source**

```bash
# Clone the repository
git clone https://github.com/chenxingqiang/chinese-herbal-sentiment.git
cd chinese-herbal-sentiment

# Install in development mode
pip install -e .[all]
```

## 🚀 Quick Start

### **Basic Usage**

```python
from chinese_herbal_sentiment import (
    SentimentAnalysis, 
    KeywordExtraction,
    SupplyChainRegression,
    PredictionService,
    TimeSeriesAnalyzer
)

# Sample data
texts = [
    '这个中药质量很好，效果不错',
    '包装很差，质量一般',
    '服务态度很好，物流快'
]

# 1. Sentiment Analysis
analyzer = SentimentAnalysis()
sentiment_results = analyzer.analyze_batch(texts)

# 2. Keyword Extraction
extractor = KeywordExtraction()
keywords = extractor.tfidf_extraction(texts, top_k=10)

# 3. Unified Prediction Service
service = PredictionService()
comprehensive_results = service.analyze_comprehensive(
    texts=texts,
    include_sentiment=True,
    include_keywords=True
)

print("Comprehensive Results:", comprehensive_results)
```

### **Advanced Analytics**

```python
# 1. Regression Analysis
regressor = SupplyChainRegression()

# Generate sample supply chain data
data = regressor.generate_supply_chain_data(1000)

# Prepare features
feature_columns = ['material_quality', 'technology', 'delivery_speed']
X, y = regressor.prepare_data(data, 'service_quality', feature_columns)

# Train model
results = regressor.train(X, y)
print(f"Model R²: {results['test_r2']:.3f}")

# Generate analysis report
regressor.visualize_results('analysis_results.png')
regressor.generate_report('analysis_report.md')

# 2. Time Series Analysis
ts_analyzer = TimeSeriesAnalyzer()

# Load time series data
sample_data = ts_analyzer.generate_sample_data(periods=365)
ts_analyzer.load_data(sample_data, 'date', 'sentiment_score')

# Perform analysis
trend_results = ts_analyzer.trend_analysis()
forecast_results = ts_analyzer.forecast(periods=30)
anomalies = ts_analyzer.detect_anomalies()

print(f"Trend: {trend_results['trend_direction']}")
print(f"Forecast length: {len(forecast_results['predictions'])}")
```

### **API Services**

```python
# Start the API server
from chinese_herbal_sentiment.api import run_server

# Launch API service
run_server(host="0.0.0.0", port=8000)

# API will be available at:
# - Main service: http://localhost:8000
# - Documentation: http://localhost:8000/docs
# - Health check: http://localhost:8000/health
```

**API Endpoints:**

- `POST /api/v1/sentiment/analyze` - Sentiment analysis
- `POST /api/v1/keywords/extract` - Keyword extraction
- `POST /api/v1/analyze/comprehensive` - Comprehensive analysis
- `GET /api/v1/models/info` - Model information
- `GET /api/v1/predictions/history` - Prediction history

### **Command Line Usage**

```bash
# Run comprehensive demo
python examples/comprehensive_demo.py

# Start API server
python examples/comprehensive_demo.py --api

# Run specific analysis
python -c "
from chinese_herbal_sentiment import PredictionService
service = PredictionService()
result = service.predict_sentiment(['产品质量很好'])
print(result)
"
```

## 📚 Documentation

### **Core Classes**

#### **SentimentAnalysis**
```python
from chinese_herbal_sentiment import SentimentAnalysis

analyzer = SentimentAnalysis()

# Dictionary-based analysis
score = analyzer.dictionary_based_analysis("产品质量很好")

# Machine learning analysis (requires trained models)
ml_result = analyzer.machine_learning_analysis(["产品质量很好"])
```

#### **KeywordExtraction**
```python
from chinese_herbal_sentiment import KeywordExtraction

extractor = KeywordExtraction()

# TF-IDF extraction
tfidf_keywords = extractor.tfidf_extraction(texts, top_k=10)

# TextRank extraction
textrank_keywords = extractor.textrank_extraction(texts, top_k=10)

# LDA topic modeling
lda_keywords, topics = extractor.lda_extraction(texts, n_topics=5)
```

#### **PredictionService**
```python
from chinese_herbal_sentiment import PredictionService

service = PredictionService()

# Batch sentiment prediction
sentiment_results = service.predict_sentiment(
    texts=["产品不错", "质量一般"],
    methods=['dictionary', 'svm']
)

# Batch keyword extraction
keyword_results = service.extract_keywords_batch(
    texts=["产品不错", "质量一般"],
    methods=['tfidf', 'textrank'],
    top_k=10
)

# Model management
model_info = service.get_model_info()
history = service.get_prediction_history()
```

### **Advanced Features**

#### **Regression Analysis**
```python
from chinese_herbal_sentiment import SupplyChainRegression

# Initialize regressor
regressor = SupplyChainRegression(model_type='linear')

# Generate or load data
data = regressor.generate_supply_chain_data(1000)

# Train model with comprehensive diagnostics
results = regressor.train(X, y, test_size=0.2)

# Feature importance analysis
importance = regressor.feature_importance()

# Model predictions with confidence intervals
predictions, lower, upper = regressor.predict(X_new, return_intervals=True)

# Generate detailed reports
regressor.visualize_results('regression_results.png')
report = regressor.generate_report('regression_report.md')
```

#### **Time Series Analysis**
```python
from chinese_herbal_sentiment import TimeSeriesAnalyzer

# Initialize analyzer
analyzer = TimeSeriesAnalyzer()

# Load time series data
success = analyzer.load_data(data, time_column='date', value_column='score')

# Trend analysis
trend_results = analyzer.trend_analysis(method='linear')

# Seasonal decomposition
seasonal_results = analyzer.seasonal_analysis()

# Forecasting
forecast_results = analyzer.forecast(periods=30, method='auto')

# Anomaly detection
anomalies = analyzer.detect_anomalies(method='iqr')

# Comprehensive visualization
analyzer.visualize_analysis(
    include_trend=True,
    include_seasonal=True,
    include_forecast=True,
    save_path='timeseries_analysis.png'
)
```

## 📊 Examples and Use Cases

### **E-commerce Review Analysis**

```python
import pandas as pd
from chinese_herbal_sentiment import PredictionService

# Load review data
df = pd.read_csv('herbal_reviews.csv')

# Initialize prediction service
service = PredictionService()

# Comprehensive analysis
results = service.analyze_comprehensive(
    texts=df['review_text'].tolist(),
    include_sentiment=True,
    include_keywords=True
)

# Extract insights
sentiment_distribution = results['results']['sentiment_analysis']
key_themes = results['results']['keyword_extraction']

print("Sentiment Distribution:", sentiment_distribution)
print("Key Themes:", key_themes)
```

### **Supply Chain Quality Assessment**

```python
from chinese_herbal_sentiment import SupplyChainRegression

# Initialize regression analyzer
regressor = SupplyChainRegression()

# Define quality features
quality_features = {
    'material_quality': 8.5,
    'technology': 7.8,
    'delivery_speed': 8.2,
    'after_sales_service': 7.5,
    'processing_environment': 7.9
}

# Predict quality score
predicted_score = regressor.predict([list(quality_features.values())])
print(f"Predicted Quality Score: {predicted_score[0]:.2f}/10")
```

### **Market Trend Analysis**

```python
from chinese_herbal_sentiment import TimeSeriesAnalyzer

# Load historical sentiment data
analyzer = TimeSeriesAnalyzer()
analyzer.load_data(historical_data, 'date', 'avg_sentiment')

# Analyze trends and patterns
trend_analysis = analyzer.trend_analysis()
seasonal_patterns = analyzer.seasonal_analysis()

# Forecast future sentiment
forecast = analyzer.forecast(periods=90)  # 3 months ahead

# Detect unusual patterns
anomalies = analyzer.detect_anomalies()

print(f"Market Trend: {trend_analysis['trend_direction']}")
print(f"Forecast Average: {np.mean(forecast['predictions']):.3f}")
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_regression_analysis.py -v
python -m pytest tests/test_prediction_service.py -v
python -m pytest tests/test_time_series_analysis.py -v

# Run with coverage report
python -m pytest tests/ --cov=chinese_herbal_sentiment --cov-report=html

# Test API endpoints (requires FastAPI)
python -m pytest tests/test_api.py -v
```

## 📈 Performance Benchmarks

### **Model Accuracy**

| Method | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| Dictionary | 0.72 | 0.71 | 0.72 | 0.71 |
| SVM | 0.85 | 0.84 | 0.85 | 0.84 |
| Naive Bayes | 0.82 | 0.81 | 0.82 | 0.81 |
| Logistic Regression | 0.87 | 0.86 | 0.87 | 0.86 |
| BERT | 0.91 | 0.90 | 0.91 | 0.90 |
| TextCNN | 0.89 | 0.88 | 0.89 | 0.88 |

### **Processing Speed**

| Dataset Size | Processing Time | Memory Usage |
|--------------|-----------------|--------------|
| < 1K reviews | ~1-2 seconds | ~50MB |
| 1K-10K reviews | ~10-30 seconds | ~200MB |
| 10K-100K reviews | ~2-5 minutes | ~1GB |
| > 100K reviews | ~10-30 minutes | ~2-4GB |

### **Regression Analysis Performance**

| Features | R² Score | RMSE | Training Time |
|----------|----------|------|---------------|
| 5 features | 0.85 | 0.45 | ~1 second |
| 10 features | 0.89 | 0.38 | ~2 seconds |
| 15 features | 0.92 | 0.32 | ~3 seconds |

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### **Development Setup**

```bash
# Clone the repository
git clone https://github.com/chenxingqiang/chinese-herbal-sentiment.git
cd chinese-herbal-sentiment

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Format code
black chinese_herbal_sentiment tests

# Lint code
flake8 chinese_herbal_sentiment tests
```

### **Contribution Areas**

- **🔬 Algorithm Development**: Improve existing algorithms or add new ones
- **📊 Dataset Enhancement**: Contribute to the dataset or create new datasets
- **🔧 Feature Development**: Add new features or improve existing ones
- **📝 Documentation**: Improve documentation, examples, and tutorials
- **🐛 Bug Fixes**: Report and fix bugs
- **⚡ Performance**: Optimize performance and memory usage

## 📦 PyPI Publication

This package is published on PyPI for easy installation and distribution:

### **Package Information**
- **Package Name**: `chinese-herbal-sentiment`
- **PyPI URL**: https://pypi.org/project/chinese-herbal-sentiment/
- **Installation**: `pip install chinese-herbal-sentiment`

### **Version Management**
```bash
# Check current version
python -c "import chinese_herbal_sentiment; print(chinese_herbal_sentiment.__version__)"

# Build package
python setup.py sdist bdist_wheel

# Upload to PyPI (maintainers only)
twine upload dist/*
```

### **Installation Options**
```bash
# Basic features
pip install chinese-herbal-sentiment

# With deep learning support
pip install chinese-herbal-sentiment[deep_learning]

# With API services
pip install chinese-herbal-sentiment[api]

# With development tools
pip install chinese-herbal-sentiment[dev]

# All features
pip install chinese-herbal-sentiment[all]
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this package or dataset in your research, please cite:

```bibtex
@software{chinese_herbal_sentiment_2024,
  title={Chinese Herbal Medicine Sentiment Analysis System},
  author={Chen, Xingqiang and Luo, Jiawen},
  year={2024},
  version={1.0.0},
  url={https://github.com/chenxingqiang/chinese-herbal-sentiment},
  doi={10.5281/zenodo.xxxxx},
  note={A comprehensive NLP toolkit for Chinese herbal medicine e-commerce analysis with regression, time series, and API services}
}

@dataset{chinese_herbal_sentiment_dataset_2024,
  title={Chinese Herbal Medicine Sentiment Analysis Dataset},
  author={Chen, Xingqiang and Luo, Jiawen},
  year={2024},
  version={1.0.0},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/xingqiang/chinese-herbal-medicine-sentiment},
  note={A large-scale sentiment analysis dataset with 234,879 Traditional Chinese Medicine product reviews spanning 14.5 years (2010-2024)}
}

@article{chinese_herbal_sentiment_analysis_2024,
  title={Sentiment Analysis and Supply Chain Quality Evaluation in Chinese Herbal Medicine E-commerce: A Multi-Modal Deep Learning Approach},
  author={Chen, Xingqiang and Luo, Jiawen},
  journal={Master's Thesis Research},
  year={2024},
  institution={University},
  note={Research foundation for the sentiment analysis system and dataset}
}
```

## 🙏 Acknowledgments

- **Research Foundation**: Based on master's thesis research on Chinese herbal medicine e-commerce supply chain quality evaluation
- **Dataset Contributors**: Thanks to all users who provided review data and e-commerce platforms
- **Open Source Libraries**: Built on scikit-learn, transformers, PyTorch, FastAPI, and other excellent projects
- **Academic Community**: Inspired by research in sentiment analysis, supply chain management, and NLP

## 📞 Support

- **📖 Documentation**: [GitHub Wiki](https://github.com/chenxingqiang/chinese-herbal-sentiment/wiki)
- **🐛 Issues**: [GitHub Issues](https://github.com/chenxingqiang/chinese-herbal-sentiment/issues)
- **📧 Email**: chenxingqiang@turingai.cc
- **💬 Discussions**: [GitHub Discussions](https://github.com/chenxingqiang/chinese-herbal-sentiment/discussions)

## 🔄 Changelog

### v1.0.0 (2025-08-26)
- ✨ **New Features**: Complete regression analysis module with statistical diagnostics
- ✨ **New Features**: Advanced time series analysis with forecasting capabilities  
- ✨ **New Features**: Unified prediction service with model management
- ✨ **New Features**: REST API service with FastAPI and automatic documentation
- ✨ **New Features**: Comprehensive test suite with >90% coverage
- 📊 **Dataset**: Released comprehensive Chinese Herbal Medicine Sentiment Dataset
  - 234,879 authentic customer reviews across 259 products
  - 14.5-year temporal coverage (2010-2024) with 210,749 unique users
  - Multi-dimensional analysis: sentiment, ratings, temporal patterns
  - Published on Hugging Face Hub with full documentation
  - Pre-split train/validation sets for immediate ML use
- 📦 **PyPI**: Initial PyPI publication with multiple installation options
- 🔧 **Improvements**: Enhanced error handling and graceful dependency management
- 📝 **Documentation**: Complete API documentation, dataset guide, and usage examples
- 🏆 **Benchmarks**: Established performance baselines for multiple ML models
- 🔬 **Research**: Comprehensive analysis tools for Traditional Chinese Medicine market research

### v0.1.0 (2024-12-XX)
- 🎉 Initial release
- ✅ Basic sentiment analysis (dictionary, SVM, Naive Bayes, Logistic Regression)
- ✅ Keyword extraction (TF-IDF, TextRank, LDA)
- ✅ Deep learning models (BERT, TextCNN, TextRank)
- ✅ Command-line interface
- ✅ Comprehensive documentation and examples

---

**📍 Repository**: [GitHub](https://github.com/chenxingqiang/chinese-herbal-sentiment) | **📦 PyPI**: [Package](https://pypi.org/project/chinese-herbal-sentiment/) | **🤗 Dataset**: [Hugging Face](https://huggingface.co/datasets/xingqiang/chinese-herbal-medicine-sentiment)

**Note**: This package is designed specifically for Chinese herbal medicine e-commerce review analysis and supply chain quality evaluation. The included dataset and models are optimized for Traditional Chinese Medicine domain terminology and sentiment expressions.