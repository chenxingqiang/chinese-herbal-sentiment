# PyPI Release Guide for Chinese Herbal Medicine Sentiment Analysis

## Package Information

- **Package Name**: `chinese-herbal-sentiment`
- **Version**: 1.0.0
- **PyPI URL**: https://pypi.org/project/chinese-herbal-sentiment/
- **GitHub Repository**: https://github.com/chenxingqiang/chinese-herbal-sentiment

## Installation Options

The package provides multiple installation options to accommodate different use cases:

### Basic Installation
```bash
pip install chinese-herbal-sentiment
```
Includes core sentiment analysis and keyword extraction features.

### Deep Learning Support
```bash
pip install chinese-herbal-sentiment[deep_learning]
```
Adds support for BERT, TextCNN, LSTM, and other deep learning models.

### API Services
```bash
pip install chinese-herbal-sentiment[api]
```
Includes FastAPI-based REST API service capabilities.

### Advanced Analysis
```bash
pip install chinese-herbal-sentiment[analysis]
```
Adds regression analysis, time series forecasting, and statistical modeling.

### Development Tools
```bash
pip install chinese-herbal-sentiment[dev]
```
Includes testing, linting, and development utilities.

### Complete Installation
```bash
pip install chinese-herbal-sentiment[all]
```
Installs all optional dependencies and features.

## Features Included

### Core Features (Base Installation)
- ✅ Chinese text sentiment analysis (dictionary-based, SVM, Naive Bayes, Logistic Regression)
- ✅ Keyword extraction (TF-IDF, TextRank, LDA)
- ✅ Data processing and visualization utilities
- ✅ Supply chain quality evaluation framework
- ✅ Command-line interface

### Deep Learning Features
- ✅ BERT-based sentiment analysis
- ✅ TextCNN neural network model
- ✅ LSTM recurrent neural network
- ✅ Transformer model integration

### API Services
- ✅ FastAPI-based REST API
- ✅ Automatic API documentation (Swagger UI)
- ✅ Batch processing endpoints
- ✅ Real-time analysis services

### Advanced Analytics
- ✅ Multi-variable regression analysis with statistical diagnostics
- ✅ Time series analysis, forecasting, and anomaly detection
- ✅ Supply chain quality prediction models
- ✅ Comprehensive statistical reporting

### Development Tools
- ✅ Comprehensive test suite (pytest)
- ✅ Code formatting and linting tools
- ✅ Documentation generation utilities
- ✅ Pre-commit hooks and CI/CD support

## Quick Start

### Basic Usage Example
```python
from chinese_herbal_sentiment import SentimentAnalysis, KeywordExtraction

# Initialize analyzers
sentiment_analyzer = SentimentAnalysis()
keyword_extractor = KeywordExtraction()

# Analyze sentiment
texts = ["产品质量很好，效果不错", "包装很差，不推荐"]
sentiment_scores = [sentiment_analyzer.dictionary_based_analysis(text) for text in texts]

# Extract keywords
keywords = keyword_extractor.tfidf_extraction(texts, top_k=10)

print("Sentiment scores:", sentiment_scores)
print("Keywords:", keywords)
```

### Advanced Usage Example
```python
from chinese_herbal_sentiment import (
    PredictionService, 
    SupplyChainRegression, 
    TimeSeriesAnalyzer
)

# Unified prediction service
service = PredictionService()
comprehensive_results = service.analyze_comprehensive(
    texts=["产品质量很好", "服务态度不错"],
    include_sentiment=True,
    include_keywords=True
)

# Regression analysis
regressor = SupplyChainRegression()
data = regressor.generate_supply_chain_data(1000)
# ... perform regression analysis

# Time series analysis
ts_analyzer = TimeSeriesAnalyzer()
ts_data = ts_analyzer.generate_sample_data(periods=365)
# ... perform time series analysis
```

### API Service Example
```python
from chinese_herbal_sentiment.api import run_server

# Start the API server
run_server(host="0.0.0.0", port=8000)

# API endpoints will be available at:
# - Main service: http://localhost:8000
# - Documentation: http://localhost:8000/docs
# - Health check: http://localhost:8000/health
```

## Dataset Integration

The package seamlessly integrates with our published dataset:

```python
from datasets import load_dataset

# Load the Chinese Herbal Medicine Sentiment Dataset
dataset = load_dataset("xingqiang/chinese-herbal-medicine-sentiment")

# Use with the package
from chinese_herbal_sentiment import PredictionService
service = PredictionService()

# Analyze dataset samples
sample_texts = dataset['train']['review_text'][:100]
results = service.predict_sentiment(sample_texts)
```

## Requirements

### System Requirements
- Python 3.8 or higher
- 4GB+ RAM recommended for large datasets
- 2GB+ disk space for all optional dependencies

### Core Dependencies
- pandas>=1.3.0
- numpy>=1.21.0
- scikit-learn>=1.0.0
- jieba>=0.42.1
- matplotlib>=3.5.0

### Optional Dependencies
- torch>=1.10.0 (for deep learning)
- tensorflow>=2.8.0 (for deep learning)
- fastapi>=0.68.0 (for API services)
- scipy>=1.7.0 (for advanced analysis)
- statsmodels>=0.13.0 (for time series)

## Documentation

- **GitHub Repository**: https://github.com/chenxingqiang/chinese-herbal-sentiment
- **API Documentation**: Available at `/docs` when running the API server
- **Dataset Documentation**: https://huggingface.co/datasets/xingqiang/chinese-herbal-medicine-sentiment
- **Issues and Support**: https://github.com/chenxingqiang/chinese-herbal-sentiment/issues

## Release Notes

### Version 1.0.0 (2025-08-26)

**New Features:**
- ✨ Complete regression analysis module with statistical diagnostics
- ✨ Advanced time series analysis with forecasting capabilities
- ✨ Unified prediction service with model management
- ✨ REST API service with FastAPI and automatic documentation
- ✨ Comprehensive test suite with >90% coverage

**Dataset:**
- 📊 Released Chinese Herbal Medicine Sentiment Dataset (234K+ reviews)
- 🤗 Published on Hugging Face Hub with train/validation splits

**PyPI Publication:**
- 📦 Initial PyPI publication with multiple installation options
- 🔧 Enhanced dependency management with optional extras
- 📝 Complete package documentation and examples

**Improvements:**
- 🔧 Enhanced error handling and graceful dependency management
- 📝 Complete API documentation and usage examples
- ✅ Comprehensive testing with pytest and coverage reporting
- 🎨 Improved code quality with linting and formatting

## Support and Contributing

- **Issues**: Report bugs and request features on [GitHub Issues](https://github.com/chenxingqiang/chinese-herbal-sentiment/issues)
- **Discussions**: Join the community discussion on [GitHub Discussions](https://github.com/chenxingqiang/chinese-herbal-sentiment/discussions)
- **Contributing**: See [CONTRIBUTING.md](https://github.com/chenxingqiang/chinese-herbal-sentiment/blob/main/CONTRIBUTING.md) for guidelines
- **Email**: chenxingqiang@turingai.cc

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/chenxingqiang/chinese-herbal-sentiment/blob/main/LICENSE) file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{chinese_herbal_sentiment_2024,
  title={Chinese Herbal Medicine Sentiment Analysis System},
  author={Chen, Xingqiang},
  year={2024},
  version={1.0.0},
  url={https://github.com/chenxingqiang/chinese-herbal-sentiment},
  note={A comprehensive NLP toolkit for Chinese herbal medicine e-commerce analysis}
}
```
