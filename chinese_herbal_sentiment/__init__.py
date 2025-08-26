"""
Chinese Herbal Medicine E-commerce Sentiment Analysis System

A comprehensive NLP toolkit for analyzing customer reviews and evaluating 
supply chain quality in Chinese herbal medicine e-commerce.

Author: Luo Jiawen, Chen Xingqiang
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Luo Jiawen, Chen Xingqiang"
__email__ = "chenxingqiang@turingai.cc"

# Import core modules (without TensorFlow dependencies)
from .core.sentiment_analysis import SentimentAnalysis
from .core.keyword_extraction import KeywordExtraction
from .core.regression_analysis import SupplyChainRegression
from .core.prediction_service import PredictionService, ModelManager
from .core.time_series_analysis import TimeSeriesAnalyzer
from .utils.dataset_loader import DatasetLoader, load_chinese_herbal_dataset

# Optional imports with error handling (avoid hard dependencies)
try:
    from .core.bert_sentiment_analysis import BERTSentimentAnalysis
    _BERT_AVAILABLE = True
except ImportError:
    _BERT_AVAILABLE = False
    BERTSentimentAnalysis = None

try:
    from .core.deep_learning_sentiment import DeepLearningSentiment
    _TENSORFLOW_AVAILABLE = True
except ImportError:
    _TENSORFLOW_AVAILABLE = False
    DeepLearningSentiment = None

try:
    from .core.textcnn_sentiment_analysis import TextCNNSentimentAnalysis
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    TextCNNSentimentAnalysis = None

# 兼容性别名
SentimentAnalyzer = SentimentAnalysis
KeywordExtractor = KeywordExtraction

__all__ = [
    # Core analysis classes
    "SentimentAnalysis",
    "SentimentAnalyzer",  # 兼容性别名
    "KeywordExtraction", 
    "KeywordExtractor",   # 兼容性别名
    
    # Advanced analysis classes
    "SupplyChainRegression",
    "PredictionService",
    "ModelManager",
    "TimeSeriesAnalyzer",
    
    # Data utilities
    "DatasetLoader",
    "load_chinese_herbal_dataset",
    
    # Metadata
    "__version__",
    "__author__",
    "__email__"
]

# Conditionally add optional modules if available
if _BERT_AVAILABLE:
    __all__.append("BERTSentimentAnalysis")

if _TENSORFLOW_AVAILABLE:
    __all__.append("DeepLearningSentiment")

if _TORCH_AVAILABLE:
    __all__.append("TextCNNSentimentAnalysis")
