"""
Chinese Herbal Medicine E-commerce Sentiment Analysis System

A comprehensive NLP toolkit for analyzing customer reviews and evaluating 
supply chain quality in Chinese herbal medicine e-commerce.

Author: Xingqiang Chen
Version: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "Luo Chengwen, Chen Xingqiang"
__email__ = "chenxingqiang@example.com"

# Import main modules
from .core.sentiment_analysis import SentimentAnalyzer
from .core.keyword_extraction import KeywordExtractor
from .utils.data_analysis import DataAnalyzer
from .utils.visualization import Visualizer

__all__ = [
    "SentimentAnalyzer",
    "KeywordExtractor", 
    "DataAnalyzer",
    "Visualizer",
    "__version__",
    "__author__",
    "__email__"
]
