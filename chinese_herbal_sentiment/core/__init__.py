"""
Core analysis modules for Chinese Herbal Medicine Sentiment Analysis.

This module contains the main analysis algorithms including:
- Sentiment analysis (dictionary-based, machine learning, deep learning)
- Keyword extraction (TF-IDF, TextRank, LDA)
- BERT-based analysis
- TextCNN analysis
- TextRank analysis
"""

# Core analysis classes (with actual class names)
from .sentiment_analysis import SentimentAnalysis
from .keyword_extraction import KeywordExtraction

# Optional deep learning imports with error handling
try:
    from .bert_sentiment_analysis import BERTSentimentAnalysis
except ImportError:
    BERTSentimentAnalysis = None

try:
    from .deep_learning_sentiment import DeepLearningSentiment
except ImportError:
    DeepLearningSentiment = None

try:
    from .textcnn_sentiment_analysis import TextCNNSentimentAnalysis
except ImportError:
    TextCNNSentimentAnalysis = None

# Aliases for backward compatibility
SentimentAnalyzer = SentimentAnalysis
KeywordExtractor = KeywordExtraction

__all__ = [
    "SentimentAnalysis",
    "SentimentAnalyzer",  # Alias
    "KeywordExtraction",
    "KeywordExtractor",   # Alias
]

# Conditionally add deep learning classes
if BERTSentimentAnalysis:
    __all__.append("BERTSentimentAnalysis")

if DeepLearningSentiment:
    __all__.append("DeepLearningSentiment")

if TextCNNSentimentAnalysis:
    __all__.append("TextCNNSentimentAnalysis")
