"""
Utility modules for Chinese Herbal Medicine Sentiment Analysis.

This module contains utility functions for:
- Data analysis and preprocessing
- Dataset loading from Hugging Face
- Data reading and processing
- Academic search tools
"""

# Core utility classes and functions
from .dataset_loader import DatasetLoader, load_chinese_herbal_dataset

# Data analysis functions
from .data_analysis import load_excel_files, preprocess_text, extract_keywords, analyze_sentiment_distribution

# Academic search functions  
from .scholar_search import search_google_scholar, generate_tcm_search_queries

__all__ = [
    # Core classes
    "DatasetLoader",
    "load_chinese_herbal_dataset",
    
    # Data analysis
    "load_excel_files",
    "preprocess_text", 
    "extract_keywords",
    "analyze_sentiment_distribution",
    
    # Academic search
    "search_google_scholar",
    "generate_tcm_search_queries"
]
