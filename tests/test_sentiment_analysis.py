"""
Tests for sentiment analysis functionality.
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

# Add the parent directory to the path to import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from chinese_herbal_sentiment.core.sentiment_analysis import SentimentAnalyzer


class TestSentimentAnalyzer:
    """Test cases for SentimentAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SentimentAnalyzer()
        self.sample_data = pd.DataFrame({
            '评论内容': [
                '这个中药质量很好，效果不错',
                '包装很差，质量一般',
                '服务态度很好，物流快',
                '价格太贵了，不推荐',
                '味道不错，值得购买'
            ]
        })
    
    def test_analyzer_initialization(self):
        """Test that the analyzer can be initialized."""
        assert self.analyzer is not None
        assert hasattr(self.analyzer, 'analyze_sentiment')
    
    def test_dictionary_sentiment_analysis(self):
        """Test dictionary-based sentiment analysis."""
        results = self.analyzer.analyze_sentiment(self.sample_data, method='dictionary')
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == len(self.sample_data)
        assert 'sentiment' in results.columns
    
    def test_svm_sentiment_analysis(self):
        """Test SVM-based sentiment analysis."""
        results = self.analyzer.analyze_sentiment(self.sample_data, method='svm')
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == len(self.sample_data)
        assert 'sentiment' in results.columns
    
    def test_naive_bayes_sentiment_analysis(self):
        """Test Naive Bayes sentiment analysis."""
        results = self.analyzer.analyze_sentiment(self.sample_data, method='naive_bayes')
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == len(self.sample_data)
        assert 'sentiment' in results.columns
    
    def test_logistic_regression_sentiment_analysis(self):
        """Test Logistic Regression sentiment analysis."""
        results = self.analyzer.analyze_sentiment(self.sample_data, method='logistic_regression')
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == len(self.sample_data)
        assert 'sentiment' in results.columns
    
    def test_all_methods_analysis(self):
        """Test analysis with all methods."""
        results = self.analyzer.analyze_all_methods(self.sample_data)
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == len(self.sample_data)
        # Should have columns for each method
        expected_columns = ['dictionary_sentiment', 'svm_sentiment', 'naive_bayes_sentiment', 'logistic_regression_sentiment']
        for col in expected_columns:
            assert col in results.columns
    
    def test_invalid_method(self):
        """Test that invalid method raises appropriate error."""
        with pytest.raises(ValueError):
            self.analyzer.analyze_sentiment(self.sample_data, method='invalid_method')
    
    def test_empty_data(self):
        """Test handling of empty data."""
        empty_data = pd.DataFrame({'评论内容': []})
        results = self.analyzer.analyze_sentiment(empty_data, method='dictionary')
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 0


if __name__ == "__main__":
    pytest.main([__file__])
