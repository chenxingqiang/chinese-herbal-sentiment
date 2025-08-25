"""
Tests for keyword extraction functionality.
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

# Add the parent directory to the path to import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from chinese_herbal_sentiment.core.keyword_extraction import KeywordExtractor


class TestKeywordExtractor:
    """Test cases for KeywordExtractor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = KeywordExtractor()
        self.sample_data = pd.DataFrame({
            '评论内容': [
                '这个中药质量很好，效果不错，包装也很精美',
                '包装很差，质量一般，但是价格便宜',
                '服务态度很好，物流快，值得推荐',
                '价格太贵了，不推荐购买，性价比低',
                '味道不错，值得购买，会再次光顾'
            ]
        })
    
    def test_extractor_initialization(self):
        """Test that the extractor can be initialized."""
        assert self.extractor is not None
        assert hasattr(self.extractor, 'extract_keywords')
    
    def test_tfidf_keyword_extraction(self):
        """Test TF-IDF keyword extraction."""
        results = self.extractor.extract_keywords(self.sample_data, method='tfidf', num_keywords=5)
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0
        assert 'keyword' in results.columns
        assert 'score' in results.columns
    
    def test_textrank_keyword_extraction(self):
        """Test TextRank keyword extraction."""
        results = self.extractor.extract_keywords(self.sample_data, method='textrank', num_keywords=5)
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0
        assert 'keyword' in results.columns
        assert 'score' in results.columns
    
    def test_lda_keyword_extraction(self):
        """Test LDA keyword extraction."""
        results = self.extractor.extract_keywords(self.sample_data, method='lda', num_keywords=5)
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0
        assert 'keyword' in results.columns
        assert 'score' in results.columns
    
    def test_all_methods_extraction(self):
        """Test extraction with all methods."""
        results = self.extractor.extract_all_methods(self.sample_data, num_keywords=5)
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0
        # Should have columns for each method
        expected_columns = ['tfidf_keywords', 'textrank_keywords', 'lda_keywords']
        for col in expected_columns:
            assert col in results.columns
    
    def test_invalid_method(self):
        """Test that invalid method raises appropriate error."""
        with pytest.raises(ValueError):
            self.extractor.extract_keywords(self.sample_data, method='invalid_method')
    
    def test_empty_data(self):
        """Test handling of empty data."""
        empty_data = pd.DataFrame({'评论内容': []})
        results = self.extractor.extract_keywords(empty_data, method='tfidf')
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 0
    
    def test_num_keywords_parameter(self):
        """Test that num_keywords parameter works correctly."""
        results = self.extractor.extract_keywords(self.sample_data, method='tfidf', num_keywords=3)
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) <= 3  # Should not exceed requested number


if __name__ == "__main__":
    pytest.main([__file__])
