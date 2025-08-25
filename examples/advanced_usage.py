"""
Advanced usage example for Chinese Herbal Medicine Sentiment Analysis System.

This example demonstrates advanced features including deep learning models,
custom analysis pipelines, and detailed evaluation metrics.
"""

import pandas as pd
import numpy as np
from chinese_herbal_sentiment import (
    SentimentAnalyzer, 
    KeywordExtractor, 
    DataAnalyzer, 
    Visualizer,
    BERTSentimentAnalyzer,
    TextCNNSentimentAnalyzer,
    TextRankSentimentAnalyzer
)


def main():
    """Main function demonstrating advanced usage."""
    
    # Load sample data (in practice, you would load from a file)
    sample_data = pd.DataFrame({
        '评论内容': [
            '这个中药质量很好，效果不错，包装也很精美，值得推荐',
            '包装很差，质量一般，但是价格便宜，勉强可以接受',
            '服务态度很好，物流快，值得推荐，会再次购买',
            '价格太贵了，不推荐购买，性价比低，很失望',
            '味道不错，值得购买，会再次光顾，推荐给朋友',
            '中药效果显著，服用后症状明显改善，医生也推荐',
            '物流太慢了，等了很久才收到，影响使用',
            '客服态度很好，解答很详细，服务很周到',
            '价格合理，性价比很高，物超所值',
            '包装很精美，送礼很合适，质量有保证'
        ],
        '评分': [5, 3, 5, 1, 4, 5, 2, 5, 4, 4]  # Simulated ratings
    })
    
    print("=== Advanced Chinese Herbal Medicine Analysis ===")
    print(f"Analyzing {len(sample_data)} reviews with advanced features...")
    print()
    
    # 1. Advanced Sentiment Analysis with Multiple Models
    print("1. Advanced Sentiment Analysis...")
    
    # Basic sentiment analysis
    sentiment_analyzer = SentimentAnalyzer()
    basic_results = sentiment_analyzer.analyze_all_methods(sample_data)
    
    # Deep learning models (if available)
    try:
        print("   - Running BERT analysis...")
        bert_analyzer = BERTSentimentAnalyzer()
        bert_results = bert_analyzer.analyze_sentiment(sample_data)
        basic_results['bert_sentiment'] = bert_results['sentiment']
    except Exception as e:
        print(f"   - BERT analysis not available: {e}")
    
    try:
        print("   - Running TextCNN analysis...")
        textcnn_analyzer = TextCNNSentimentAnalyzer()
        textcnn_results = textcnn_analyzer.analyze_sentiment(sample_data)
        basic_results['textcnn_sentiment'] = textcnn_results['sentiment']
    except Exception as e:
        print(f"   - TextCNN analysis not available: {e}")
    
    try:
        print("   - Running TextRank analysis...")
        textrank_analyzer = TextRankSentimentAnalyzer()
        textrank_results = textrank_analyzer.analyze_sentiment(sample_data)
        basic_results['textrank_sentiment'] = textrank_results['sentiment']
    except Exception as e:
        print(f"   - TextRank analysis not available: {e}")
    
    print("Sentiment Analysis Results:")
    print(basic_results.head())
    print()
    
    # 2. Advanced Keyword Extraction with Custom Parameters
    print("2. Advanced Keyword Extraction...")
    keyword_extractor = KeywordExtractor()
    
    # Extract keywords with different parameters
    tfidf_keywords = keyword_extractor.extract_keywords(
        sample_data, method='tfidf', num_keywords=15, min_df=1, max_df=0.8
    )
    
    textrank_keywords = keyword_extractor.extract_keywords(
        sample_data, method='textrank', num_keywords=15, window_size=4
    )
    
    lda_keywords = keyword_extractor.extract_keywords(
        sample_data, method='lda', num_keywords=15, num_topics=3
    )
    
    print("TF-IDF Keywords:")
    print(tfidf_keywords.head())
    print()
    
    print("TextRank Keywords:")
    print(textrank_keywords.head())
    print()
    
    print("LDA Keywords:")
    print(lda_keywords.head())
    print()
    
    # 3. Performance Evaluation
    print("3. Performance Evaluation...")
    
    # Calculate agreement between different methods
    sentiment_columns = [col for col in basic_results.columns if 'sentiment' in col]
    
    print("Method Agreement Analysis:")
    for i, col1 in enumerate(sentiment_columns):
        for col2 in sentiment_columns[i+1:]:
            agreement = (basic_results[col1] == basic_results[col2]).mean()
            print(f"   {col1} vs {col2}: {agreement:.2%} agreement")
    
    print()
    
    # 4. Advanced Visualizations
    print("4. Generating Advanced Visualizations...")
    visualizer = Visualizer()
    
    # Create comprehensive sentiment comparison
    visualizer.plot_sentiment_comparison(basic_results, save_path="sentiment_comparison.png")
    print("   - Sentiment comparison plot saved")
    
    # Create keyword comparison
    all_keywords = pd.concat([
        tfidf_keywords.assign(method='TF-IDF'),
        textrank_keywords.assign(method='TextRank'),
        lda_keywords.assign(method='LDA')
    ])
    visualizer.plot_keyword_comparison(all_keywords, save_path="keyword_comparison.png")
    print("   - Keyword comparison plot saved")
    
    # 5. Supply Chain Quality Evaluation
    print("5. Supply Chain Quality Evaluation...")
    
    # Map keywords to supply chain dimensions
    supply_chain_keywords = {
        '上游': ['质量', '原料', '药材', '产地'],
        '中游': ['包装', '加工', '工艺', '生产'],
        '下游': ['物流', '服务', '客服', '配送', '价格']
    }
    
    print("Supply Chain Dimension Analysis:")
    for dimension, keywords in supply_chain_keywords.items():
        count = sum(1 for keyword in keywords 
                   if any(keyword in str(kw) for kw in tfidf_keywords['keyword']))
        print(f"   {dimension}: {count} relevant keywords found")
    
    print()
    print("=== Advanced Analysis Complete ===")
    print("All results and visualizations have been generated.")


if __name__ == "__main__":
    main()
