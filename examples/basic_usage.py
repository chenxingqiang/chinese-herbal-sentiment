"""
Basic usage example for Chinese Herbal Medicine Sentiment Analysis System.

This example demonstrates how to use the package for basic sentiment analysis
and keyword extraction.
"""

import pandas as pd
from chinese_herbal_sentiment import SentimentAnalyzer, KeywordExtractor, DataAnalyzer, Visualizer


def main():
    """Main function demonstrating basic usage."""
    
    # Sample data (in practice, you would load from a file)
    sample_data = pd.DataFrame({
        '评论内容': [
            '这个中药质量很好，效果不错，包装也很精美',
            '包装很差，质量一般，但是价格便宜',
            '服务态度很好，物流快，值得推荐',
            '价格太贵了，不推荐购买，性价比低',
            '味道不错，值得购买，会再次光顾',
            '中药效果显著，服用后症状明显改善',
            '物流太慢了，等了很久才收到',
            '客服态度很好，解答很详细',
            '价格合理，性价比很高',
            '包装很精美，送礼很合适'
        ]
    })
    
    print("=== Chinese Herbal Medicine Sentiment Analysis ===")
    print(f"Analyzing {len(sample_data)} reviews...")
    print()
    
    # 1. Sentiment Analysis
    print("1. Performing Sentiment Analysis...")
    sentiment_analyzer = SentimentAnalyzer()
    
    # Analyze with all methods
    sentiment_results = sentiment_analyzer.analyze_all_methods(sample_data)
    
    print("Sentiment Analysis Results:")
    print(sentiment_results.head())
    print()
    
    # 2. Keyword Extraction
    print("2. Performing Keyword Extraction...")
    keyword_extractor = KeywordExtractor()
    
    # Extract keywords with all methods
    keyword_results = keyword_extractor.extract_all_methods(sample_data, num_keywords=10)
    
    print("Keyword Extraction Results:")
    print(keyword_results.head())
    print()
    
    # 3. Visualization
    print("3. Generating Visualizations...")
    visualizer = Visualizer()
    
    # Create sentiment distribution plot
    visualizer.plot_sentiment_distribution(sentiment_results, save_path="sentiment_distribution.png")
    print("Sentiment distribution plot saved as 'sentiment_distribution.png'")
    
    # Create keyword cloud
    visualizer.plot_keyword_cloud(keyword_results, save_path="keyword_cloud.png")
    print("Keyword cloud saved as 'keyword_cloud.png'")
    
    print()
    print("=== Analysis Complete ===")
    print("Results have been saved to CSV files and visualizations generated.")


if __name__ == "__main__":
    main()
