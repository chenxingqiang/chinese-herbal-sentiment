#!/usr/bin/env python3
"""
Command-line script for full analysis of Chinese herbal medicine reviews.
"""

import argparse
import sys
import os
from pathlib import Path

# Add the parent directory to the path to import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from chinese_herbal_sentiment.core.sentiment_analysis import SentimentAnalyzer
from chinese_herbal_sentiment.core.keyword_extraction import KeywordExtractor
from chinese_herbal_sentiment.utils.data_analysis import DataAnalyzer
from chinese_herbal_sentiment.utils.visualization import Visualizer


def main():
    """Main function for full analysis CLI."""
    parser = argparse.ArgumentParser(
        description="Perform full analysis of Chinese herbal medicine reviews"
    )
    parser.add_argument(
        "data_path",
        help="Path to the data file or directory containing review data"
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        default="analysis_results",
        help="Output directory for results (default: analysis_results)"
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["sentiment", "keyword", "all"],
        default="all",
        help="Analysis mode (default: all)"
    )
    parser.add_argument(
        "--sample_size",
        "-s",
        type=int,
        help="Number of samples to analyze (default: all)"
    )
    parser.add_argument(
        "--use_deep_learning",
        action="store_true",
        help="Use deep learning models for sentiment analysis"
    )
    parser.add_argument(
        "--use_bert",
        action="store_true",
        help="Use BERT model for sentiment analysis"
    )
    parser.add_argument(
        "--use_textcnn",
        action="store_true",
        help="Use TextCNN model for sentiment analysis"
    )
    parser.add_argument(
        "--use_textrank_sa",
        action="store_true",
        help="Use TextRank for sentiment analysis"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Load data
        data_analyzer = DataAnalyzer()
        data = data_analyzer.load_data(args.data_path, sample_size=args.sample_size)
        
        if args.verbose:
            print(f"Loaded {len(data)} reviews for analysis")
        
        results = {}
        
        # Perform sentiment analysis
        if args.mode in ["sentiment", "all"]:
            if args.verbose:
                print("Performing sentiment analysis...")
            
            sentiment_analyzer = SentimentAnalyzer()
            
            # Basic sentiment analysis
            sentiment_results = sentiment_analyzer.analyze_all_methods(data)
            results["sentiment"] = sentiment_results
            
            # Save sentiment results
            sentiment_results.to_csv(output_dir / "sentiment_analysis_results.csv", index=False)
            
            # Deep learning models if requested
            if args.use_deep_learning or args.use_bert or args.use_textcnn or args.use_textrank_sa:
                if args.verbose:
                    print("Running deep learning models...")
                
                # This would integrate with the deep learning modules
                # For now, we'll just note that this is where they would be called
                pass
        
        # Perform keyword extraction
        if args.mode in ["keyword", "all"]:
            if args.verbose:
                print("Performing keyword extraction...")
            
            keyword_extractor = KeywordExtractor()
            keyword_results = keyword_extractor.extract_all_methods(data)
            results["keywords"] = keyword_results
            
            # Save keyword results
            keyword_results.to_csv(output_dir / "keyword_extraction_results.csv", index=False)
        
        # Generate visualizations
        if args.verbose:
            print("Generating visualizations...")
        
        visualizer = Visualizer()
        
        if "sentiment" in results:
            visualizer.plot_sentiment_distribution(results["sentiment"], 
                                                 save_path=output_dir / "sentiment_distribution.png")
        
        if "keywords" in results:
            visualizer.plot_keyword_cloud(results["keywords"], 
                                        save_path=output_dir / "keyword_cloud.png")
        
        if args.verbose:
            print(f"All results saved to {output_dir}")
            print("Full analysis completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
