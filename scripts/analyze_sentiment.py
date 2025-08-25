#!/usr/bin/env python3
"""
Command-line script for sentiment analysis of Chinese herbal medicine reviews.
"""

import argparse
import sys
import os
from pathlib import Path

# Add the parent directory to the path to import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from chinese_herbal_sentiment.core.sentiment_analysis import SentimentAnalyzer
from chinese_herbal_sentiment.utils.data_analysis import DataAnalyzer


def main():
    """Main function for sentiment analysis CLI."""
    parser = argparse.ArgumentParser(
        description="Analyze sentiment of Chinese herbal medicine reviews"
    )
    parser.add_argument(
        "data_path",
        help="Path to the data file or directory containing review data"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="sentiment_results.csv",
        help="Output file path for results (default: sentiment_results.csv)"
    )
    parser.add_argument(
        "--method",
        "-m",
        choices=["dictionary", "svm", "naive_bayes", "logistic_regression", "all"],
        default="all",
        help="Sentiment analysis method to use (default: all)"
    )
    parser.add_argument(
        "--sample_size",
        "-s",
        type=int,
        help="Number of samples to analyze (default: all)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    try:
        # Initialize analyzer
        analyzer = SentimentAnalyzer()
        
        # Load data
        data_analyzer = DataAnalyzer()
        data = data_analyzer.load_data(args.data_path, sample_size=args.sample_size)
        
        if args.verbose:
            print(f"Loaded {len(data)} reviews for analysis")
        
        # Perform sentiment analysis
        if args.method == "all":
            results = analyzer.analyze_all_methods(data)
        else:
            results = analyzer.analyze_sentiment(data, method=args.method)
        
        # Save results
        results.to_csv(args.output, index=False)
        
        if args.verbose:
            print(f"Results saved to {args.output}")
            print(f"Analysis completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
