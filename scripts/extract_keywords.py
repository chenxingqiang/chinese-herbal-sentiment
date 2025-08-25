#!/usr/bin/env python3
"""
Command-line script for keyword extraction from Chinese herbal medicine reviews.
"""

import argparse
import sys
import os
from pathlib import Path

# Add the parent directory to the path to import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from chinese_herbal_sentiment.core.keyword_extraction import KeywordExtractor
from chinese_herbal_sentiment.utils.data_analysis import DataAnalyzer


def main():
    """Main function for keyword extraction CLI."""
    parser = argparse.ArgumentParser(
        description="Extract keywords from Chinese herbal medicine reviews"
    )
    parser.add_argument(
        "data_path",
        help="Path to the data file or directory containing review data"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="keywords_results.csv",
        help="Output file path for results (default: keywords_results.csv)"
    )
    parser.add_argument(
        "--method",
        "-m",
        choices=["tfidf", "textrank", "lda", "all"],
        default="all",
        help="Keyword extraction method to use (default: all)"
    )
    parser.add_argument(
        "--num_keywords",
        "-n",
        type=int,
        default=20,
        help="Number of keywords to extract (default: 20)"
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
        # Initialize extractor
        extractor = KeywordExtractor()
        
        # Load data
        data_analyzer = DataAnalyzer()
        data = data_analyzer.load_data(args.data_path, sample_size=args.sample_size)
        
        if args.verbose:
            print(f"Loaded {len(data)} reviews for keyword extraction")
        
        # Extract keywords
        if args.method == "all":
            results = extractor.extract_all_methods(data, num_keywords=args.num_keywords)
        else:
            results = extractor.extract_keywords(data, method=args.method, num_keywords=args.num_keywords)
        
        # Save results
        results.to_csv(args.output, index=False)
        
        if args.verbose:
            print(f"Results saved to {args.output}")
            print(f"Keyword extraction completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
