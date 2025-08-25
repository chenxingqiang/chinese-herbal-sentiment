import pandas as pd
import os
import jieba
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

def load_excel_files(directory):
    """Load all Excel files from the specified directory"""
    data = {
        'positive': [],
        'neutral': [],
        'negative': []
    }
    
    for filename in os.listdir(directory):
        if filename.endswith(('.xls', '.xlsx')):
            filepath = os.path.join(directory, filename)
            try:
                df = pd.read_excel(filepath)
                if '好评' in filename:
                    data['positive'].append(df)
                elif '中评' in filename:
                    data['neutral'].append(df)
                elif '差评' in filename:
                    data['negative'].append(df)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    
    return {k: pd.concat(v, ignore_index=True) if v else pd.DataFrame() 
            for k, v in data.items()}

def preprocess_text(text):
    """Preprocess Chinese text"""
    if not isinstance(text, str):
        return ""
    words = jieba.cut(text)
    return " ".join(words)

def extract_keywords(text_series, top_n=20):
    """Extract keywords using TF-IDF"""
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(text_series)
    
    # Get feature names and their average TF-IDF scores
    feature_names = vectorizer.get_feature_names_out()
    avg_scores = np.mean(tfidf_matrix.toarray(), axis=0)
    
    # Create a dictionary of words and their scores
    word_scores = dict(zip(feature_names, avg_scores))
    return dict(sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:top_n])

def analyze_sentiment_distribution(data):
    """Analyze sentiment distribution"""
    total_reviews = sum(len(df) for df in data.values())
    distribution = {k: len(v) / total_reviews * 100 for k, v in data.items()}
    
    plt.figure(figsize=(10, 6))
    plt.bar(distribution.keys(), distribution.values())
    plt.title('Sentiment Distribution of Reviews')
    plt.ylabel('Percentage')
    plt.savefig('sentiment_distribution.png')
    plt.close()
    
    return distribution

def main():
    # Load data
    data = load_excel_files('../comments')
    
    # Analyze sentiment distribution
    sentiment_dist = analyze_sentiment_distribution(data)
    print("\nSentiment Distribution:")
    for sentiment, percentage in sentiment_dist.items():
        print(f"{sentiment}: {percentage:.2f}%")
    
    # Extract keywords for each sentiment
    for sentiment, df in data.items():
        if not df.empty:
            # Assuming the review text is in the first column
            processed_texts = df.iloc[:, 0].apply(preprocess_text)
            keywords = extract_keywords(processed_texts)
            
            print(f"\nTop keywords for {sentiment} reviews:")
            for word, score in keywords.items():
                print(f"{word}: {score:.4f}")

if __name__ == "__main__":
    main() 