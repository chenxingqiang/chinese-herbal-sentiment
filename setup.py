"""
Setup configuration for Chinese Herbal Medicine Sentiment Analysis System.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements(filename):
    with open(filename, "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="chinese-herbal-sentiment",
    version="0.1.0",
    description="Chinese Herbal Medicine E-commerce Sentiment Analysis System",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Xingqiang Chen",
    author_email="chenxingqiang@example.com",
    url="https://github.com/chenxingqiang/chinese-herbal-sentiment",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "jieba>=0.42.1",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "wordcloud>=1.8.0",
        "networkx>=2.6.0",
        "openpyxl>=3.0.0",
        "tqdm>=4.62.0",
        "gensim>=4.0.0",
        "xlrd>=2.0.0",
    ],
    extras_require={
        "deep_learning": [
            "torch>=1.9.0",
            "tensorflow>=2.6.0",
            "transformers>=4.11.0",
            "keras>=2.6.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "all": [
            "torch>=1.9.0",
            "tensorflow>=2.6.0",
            "transformers>=4.11.0",
            "keras>=2.6.0",
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "chinese-herbal-analyze=scripts.analyze_sentiment:main",
            "chinese-herbal-keywords=scripts.extract_keywords:main",
            "chinese-herbal-full=scripts.full_analysis:main",
        ],
    },
    include_package_data=True,
    package_data={
        "chinese_herbal_sentiment": [
            "models/*",
            "models/bert_sentiment_model/*",
        ],
    },
    keywords=[
        "sentiment-analysis",
        "chinese-herbal-medicine",
        "e-commerce",
        "nlp",
        "machine-learning",
        "deep-learning",
        "bert",
        "textcnn",
        "textrank",
        "supply-chain",
        "quality-evaluation",
    ],
    project_urls={
        "Bug Reports": "https://github.com/chenxingqiang/chinese-herbal-sentiment/issues",
        "Source": "https://github.com/chenxingqiang/chinese-herbal-sentiment",
        "Documentation": "https://github.com/chenxingqiang/chinese-herbal-sentiment#readme",
    },
)
