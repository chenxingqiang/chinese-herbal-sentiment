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
    version="1.0.0",
    description="Comprehensive Chinese Herbal Medicine Sentiment Analysis and Supply Chain Quality Evaluation System",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Luo Jiawen, Chen Xingqiang",
    author_email="chenxingqiang@turingai.cc",
    url="https://github.com/chenxingqiang/chinese-herbal-sentiment",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Text Processing :: General",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Internet :: WWW/HTTP :: Dynamic Content",
        "Operating System :: OS Independent",
        "Natural Language :: Chinese (Simplified)",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "deep_learning": [
            "torch>=1.10.0",
            "tensorflow>=2.8.0", 
            "transformers>=4.20.0",
        ],
        "api": [
            "fastapi>=0.68.0",
            "uvicorn>=0.15.0",
            "python-multipart>=0.0.5",
        ],
        "analysis": [
            "scipy>=1.7.0",
            "statsmodels>=0.13.0",
            "prophet>=1.1.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "pytest-asyncio>=0.18.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "pre-commit>=2.15.0",
        ],
        "all": [
            # Deep learning
            "torch>=1.10.0",
            "tensorflow>=2.8.0",
            "transformers>=4.20.0",
            # API services
            "fastapi>=0.68.0",
            "uvicorn>=0.15.0",
            "python-multipart>=0.0.5",
            # Advanced analysis
            "scipy>=1.7.0",
            "statsmodels>=0.13.0",
            "prophet>=1.1.0",
            # Development tools
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "pytest-asyncio>=0.18.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "pre-commit>=2.15.0",
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
        "traditional-chinese-medicine",
        "e-commerce",
        "nlp",
        "machine-learning",
        "deep-learning",
        "bert",
        "textcnn",
        "textrank",
        "supply-chain",
        "quality-evaluation",
        "regression-analysis",
        "time-series-analysis",
        "api-service",
        "fastapi",
        "healthcare",
        "chinese-nlp",
    ],
    project_urls={
        "Homepage": "https://github.com/chenxingqiang/chinese-herbal-sentiment",
        "Bug Reports": "https://github.com/chenxingqiang/chinese-herbal-sentiment/issues",
        "Source": "https://github.com/chenxingqiang/chinese-herbal-sentiment",
        "Documentation": "https://github.com/chenxingqiang/chinese-herbal-sentiment#readme",
        "Dataset": "https://huggingface.co/datasets/xingqiang/chinese-herbal-medicine-sentiment",
        "PyPI": "https://pypi.org/project/chinese-herbal-sentiment/",
        "Discussions": "https://github.com/chenxingqiang/chinese-herbal-sentiment/discussions",
    },
)
