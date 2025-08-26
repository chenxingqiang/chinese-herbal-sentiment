"""
API module for Chinese Herbal Medicine Sentiment Analysis
中药材情感分析API模块
"""

try:
    from .app import app, run_server
    __all__ = ['app', 'run_server']
except ImportError:
    print("Warning: FastAPI not available. API module disabled.")
    __all__ = []
