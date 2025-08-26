#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chinese Herbal Medicine Sentiment Analysis REST API
基于FastAPI的中药材情感分析REST API服务
"""

import os
import sys
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import traceback

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

try:
    from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, FileResponse
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    print("Warning: FastAPI not available. Please install: pip install fastapi uvicorn")
    FASTAPI_AVAILABLE = False

import pandas as pd
import numpy as np
import json
import tempfile
import shutil

# 导入核心模块
try:
    from chinese_herbal_sentiment.core.sentiment_analysis import SentimentAnalysis
    from chinese_herbal_sentiment.core.keyword_extraction import KeywordExtraction
    from chinese_herbal_sentiment.core.prediction_service import PredictionService
    from chinese_herbal_sentiment.core.regression_analysis import SupplyChainRegression
    from chinese_herbal_sentiment.core.time_series_analysis import TimeSeriesAnalyzer
except ImportError as e:
    print(f"Warning: Core modules import failed: {e}")

# Pydantic 模型定义
if FASTAPI_AVAILABLE:
    class TextInput(BaseModel):
        """单个文本输入"""
        text: str = Field(..., description="输入文本", min_length=1, max_length=10000)
        
    class BatchTextInput(BaseModel):
        """批量文本输入"""
        texts: List[str] = Field(..., description="文本列表", min_items=1, max_items=100)
        methods: Optional[List[str]] = Field(default=['dictionary', 'svm'], description="分析方法列表")
        
    class SentimentRequest(BaseModel):
        """情感分析请求"""
        texts: Union[str, List[str]] = Field(..., description="待分析文本")
        methods: Optional[List[str]] = Field(default=['dictionary'], description="分析方法")
        return_probabilities: Optional[bool] = Field(default=False, description="是否返回概率")
        
    class KeywordRequest(BaseModel):
        """关键词提取请求"""
        texts: List[str] = Field(..., description="文本列表")
        methods: Optional[List[str]] = Field(default=['tfidf'], description="提取方法")
        top_k: Optional[int] = Field(default=10, description="返回关键词数量", ge=1, le=50)


# 创建应用（如果FastAPI可用）
if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="Chinese Herbal Medicine Sentiment Analysis API",
        description="中药材电商评论情感分析系统API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 全局变量
    sentiment_analyzer = None
    keyword_extractor = None
    prediction_service = None
    
    @app.on_event("startup")
    async def startup_event():
        """应用启动时初始化服务"""
        global sentiment_analyzer, keyword_extractor, prediction_service
        
        try:
            print("Initializing services...")
            sentiment_analyzer = SentimentAnalysis()
            keyword_extractor = KeywordExtraction()
            prediction_service = PredictionService()
            print("All services initialized successfully!")
        except Exception as e:
            print(f"Failed to initialize services: {e}")
    
    @app.get("/")
    async def root():
        """根端点"""
        return {
            "message": "Chinese Herbal Medicine Sentiment Analysis API",
            "version": "1.0.0",
            "status": "running",
            "timestamp": datetime.now().isoformat(),
            "docs": "/docs"
        }
    
    @app.get("/health")
    async def health_check():
        """健康检查"""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "sentiment_analyzer": sentiment_analyzer is not None,
                "keyword_extractor": keyword_extractor is not None,
                "prediction_service": prediction_service is not None
            }
        }
    
    @app.post("/api/v1/sentiment/analyze")
    async def analyze_sentiment(request: SentimentRequest):
        """情感分析端点"""
        try:
            if prediction_service is None:
                raise HTTPException(status_code=500, detail="Prediction service not initialized")
            
            result = prediction_service.predict_sentiment(
                texts=request.texts,
                methods=request.methods,
                return_probabilities=request.return_probabilities
            )
            
            return {
                "success": True,
                "data": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {str(e)}")
    
    @app.post("/api/v1/keywords/extract")
    async def extract_keywords(request: KeywordRequest):
        """关键词提取端点"""
        try:
            if prediction_service is None:
                raise HTTPException(status_code=500, detail="Prediction service not initialized")
            
            result = prediction_service.extract_keywords_batch(
                texts=request.texts,
                methods=request.methods,
                top_k=request.top_k
            )
            
            return {
                "success": True,
                "data": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Keyword extraction failed: {str(e)}")
    
    @app.get("/api/v1/models/info")
    async def get_model_info():
        """获取模型信息端点"""
        try:
            if prediction_service is None:
                raise HTTPException(status_code=500, detail="Prediction service not initialized")
            
            model_info = prediction_service.get_model_info()
            
            return {
                "success": True,
                "data": model_info,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

else:
    # FastAPI不可用时的占位符
    app = None


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """运行API服务器"""
    if not FASTAPI_AVAILABLE:
        print("Error: FastAPI is not available. Please install: pip install fastapi uvicorn")
        return
    
    print(f"Starting Chinese Herbal Medicine Sentiment Analysis API...")
    print(f"Server will be available at: http://{host}:{port}")
    print(f"API documentation: http://{host}:{port}/docs")
    
    uvicorn.run(
        "chinese_herbal_sentiment.api.app_fixed:app",
        host=host,
        port=port,
        reload=reload,
        access_log=True
    )


if __name__ == "__main__":
    run_server(host="127.0.0.1", port=8000, reload=True)
