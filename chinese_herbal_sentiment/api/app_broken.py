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

    class QualityFeatures(BaseModel):
        """质量评分特征"""
        material_quality: Optional[float] = Field(default=7.5, description="原料质量", ge=1.0, le=10.0)
        technology: Optional[float] = Field(default=7.5, description="工艺技术", ge=1.0, le=10.0)
        delivery_speed: Optional[float] = Field(default=7.5, description="交货速度", ge=1.0, le=10.0)
        after_sales_service: Optional[float] = Field(default=7.5, description="售后服务", ge=1.0, le=10.0)
        # 可以添加更多特征字段

    class QualityPredictionRequest(BaseModel):
        """质量预测请求"""
        features: Union[QualityFeatures, Dict[str, float]] = Field(..., description="特征数据")
        confidence_interval: Optional[bool] = Field(default=True, description="是否返回置信区间")

    class TimeSeriesRequest(BaseModel):
        """时间序列分析请求"""
        data: List[Dict[str, Any]] = Field(..., description="时间序列数据")
        time_column: str = Field(..., description="时间列名")
        value_column: str = Field(..., description="数值列名")
        forecast_periods: Optional[int] = Field(default=30, description="预测期数", ge=1, le=365)

    class ComprehensiveAnalysisRequest(BaseModel):
        """综合分析请求"""
        texts: List[str] = Field(..., description="文本数据")
        include_sentiment: Optional[bool] = Field(default=True, description="包含情感分析")
        include_keywords: Optional[bool] = Field(default=True, description="包含关键词提取")
        include_quality: Optional[bool] = Field(default=False, description="包含质量预测")
        quality_features: Optional[QualityFeatures] = Field(default=None, description="质量特征")


# 创建FastAPI应用
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
        allow_origins=["*"],  # 生产环境应该限制具体域名
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 全局变量存储服务实例
    sentiment_analyzer = None
    keyword_extractor = None
    prediction_service = None
    regression_analyzer = None
    timeseries_analyzer = None

    # 初始化服务
    @app.on_event("startup")
    async def startup_event():
        """应用启动时初始化服务"""
        global sentiment_analyzer, keyword_extractor, prediction_service
        global regression_analyzer, timeseries_analyzer

        try:
            print("Initializing services...")

            # 初始化核心分析器
            sentiment_analyzer = SentimentAnalysis()
            keyword_extractor = KeywordExtraction()
            prediction_service = PredictionService()
            regression_analyzer = SupplyChainRegression()
            timeseries_analyzer = TimeSeriesAnalyzer()

            print("All services initialized successfully!")

        except Exception as e:
            print(f"Failed to initialize services: {e}")
            traceback.print_exc()


    # 健康检查端点
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
            "prediction_service": prediction_service is not None,
            "regression_analyzer": regression_analyzer is not None,
            "timeseries_analyzer": timeseries_analyzer is not None
        }
    }


# 情感分析端点
@app.post("/api/v1/sentiment/analyze")
async def analyze_sentiment(request: SentimentRequest):
    """
    情感分析端点

    分析文本的情感倾向
    """
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


@app.post("/api/v1/sentiment/batch")
async def batch_sentiment_analysis(file: UploadFile = File(...)):
    """
    批量情感分析端点

    上传CSV/Excel文件进行批量情感分析
    """
    try:
        if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload CSV or Excel file.")

        # 保存上传文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name

        try:
            # 读取文件
            if file.filename.endswith('.csv'):
                df = pd.read_csv(tmp_path)
            else:
                df = pd.read_excel(tmp_path)

            # 检查必需的列
            if '评论内容' not in df.columns and 'text' not in df.columns:
                raise HTTPException(status_code=400, detail="File must contain '评论内容' or 'text' column")

            text_column = '评论内容' if '评论内容' in df.columns else 'text'
            texts = df[text_column].dropna().tolist()

            if len(texts) == 0:
                raise HTTPException(status_code=400, detail="No valid text data found")

            if len(texts) > 1000:
                texts = texts[:1000]  # 限制批量处理数量

            # 进行情感分析
            result = prediction_service.predict_sentiment(
                texts=texts,
                methods=['dictionary', 'svm']
            )

            return {
                "success": True,
                "data": result,
                "processed_count": len(texts),
                "timestamp": datetime.now().isoformat()
            }

        finally:
            # 清理临时文件
            os.unlink(tmp_path)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")


# 关键词提取端点
@app.post("/api/v1/keywords/extract")
async def extract_keywords(request: KeywordRequest):
    """
    关键词提取端点

    从文本中提取关键词
    """
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


# 质量预测端点
@app.post("/api/v1/quality/predict")
async def predict_quality(request: QualityPredictionRequest):
    """
    质量评分预测端点

    基于供应链特征预测质量评分
    """
    try:
        if prediction_service is None:
            raise HTTPException(status_code=500, detail="Prediction service not initialized")

        # 处理特征数据
        if isinstance(request.features, QualityFeatures):
            features_dict = request.features.dict()
        else:
            features_dict = request.features

        result = prediction_service.predict_quality_score(
            features=features_dict,
            confidence_interval=request.confidence_interval
        )

        return {
            "success": True,
            "data": result,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quality prediction failed: {str(e)}")


# 时间序列分析端点
@app.post("/api/v1/timeseries/analyze")
async def analyze_timeseries(request: TimeSeriesRequest):
    """
    时间序列分析端点

    分析时间序列数据的趋势、季节性和预测
    """
    try:
        if timeseries_analyzer is None:
            raise HTTPException(status_code=500, detail="Time series analyzer not initialized")

        # 创建DataFrame
        df = pd.DataFrame(request.data)

        # 加载数据
        success = timeseries_analyzer.load_data(
            data=df,
            time_column=request.time_column,
            value_column=request.value_column
        )

        if not success:
            raise HTTPException(status_code=400, detail="Failed to load time series data")

        # 进行分析
        results = {}

        # 趋势分析
        try:
            trend_result = timeseries_analyzer.trend_analysis(method='linear')
            results['trend_analysis'] = trend_result
        except Exception as e:
            results['trend_analysis'] = {'error': str(e)}

        # 季节性分析
        try:
            seasonal_result = timeseries_analyzer.seasonal_analysis()
            results['seasonal_analysis'] = seasonal_result
        except Exception as e:
            results['seasonal_analysis'] = {'error': str(e)}

        # 预测
        try:
            forecast_result = timeseries_analyzer.forecast(
                periods=request.forecast_periods,
                method='auto'
            )
            results['forecast'] = forecast_result
        except Exception as e:
            results['forecast'] = {'error': str(e)}

        # 异常值检测
        try:
            anomaly_result = timeseries_analyzer.detect_anomalies()
            results['anomaly_detection'] = anomaly_result
        except Exception as e:
            results['anomaly_detection'] = {'error': str(e)}

        return {
            "success": True,
            "data": results,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Time series analysis failed: {str(e)}")


# 综合分析端点
@app.post("/api/v1/analyze/comprehensive")
async def comprehensive_analysis(request: ComprehensiveAnalysisRequest):
    """
    综合分析端点

    同时进行情感分析、关键词提取等多种分析
    """
    try:
        if prediction_service is None:
            raise HTTPException(status_code=500, detail="Prediction service not initialized")

        # 处理质量特征
        quality_features = None
        if request.include_quality and request.quality_features:
            if isinstance(request.quality_features, QualityFeatures):
                quality_features = request.quality_features.dict()
            else:
                quality_features = request.quality_features

        result = prediction_service.analyze_comprehensive(
            texts=request.texts,
            include_sentiment=request.include_sentiment,
            include_keywords=request.include_keywords,
            include_quality=request.include_quality,
            quality_features=quality_features
        )

        return {
            "success": True,
            "data": result,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comprehensive analysis failed: {str(e)}")


# 模型信息端点
@app.get("/api/v1/models/info")
async def get_model_info():
    """
    获取模型信息端点

    返回当前加载的模型信息
    """
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


# 预测历史端点
@app.get("/api/v1/predictions/history")
async def get_prediction_history(limit: int = 10):
    """
    获取预测历史端点

    返回最近的预测历史记录
    """
    try:
        if prediction_service is None:
            raise HTTPException(status_code=500, detail="Prediction service not initialized")

        history = prediction_service.get_prediction_history(limit=limit)

        return {
            "success": True,
            "data": history,
            "count": len(history),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get prediction history: {str(e)}")


# 清除历史端点
@app.delete("/api/v1/predictions/history")
async def clear_prediction_history():
    """
    清除预测历史端点

    清空所有预测历史记录
    """
    try:
        if prediction_service is None:
            raise HTTPException(status_code=500, detail="Prediction service not initialized")

        prediction_service.clear_prediction_history()

        return {
            "success": True,
            "message": "Prediction history cleared",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear prediction history: {str(e)}")


# 回归分析端点
@app.post("/api/v1/regression/train")
async def train_regression_model(background_tasks: BackgroundTasks):
    """
    训练回归模型端点

    使用模拟数据训练供应链质量回归模型
    """
    try:
        if regression_analyzer is None:
            raise HTTPException(status_code=500, detail="Regression analyzer not initialized")

        # 后台任务：训练模型
        def train_model_task():
            try:
                # 生成训练数据
                data = regression_analyzer.generate_supply_chain_data(1000)

                # 准备特征
                feature_columns = [col for col in data.columns
                                 if col not in ['service_quality', 'enterprise_size', 'product_type', 'region']]
                categorical_columns = ['enterprise_size', 'product_type', 'region']

                X, y = regression_analyzer.prepare_data(
                    data=data,
                    target_column='service_quality',
                    feature_columns=feature_columns + categorical_columns,
                    categorical_columns=categorical_columns
                )

                # 训练模型
                results = regression_analyzer.train(X, y, test_size=0.2)

                # 保存模型
                os.makedirs('models', exist_ok=True)
                regression_analyzer.save_model('models/api_regression_model.pkl')

                print("Regression model training completed successfully")
                return results

            except Exception as e:
                print(f"Regression model training failed: {e}")
                return None

        # 添加后台任务
        background_tasks.add_task(train_model_task)

        return {
            "success": True,
            "message": "Regression model training started in background",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start regression training: {str(e)}")


# 错误处理
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "success": False,
            "error": "Endpoint not found",
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )


# 运行服务器的辅助函数
def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    运行API服务器

    Args:
        host: 主机地址
        port: 端口号
        reload: 是否自动重载
    """
    if not FASTAPI_AVAILABLE:
        print("Error: FastAPI is not available. Please install: pip install fastapi uvicorn")
        return

    print(f"Starting Chinese Herbal Medicine Sentiment Analysis API...")
    print(f"Server will be available at: http://{host}:{port}")
    print(f"API documentation: http://{host}:{port}/docs")
    print(f"ReDoc documentation: http://{host}:{port}/redoc")

    uvicorn.run(
        "chinese_herbal_sentiment.api.app:app",
        host=host,
        port=port,
        reload=reload,
        access_log=True
    )


if __name__ == "__main__":
    # 直接运行时启动服务器
    run_server(host="127.0.0.1", port=8000, reload=True)
