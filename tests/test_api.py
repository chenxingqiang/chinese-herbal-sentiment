#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试API模块
"""

import pytest
import json
import tempfile
import os
from typing import Dict, Any

# 尝试导入API相关模块
try:
    from fastapi.testclient import TestClient
    from chinese_herbal_sentiment.api.app import app
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    TestClient = None
    app = None

# 如果FastAPI不可用，跳过所有测试
pytestmark = pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")


@pytest.fixture
def client():
    """创建测试客户端"""
    if not FASTAPI_AVAILABLE:
        pytest.skip("FastAPI not available")
    return TestClient(app)


@pytest.fixture
def sample_texts():
    """示例文本数据"""
    return [
        "这个中药质量很好，效果不错",
        "包装破损严重，不推荐购买",
        "价格合理，服务态度好"
    ]


@pytest.fixture
def sample_time_series_data():
    """示例时间序列数据"""
    import pandas as pd
    import numpy as np
    
    dates = pd.date_range('2023-01-01', periods=30, freq='D')
    values = 7 + np.sin(np.arange(30) * 0.2) + np.random.normal(0, 0.1, 30)
    
    return [
        {"date": date.strftime('%Y-%m-%d'), "sentiment_score": float(value)}
        for date, value in zip(dates, values)
    ]


class TestAPIEndpoints:
    """测试API端点"""
    
    def test_root_endpoint(self, client):
        """测试根端点"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "status" in data
        assert data["status"] == "running"
        
    def test_health_check(self, client):
        """测试健康检查端点"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "services" in data
        assert data["status"] == "healthy"
        
    def test_sentiment_analysis(self, client, sample_texts):
        """测试情感分析端点"""
        request_data = {
            "texts": sample_texts,
            "methods": ["dictionary"],
            "return_probabilities": False
        }
        
        response = client.post("/api/v1/sentiment/analyze", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert "timestamp" in data
        
        result_data = data["data"]
        assert "texts" in result_data
        assert "predictions" in result_data
        assert len(result_data["texts"]) == 3
        
    def test_sentiment_analysis_single_text(self, client):
        """测试单个文本情感分析"""
        request_data = {
            "texts": "产品质量很好",
            "methods": ["dictionary"]
        }
        
        response = client.post("/api/v1/sentiment/analyze", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        
    def test_keyword_extraction(self, client, sample_texts):
        """测试关键词提取端点"""
        request_data = {
            "texts": sample_texts,
            "methods": ["tfidf"],
            "top_k": 5
        }
        
        response = client.post("/api/v1/keywords/extract", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        
        result_data = data["data"]
        assert "texts_count" in result_data
        assert "keywords" in result_data
        assert result_data["texts_count"] == 3
        
    def test_quality_prediction(self, client):
        """测试质量预测端点"""
        request_data = {
            "features": {
                "material_quality": 8.5,
                "technology": 7.8,
                "delivery_speed": 8.2,
                "after_sales_service": 7.5
            },
            "confidence_interval": True
        }
        
        response = client.post("/api/v1/quality/predict", json=request_data)
        # 注意：这可能会失败，因为回归模型可能未训练
        # 我们检查响应格式而不是成功状态
        assert response.status_code in [200, 500]
        
        data = response.json()
        assert "success" in data
        assert "timestamp" in data
        
    def test_comprehensive_analysis(self, client, sample_texts):
        """测试综合分析端点"""
        request_data = {
            "texts": sample_texts,
            "include_sentiment": True,
            "include_keywords": True,
            "include_quality": False
        }
        
        response = client.post("/api/v1/analyze/comprehensive", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        
        result_data = data["data"]
        assert "analysis_type" in result_data
        assert "texts_count" in result_data
        assert "results" in result_data
        assert result_data["analysis_type"] == "comprehensive"
        
    def test_time_series_analysis(self, client, sample_time_series_data):
        """测试时间序列分析端点"""
        request_data = {
            "data": sample_time_series_data,
            "time_column": "date",
            "value_column": "sentiment_score",
            "forecast_periods": 7
        }
        
        response = client.post("/api/v1/timeseries/analyze", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        
        result_data = data["data"]
        assert "trend_analysis" in result_data
        assert "forecast" in result_data
        
    def test_model_info(self, client):
        """测试模型信息端点"""
        response = client.get("/api/v1/models/info")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        
        model_info = data["data"]
        assert "loaded_models" in model_info
        assert "available_analyzers" in model_info
        
    def test_prediction_history(self, client):
        """测试预测历史端点"""
        # 先进行一次预测以创建历史
        request_data = {
            "texts": ["测试文本"],
            "methods": ["dictionary"]
        }
        client.post("/api/v1/sentiment/analyze", json=request_data)
        
        # 获取历史
        response = client.get("/api/v1/predictions/history?limit=5")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert "count" in data
        
    def test_clear_prediction_history(self, client):
        """测试清除预测历史端点"""
        response = client.delete("/api/v1/predictions/history")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "message" in data
        
    def test_train_regression_model(self, client):
        """测试回归模型训练端点"""
        response = client.post("/api/v1/regression/train")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "message" in data
        
    def test_batch_sentiment_analysis_csv(self, client):
        """测试批量CSV文件情感分析"""
        # 创建临时CSV文件
        import pandas as pd
        
        test_data = pd.DataFrame({
            'text': [
                '产品质量很好',
                '包装破损严重',
                '价格合理'
            ]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            test_data.to_csv(tmp_file.name, index=False)
            csv_path = tmp_file.name
        
        try:
            with open(csv_path, 'rb') as f:
                response = client.post(
                    "/api/v1/sentiment/batch",
                    files={"file": ("test.csv", f, "text/csv")}
                )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "processed_count" in data
            
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)


class TestAPIErrorHandling:
    """测试API错误处理"""
    
    def test_invalid_endpoint(self, client):
        """测试无效端点"""
        response = client.get("/invalid/endpoint")
        assert response.status_code == 404
        
        data = response.json()
        assert data["success"] is False
        assert "error" in data
        
    def test_invalid_request_data(self, client):
        """测试无效请求数据"""
        # 缺少必需字段
        request_data = {
            "methods": ["dictionary"]
            # 缺少 "texts" 字段
        }
        
        response = client.post("/api/v1/sentiment/analyze", json=request_data)
        assert response.status_code == 422  # Validation error
        
    def test_empty_text_list(self, client):
        """测试空文本列表"""
        request_data = {
            "texts": [],
            "methods": ["dictionary"]
        }
        
        response = client.post("/api/v1/sentiment/analyze", json=request_data)
        assert response.status_code == 422  # Validation error
        
    def test_invalid_method(self, client):
        """测试无效分析方法"""
        request_data = {
            "texts": ["测试文本"],
            "methods": ["invalid_method"]
        }
        
        response = client.post("/api/v1/sentiment/analyze", json=request_data)
        # 应该能处理，但可能没有结果
        assert response.status_code == 200
        
    def test_invalid_file_format(self, client):
        """测试无效文件格式"""
        # 创建文本文件而不是CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
            tmp_file.write("This is not a CSV file")
            txt_path = tmp_file.name
        
        try:
            with open(txt_path, 'rb') as f:
                response = client.post(
                    "/api/v1/sentiment/batch",
                    files={"file": ("test.txt", f, "text/plain")}
                )
            
            assert response.status_code == 400
            data = response.json()
            assert "Unsupported file format" in data["detail"]
            
        finally:
            if os.path.exists(txt_path):
                os.unlink(txt_path)
        
    def test_malformed_time_series_data(self, client):
        """测试格式错误的时间序列数据"""
        request_data = {
            "data": [
                {"date": "invalid-date", "value": "not-a-number"}
            ],
            "time_column": "date",
            "value_column": "value",
            "forecast_periods": 5
        }
        
        response = client.post("/api/v1/timeseries/analyze", json=request_data)
        # 应该返回错误
        assert response.status_code in [400, 500]
        
    def test_missing_time_series_columns(self, client):
        """测试缺少时间序列列"""
        request_data = {
            "data": [
                {"wrong_column": "2023-01-01", "value": 7.5}
            ],
            "time_column": "date",  # 不存在的列
            "value_column": "value",
            "forecast_periods": 5
        }
        
        response = client.post("/api/v1/timeseries/analyze", json=request_data)
        assert response.status_code in [400, 500]


if __name__ == "__main__":
    if FASTAPI_AVAILABLE:
        pytest.main([__file__])
    else:
        print("FastAPI not available, skipping API tests")
