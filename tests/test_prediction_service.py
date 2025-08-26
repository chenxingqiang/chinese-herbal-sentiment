#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试预测服务模块
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from chinese_herbal_sentiment.core.prediction_service import PredictionService, ModelManager


class TestModelManager:
    """测试模型管理器"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        self.manager = ModelManager()
        
    def test_initialization(self):
        """测试初始化"""
        assert self.manager.models_dir == "models"
        assert isinstance(self.manager.loaded_models, dict)
        assert isinstance(self.manager.model_metadata, dict)
        assert len(self.manager.loaded_models) == 0
        
    def test_detect_model_type(self):
        """测试模型类型检测"""
        assert self.manager._detect_model_type("model.pkl") == "sklearn"
        assert self.manager._detect_model_type("model.pt") == "torch"
        assert self.manager._detect_model_type("model.pth") == "torch"
        assert self.manager._detect_model_type("regression_model.pkl") == "regression"
        
    def test_list_models(self):
        """测试模型列表"""
        models = self.manager.list_models()
        assert isinstance(models, dict)
        assert len(models) == 0
        
    def test_get_nonexistent_model(self):
        """测试获取不存在的模型"""
        model = self.manager.get_model("nonexistent")
        assert model is None


class TestPredictionService:
    """测试预测服务"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        self.service = PredictionService()
        
    def test_initialization(self):
        """测试初始化"""
        assert self.service.model_manager is not None
        assert self.service.sentiment_analyzer is not None
        assert self.service.keyword_extractor is not None
        assert isinstance(self.service.prediction_history, list)
        
    def test_predict_sentiment_basic(self):
        """测试基础情感分析"""
        texts = [
            "这个产品质量很好，值得推荐",
            "包装破损，质量很差",
            "一般般，没有特别的感觉"
        ]
        
        # 使用词典方法
        result = self.service.predict_sentiment(texts, methods=['dictionary'])
        
        assert 'texts' in result
        assert 'predictions' in result
        assert 'timestamp' in result
        assert 'methods_used' in result
        
        assert len(result['texts']) == 3
        assert 'dictionary' in result['predictions']
        assert len(result['predictions']['dictionary']['labels']) == 3
        assert len(result['predictions']['dictionary']['scores']) == 3
        
    def test_predict_sentiment_single_text(self):
        """测试单个文本情感分析"""
        text = "产品质量非常好"
        
        result = self.service.predict_sentiment(text, methods=['dictionary'])
        
        assert len(result['texts']) == 1
        assert result['texts'][0] == text
        
    def test_extract_keywords_batch(self):
        """测试批量关键词提取"""
        texts = [
            "中药材质量很好，包装精美，物流快速",
            "价格合理，效果明显，服务态度好",
            "产品新鲜，配送及时，包装完整"
        ]
        
        result = self.service.extract_keywords_batch(
            texts, 
            methods=['tfidf'], 
            top_k=5
        )
        
        assert 'texts_count' in result
        assert 'methods_used' in result
        assert 'keywords' in result
        assert 'timestamp' in result
        
        assert result['texts_count'] == 3
        assert 'tfidf' in result['keywords']
        
    def test_analyze_comprehensive(self):
        """测试综合分析"""
        texts = [
            "中药材质量优秀，配送及时",
            "产品效果好，价格合理"
        ]
        
        result = self.service.analyze_comprehensive(
            texts=texts,
            include_sentiment=True,
            include_keywords=True,
            include_quality=False
        )
        
        assert 'analysis_type' in result
        assert 'texts_count' in result
        assert 'results' in result
        
        assert result['analysis_type'] == 'comprehensive'
        assert result['texts_count'] == 2
        
        results = result['results']
        assert 'sentiment_analysis' in results
        assert 'keyword_extraction' in results
        
    def test_prediction_history(self):
        """测试预测历史"""
        # 初始历史应该为空
        history = self.service.get_prediction_history()
        initial_count = len(history)
        
        # 进行一次预测
        texts = ["测试文本"]
        self.service.predict_sentiment(texts, methods=['dictionary'])
        
        # 检查历史增加
        new_history = self.service.get_prediction_history()
        assert len(new_history) == initial_count + 1
        
        # 清空历史
        self.service.clear_prediction_history()
        empty_history = self.service.get_prediction_history()
        assert len(empty_history) == 0
        
    def test_get_model_info(self):
        """测试获取模型信息"""
        model_info = self.service.get_model_info()
        
        assert 'loaded_models' in model_info
        assert 'available_analyzers' in model_info
        assert 'prediction_history_count' in model_info
        
        analyzers = model_info['available_analyzers']
        assert 'sentiment_analyzer' in analyzers
        assert 'keyword_extractor' in analyzers
        assert analyzers['sentiment_analyzer'] is True
        assert analyzers['keyword_extractor'] is True
        
    def test_save_load_predictions(self):
        """测试预测结果保存和加载"""
        # 创建一些预测
        texts = ["测试文本1", "测试文本2"]
        result1 = self.service.predict_sentiment(texts, methods=['dictionary'])
        result2 = self.service.extract_keywords_batch(texts, methods=['tfidf'])
        
        predictions = [result1, result2]
        
        # 保存到临时文件
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_file:
            file_path = tmp_file.name
        
        try:
            self.service.save_predictions(file_path, predictions)
            assert os.path.exists(file_path)
            
            # 加载预测
            loaded_predictions = self.service.load_predictions(file_path)
            
            assert len(loaded_predictions) == 2
            assert loaded_predictions[0]['texts'] == result1['texts']
            assert loaded_predictions[1]['texts_count'] == result2['texts_count']
            
        finally:
            if os.path.exists(file_path):
                os.unlink(file_path)
                
    def test_empty_text_handling(self):
        """测试空文本处理"""
        # 空列表
        with pytest.raises((ValueError, IndexError)):
            self.service.predict_sentiment([], methods=['dictionary'])
            
        # 空字符串
        result = self.service.predict_sentiment("", methods=['dictionary'])
        # 应该能处理但可能返回错误或特殊值
        assert 'texts' in result
        
    def test_invalid_methods(self):
        """测试无效方法处理"""
        texts = ["测试文本"]
        
        # 使用不存在的方法
        result = self.service.predict_sentiment(texts, methods=['nonexistent_method'])
        
        # 应该不会崩溃，但可能没有预期的结果
        assert 'texts' in result
        assert 'predictions' in result
        
    def test_large_text_batch(self):
        """测试大批量文本处理"""
        # 创建大量文本
        texts = [f"测试文本{i}" for i in range(50)]
        
        result = self.service.predict_sentiment(texts, methods=['dictionary'])
        
        assert len(result['texts']) == 50
        assert 'dictionary' in result['predictions']
        assert len(result['predictions']['dictionary']['labels']) == 50


if __name__ == "__main__":
    pytest.main([__file__])
