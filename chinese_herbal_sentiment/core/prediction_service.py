#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
预测服务模块
提供统一的预测接口，支持情感分析、质量评分、关键词预测等功能
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Any, Optional, Union, Tuple
import warnings
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# 尝试导入各种模型
try:
    from .sentiment_analysis import SentimentAnalysis
    from .keyword_extraction import KeywordExtraction
    from .regression_analysis import SupplyChainRegression
except ImportError:
    try:
        from chinese_herbal_sentiment.core.sentiment_analysis import SentimentAnalysis
        from chinese_herbal_sentiment.core.keyword_extraction import KeywordExtraction
        from chinese_herbal_sentiment.core.regression_analysis import SupplyChainRegression
    except ImportError:
        print("Warning: Some core modules not available")

# 可选的深度学习模型导入
try:
    from .bert_sentiment_analysis import BERTSentimentAnalysis
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False

try:
    from .deep_learning_sentiment import DeepLearningSentiment
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False

try:
    from .textcnn_sentiment_analysis import TextCNNSentimentAnalysis
    TEXTCNN_AVAILABLE = True
except ImportError:
    TEXTCNN_AVAILABLE = False


class ModelManager:
    """模型管理器"""
    
    def __init__(self, models_dir: str = "models"):
        """
        初始化模型管理器
        
        Args:
            models_dir: 模型存储目录
        """
        self.models_dir = models_dir
        self.loaded_models = {}
        self.model_metadata = {}
        
    def load_model(self, model_name: str, model_path: str, model_type: str = "auto") -> bool:
        """
        加载模型
        
        Args:
            model_name: 模型名称
            model_path: 模型路径
            model_type: 模型类型
            
        Returns:
            是否加载成功
        """
        try:
            if model_type == "auto":
                model_type = self._detect_model_type(model_path)
            
            if model_type == "sklearn":
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            elif model_type == "regression":
                model = SupplyChainRegression()
                model.load_model(model_path)
            elif model_type == "torch":
                model = torch.load(model_path, map_location='cpu')
            else:
                print(f"Unsupported model type: {model_type}")
                return False
            
            self.loaded_models[model_name] = model
            self.model_metadata[model_name] = {
                'type': model_type,
                'path': model_path,
                'loaded_at': datetime.now().isoformat()
            }
            
            print(f"Successfully loaded model: {model_name}")
            return True
            
        except Exception as e:
            print(f"Failed to load model {model_name}: {str(e)}")
            return False
    
    def _detect_model_type(self, model_path: str) -> str:
        """
        自动检测模型类型
        
        Args:
            model_path: 模型路径
            
        Returns:
            模型类型
        """
        if model_path.endswith('.pkl'):
            return "sklearn"
        elif model_path.endswith('.pt') or model_path.endswith('.pth'):
            return "torch"
        elif 'regression' in model_path.lower():
            return "regression"
        else:
            return "sklearn"
    
    def get_model(self, model_name: str):
        """获取已加载的模型"""
        return self.loaded_models.get(model_name)
    
    def list_models(self) -> Dict[str, Dict]:
        """列出所有已加载的模型"""
        return self.model_metadata


class PredictionService:
    """预测服务类"""
    
    def __init__(self, models_dir: str = "models"):
        """
        初始化预测服务
        
        Args:
            models_dir: 模型目录
        """
        self.models_dir = models_dir
        self.model_manager = ModelManager(models_dir)
        
        # 初始化分析器
        self.sentiment_analyzer = SentimentAnalysis()
        self.keyword_extractor = KeywordExtraction()
        
        # 尝试初始化深度学习模型
        self.bert_analyzer = None
        self.lstm_analyzer = None
        self.textcnn_analyzer = None
        
        if BERT_AVAILABLE:
            try:
                self.bert_analyzer = BERTSentimentAnalysis()
            except:
                pass
        
        if LSTM_AVAILABLE:
            try:
                self.lstm_analyzer = DeepLearningSentiment()
            except:
                pass
        
        if TEXTCNN_AVAILABLE:
            try:
                self.textcnn_analyzer = TextCNNSentimentAnalysis()
            except:
                pass
        
        self.regression_model = None
        
        # 预测历史
        self.prediction_history = []
    
    def load_trained_models(self, models_config: Dict[str, str]) -> Dict[str, bool]:
        """
        批量加载训练好的模型
        
        Args:
            models_config: 模型配置字典 {model_name: model_path}
            
        Returns:
            加载结果字典
        """
        results = {}
        for model_name, model_path in models_config.items():
            if os.path.exists(model_path):
                results[model_name] = self.model_manager.load_model(model_name, model_path)
            else:
                print(f"Model file not found: {model_path}")
                results[model_name] = False
        
        return results
    
    def predict_sentiment(self, 
                         texts: Union[str, List[str]], 
                         methods: List[str] = ['dictionary', 'svm'],
                         return_probabilities: bool = False) -> Dict[str, Any]:
        """
        预测文本情感
        
        Args:
            texts: 输入文本或文本列表
            methods: 使用的方法列表
            return_probabilities: 是否返回概率
            
        Returns:
            预测结果字典
        """
        if isinstance(texts, str):
            texts = [texts]
        
        results = {
            'texts': texts,
            'predictions': {},
            'timestamp': datetime.now().isoformat(),
            'methods_used': methods
        }
        
        # 创建临时DataFrame
        df = pd.DataFrame({'评论内容': texts})
        
        try:
            # 词典方法
            if 'dictionary' in methods:
                dict_preds = []
                dict_scores = []
                for text in texts:
                    score = self.sentiment_analyzer.dictionary_based_analysis(text)
                    dict_scores.append(score)
                    if score > 0.2:
                        dict_preds.append(1)  # 正面
                    elif score < -0.2:
                        dict_preds.append(-1)  # 负面
                    else:
                        dict_preds.append(0)  # 中性
                
                results['predictions']['dictionary'] = {
                    'labels': dict_preds,
                    'scores': dict_scores
                }
            
            # 机器学习方法
            ml_methods = [m for m in methods if m in ['svm', 'nb', 'lr']]
            if ml_methods:
                # 提取特征
                features, vectorizer = self.sentiment_analyzer.extract_features(texts, save_vectorizer=False)
                
                for method in ml_methods:
                    model_name = f"{method}_model"
                    model = self.model_manager.get_model(model_name)
                    
                    if model is not None:
                        preds = model.predict(features)
                        if return_probabilities and hasattr(model, 'predict_proba'):
                            probs = model.predict_proba(features)
                            results['predictions'][method] = {
                                'labels': preds.tolist(),
                                'probabilities': probs.tolist()
                            }
                        else:
                            results['predictions'][method] = {
                                'labels': preds.tolist()
                            }
                    else:
                        print(f"Model {model_name} not loaded, skipping {method}")
            
            # 深度学习方法（如果可用）
            if 'bert' in methods and self.bert_analyzer:
                try:
                    # 注意：这需要预训练的BERT模型
                    print("BERT prediction requires pre-trained model")
                except Exception as e:
                    print(f"BERT prediction failed: {e}")
            
        except Exception as e:
            results['error'] = str(e)
        
        # 记录预测历史
        self.prediction_history.append(results)
        
        return results
    
    def predict_quality_score(self, 
                             features: Union[Dict[str, float], pd.DataFrame, np.ndarray],
                             confidence_interval: bool = True) -> Dict[str, Any]:
        """
        预测质量评分
        
        Args:
            features: 特征数据
            confidence_interval: 是否返回置信区间
            
        Returns:
            预测结果
        """
        if self.regression_model is None:
            # 尝试加载回归模型
            regression_model_path = os.path.join(self.models_dir, 'supply_chain_regression_model.pkl')
            if os.path.exists(regression_model_path):
                self.regression_model = SupplyChainRegression()
                self.regression_model.load_model(regression_model_path)
            else:
                return {'error': 'Regression model not available'}
        
        try:
            # 处理输入数据
            if isinstance(features, dict):
                # 单个样本
                df = pd.DataFrame([features])
                X = df.values
            elif isinstance(features, pd.DataFrame):
                X = features.values
            elif isinstance(features, np.ndarray):
                X = features
            else:
                return {'error': 'Unsupported input format'}
            
            # 预测
            if confidence_interval:
                predictions, lower_bound, upper_bound = self.regression_model.predict(
                    X, return_intervals=True
                )
                results = {
                    'predictions': predictions.tolist(),
                    'lower_bound': lower_bound.tolist(),
                    'upper_bound': upper_bound.tolist(),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                predictions = self.regression_model.predict(X)
                results = {
                    'predictions': predictions.tolist(),
                    'timestamp': datetime.now().isoformat()
                }
            
            return results
            
        except Exception as e:
            return {'error': str(e)}
    
    def extract_keywords_batch(self, 
                              texts: List[str], 
                              methods: List[str] = ['tfidf', 'textrank'],
                              top_k: int = 10) -> Dict[str, Any]:
        """
        批量关键词提取
        
        Args:
            texts: 文本列表
            methods: 提取方法
            top_k: 返回关键词数量
            
        Returns:
            关键词提取结果
        """
        try:
            results = {
                'texts_count': len(texts),
                'methods_used': methods,
                'keywords': {},
                'timestamp': datetime.now().isoformat()
            }
            
            if 'tfidf' in methods:
                tfidf_keywords = self.keyword_extractor.tfidf_extraction(texts, top_k)
                results['keywords']['tfidf'] = tfidf_keywords
            
            if 'textrank' in methods:
                textrank_keywords = self.keyword_extractor.textrank_extraction(texts, top_k)
                results['keywords']['textrank'] = textrank_keywords
            
            if 'lda' in methods:
                lda_keywords, lda_topics = self.keyword_extractor.lda_extraction(texts, top_k=top_k)
                results['keywords']['lda'] = {
                    'keywords': lda_keywords,
                    'topics': lda_topics
                }
            
            return results
            
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_comprehensive(self, 
                            texts: Union[str, List[str]], 
                            include_sentiment: bool = True,
                            include_keywords: bool = True,
                            include_quality: bool = False,
                            quality_features: Optional[Dict] = None) -> Dict[str, Any]:
        """
        综合分析：同时进行情感分析、关键词提取等
        
        Args:
            texts: 输入文本
            include_sentiment: 是否包含情感分析
            include_keywords: 是否包含关键词提取
            include_quality: 是否包含质量评分
            quality_features: 质量评分的特征数据
            
        Returns:
            综合分析结果
        """
        if isinstance(texts, str):
            texts = [texts]
        
        results = {
            'analysis_type': 'comprehensive',
            'texts_count': len(texts),
            'timestamp': datetime.now().isoformat(),
            'results': {}
        }
        
        try:
            # 情感分析
            if include_sentiment:
                sentiment_results = self.predict_sentiment(texts, methods=['dictionary', 'svm'])
                results['results']['sentiment_analysis'] = sentiment_results
            
            # 关键词提取
            if include_keywords:
                keyword_results = self.extract_keywords_batch(texts, methods=['tfidf', 'textrank'])
                results['results']['keyword_extraction'] = keyword_results
            
            # 质量评分
            if include_quality and quality_features:
                quality_results = self.predict_quality_score(quality_features)
                results['results']['quality_prediction'] = quality_results
            
            return results
            
        except Exception as e:
            results['error'] = str(e)
            return results
    
    def get_prediction_history(self, limit: int = 10) -> List[Dict]:
        """
        获取预测历史
        
        Args:
            limit: 返回记录数量限制
            
        Returns:
            预测历史列表
        """
        return self.prediction_history[-limit:]
    
    def clear_prediction_history(self):
        """清空预测历史"""
        self.prediction_history = []
        print("Prediction history cleared")
    
    def save_predictions(self, file_path: str, predictions: List[Dict]):
        """
        保存预测结果到文件
        
        Args:
            file_path: 保存路径
            predictions: 预测结果列表
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(predictions, f, ensure_ascii=False, indent=2)
            print(f"Predictions saved to: {file_path}")
        except Exception as e:
            print(f"Failed to save predictions: {e}")
    
    def load_predictions(self, file_path: str) -> List[Dict]:
        """
        从文件加载预测结果
        
        Args:
            file_path: 文件路径
            
        Returns:
            预测结果列表
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                predictions = json.load(f)
            print(f"Predictions loaded from: {file_path}")
            return predictions
        except Exception as e:
            print(f"Failed to load predictions: {e}")
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            'loaded_models': self.model_manager.list_models(),
            'available_analyzers': {
                'sentiment_analyzer': True,
                'keyword_extractor': True,
                'bert_analyzer': self.bert_analyzer is not None,
                'lstm_analyzer': self.lstm_analyzer is not None,
                'textcnn_analyzer': self.textcnn_analyzer is not None,
                'regression_model': self.regression_model is not None
            },
            'prediction_history_count': len(self.prediction_history)
        }


def demo_prediction_service():
    """演示预测服务功能"""
    print("=== 预测服务演示 ===\n")
    
    # 创建预测服务
    service = PredictionService()
    
    # 演示文本
    demo_texts = [
        "这个中药质量很好，效果不错，物流也很快",
        "包装破损严重，产品质量很差，不推荐购买",
        "服务态度一般，产品还可以，价格合理",
        "非常满意的购买体验，产品质量优秀，会再次购买"
    ]
    
    print("1. 情感分析预测演示")
    print("输入文本:")
    for i, text in enumerate(demo_texts, 1):
        print(f"   {i}. {text}")
    print()
    
    # 情感分析
    sentiment_results = service.predict_sentiment(demo_texts, methods=['dictionary'])
    print("情感分析结果:")
    for i, (text, pred, score) in enumerate(zip(
        demo_texts, 
        sentiment_results['predictions']['dictionary']['labels'],
        sentiment_results['predictions']['dictionary']['scores']
    ), 1):
        sentiment_name = "正面" if pred == 1 else ("负面" if pred == -1 else "中性")
        print(f"   {i}. {sentiment_name} (评分: {score:.3f})")
    print()
    
    # 关键词提取
    print("2. 关键词提取演示")
    keyword_results = service.extract_keywords_batch(demo_texts, methods=['tfidf'])
    
    if 'tfidf' in keyword_results['keywords']:
        print("TF-IDF关键词:")
        for i, keywords in enumerate(keyword_results['keywords']['tfidf'][:3], 1):
            top_keywords = [word for word, score in keywords[:5]]
            print(f"   文本{i}: {', '.join(top_keywords)}")
    print()
    
    # 质量评分预测演示（使用模拟数据）
    print("3. 质量评分预测演示")
    
    # 创建并训练回归模型用于演示
    try:
        from .regression_analysis import SupplyChainRegression
        regressor = SupplyChainRegression()
        
        # 生成模拟数据并训练
        print("   正在训练回归模型...")
        demo_data = regressor.generate_supply_chain_data(100)
        feature_columns = [col for col in demo_data.columns 
                          if col not in ['service_quality', 'enterprise_size', 'product_type', 'region']]
        
        X, y = regressor.prepare_data(
            demo_data, 
            'service_quality', 
            feature_columns
        )
        regressor.train(X, y, test_size=0.2)
        
        # 保存模型
        os.makedirs('output/models', exist_ok=True)
        regressor.save_model('output/models/demo_regression_model.pkl')
        
        # 加载到预测服务
        service.regression_model = regressor
        
        # 预测演示
        demo_features = {
            'material_quality': 8.5,
            'technology': 7.8,
            'delivery_speed': 8.2,
            'after_sales_service': 7.5
        }
        
        # 补全所有需要的特征
        for col in feature_columns:
            if col not in demo_features:
                demo_features[col] = 7.5  # 默认值
        
        quality_result = service.predict_quality_score(demo_features)
        
        if 'error' not in quality_result:
            print(f"   质量评分预测: {quality_result['predictions'][0]:.2f}")
        else:
            print(f"   质量评分预测失败: {quality_result['error']}")
        
    except Exception as e:
        print(f"   质量评分演示失败: {e}")
    
    print()
    
    # 综合分析演示
    print("4. 综合分析演示")
    comprehensive_results = service.analyze_comprehensive(
        demo_texts[:2],  # 使用前两个文本
        include_sentiment=True,
        include_keywords=True
    )
    
    print(f"   分析了 {comprehensive_results['texts_count']} 条文本")
    print(f"   包含的分析类型: {list(comprehensive_results['results'].keys())}")
    print()
    
    # 模型信息
    print("5. 模型信息")
    model_info = service.get_model_info()
    print(f"   可用的分析器: {model_info['available_analyzers']}")
    print(f"   预测历史记录数: {model_info['prediction_history_count']}")
    
    print("\n=== 演示完成 ===")
    return service


if __name__ == "__main__":
    demo_prediction_service()
