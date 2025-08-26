#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试时间序列分析模块
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from datetime import datetime, timedelta
from chinese_herbal_sentiment.core.time_series_analysis import TimeSeriesAnalyzer


class TestTimeSeriesAnalyzer:
    """测试时间序列分析器"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        self.analyzer = TimeSeriesAnalyzer()
        
    def test_initialization(self):
        """测试初始化"""
        assert self.analyzer.data is None
        assert self.analyzer.time_column is None
        assert self.analyzer.value_column is None
        assert self.analyzer.frequency is None
        assert isinstance(self.analyzer.trend_results, dict)
        assert isinstance(self.analyzer.seasonal_results, dict)
        assert isinstance(self.analyzer.forecast_results, dict)
        
    def test_generate_sample_data(self):
        """测试生成示例数据"""
        data = self.analyzer.generate_sample_data(
            start_date='2023-01-01',
            periods=100,
            frequency='D'
        )
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 100
        assert 'date' in data.columns
        assert 'sentiment_score' in data.columns
        assert 'review_count' in data.columns
        
        # 检查数据范围
        assert data['sentiment_score'].min() >= 1.0
        assert data['sentiment_score'].max() <= 10.0
        
        # 检查日期范围
        assert data['date'].min() == pd.Timestamp('2023-01-01')
        assert len(data['date'].unique()) == 100
        
    def test_load_data(self):
        """测试数据加载"""
        # 生成测试数据
        sample_data = self.analyzer.generate_sample_data(periods=50)
        
        # 加载数据
        success = self.analyzer.load_data(
            data=sample_data,
            time_column='date',
            value_column='sentiment_score'
        )
        
        assert success is True
        assert self.analyzer.data is not None
        assert len(self.analyzer.data) == 50
        assert self.analyzer.time_column == 'date'
        assert self.analyzer.value_column == 'sentiment_score'
        assert self.analyzer.frequency is not None
        
    def test_detect_frequency(self):
        """测试频率检测"""
        # 日频率数据
        daily_data = self.analyzer.generate_sample_data(periods=30, frequency='D')
        self.analyzer.load_data(daily_data, 'date', 'sentiment_score')
        assert self.analyzer.frequency == 'D'
        
        # 周频率数据
        weekly_dates = pd.date_range('2023-01-01', periods=10, freq='W')
        weekly_data = pd.DataFrame({
            'date': weekly_dates,
            'value': np.random.random(10)
        })
        
        analyzer_weekly = TimeSeriesAnalyzer()
        analyzer_weekly.load_data(weekly_data, 'date', 'value')
        assert analyzer_weekly.frequency == 'W'
        
    def test_trend_analysis_linear(self):
        """测试线性趋势分析"""
        # 创建带明显趋势的数据
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        trend_values = np.linspace(5, 8, 100) + np.random.normal(0, 0.1, 100)
        
        data = pd.DataFrame({
            'date': dates,
            'value': trend_values
        })
        
        self.analyzer.load_data(data, 'date', 'value')
        
        # 进行趋势分析
        results = self.analyzer.trend_analysis(method='linear')
        
        assert 'method' in results
        assert 'trend_direction' in results
        assert 'trend_strength' in results
        assert 'trend_line' in results
        
        assert results['method'] == 'linear'
        assert results['trend_direction'] == 'increasing'  # 应该检测到上升趋势
        assert results['trend_strength'] > 0.5  # 趋势应该较强
        
    def test_trend_analysis_polynomial(self):
        """测试多项式趋势分析"""
        sample_data = self.analyzer.generate_sample_data(periods=50)
        self.analyzer.load_data(sample_data, 'date', 'sentiment_score')
        
        results = self.analyzer.trend_analysis(method='polynomial')
        
        assert results['method'] == 'polynomial'
        assert 'coefficients' in results
        assert 'trend_line' in results
        assert len(results['coefficients']) == 3  # 二次多项式
        
    def test_trend_analysis_moving_average(self):
        """测试移动平均趋势分析"""
        sample_data = self.analyzer.generate_sample_data(periods=50)
        self.analyzer.load_data(sample_data, 'date', 'sentiment_score')
        
        results = self.analyzer.trend_analysis(method='moving_average', window=7)
        
        assert results['method'] == 'moving_average'
        assert results['window_size'] == 7
        assert 'trend_line' in results
        
    def test_seasonal_analysis(self):
        """测试季节性分析"""
        # 创建带明显季节性的数据
        dates = pd.date_range('2023-01-01', periods=365, freq='D')
        seasonal_pattern = np.sin(2 * np.pi * np.arange(365) / 7)  # 周季节性
        trend = np.linspace(7, 8, 365)
        noise = np.random.normal(0, 0.1, 365)
        values = trend + seasonal_pattern + noise
        
        data = pd.DataFrame({
            'date': dates,
            'value': values
        })
        
        self.analyzer.load_data(data, 'date', 'value')
        
        # 进行季节性分析
        results = self.analyzer.seasonal_analysis()
        
        assert 'model' in results
        assert 'period' in results
        
        # 简单季节性分析至少应该有这些字段
        if 'seasonal_strength' in results:
            assert isinstance(results['seasonal_strength'], (int, float))
            
    def test_forecast_linear(self):
        """测试线性预测"""
        sample_data = self.analyzer.generate_sample_data(periods=50)
        self.analyzer.load_data(sample_data, 'date', 'sentiment_score')
        
        results = self.analyzer.forecast(periods=10, method='linear')
        
        assert 'method' in results
        assert 'periods' in results
        assert 'predictions' in results
        assert 'future_dates' in results
        
        assert results['method'] == 'linear'
        assert results['periods'] == 10
        assert len(results['predictions']) == 10
        assert len(results['future_dates']) == 10
        
        # 检查预测值是否合理
        predictions = results['predictions']
        assert all(isinstance(p, (int, float)) for p in predictions)
        
    def test_forecast_auto_method(self):
        """测试自动方法选择预测"""
        sample_data = self.analyzer.generate_sample_data(periods=30)
        self.analyzer.load_data(sample_data, 'date', 'sentiment_score')
        
        results = self.analyzer.forecast(periods=5, method='auto')
        
        assert 'method' in results
        assert 'predictions' in results
        assert len(results['predictions']) == 5
        
    def test_detect_anomalies_iqr(self):
        """测试IQR异常值检测"""
        # 创建含异常值的数据
        normal_values = np.random.normal(7, 1, 95)
        anomaly_values = [2.0, 15.0, 1.5, 12.0, 0.5]  # 明显异常值
        all_values = np.concatenate([normal_values, anomaly_values])
        
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'value': all_values
        })
        
        self.analyzer.load_data(data, 'date', 'value')
        
        results = self.analyzer.detect_anomalies(method='iqr', sensitivity=1.5)
        
        assert 'method' in results
        assert 'anomaly_count' in results
        assert 'anomaly_indices' in results
        assert 'anomaly_percentage' in results
        
        assert results['method'] == 'iqr'
        assert results['anomaly_count'] > 0  # 应该检测到异常值
        assert len(results['anomaly_indices']) == results['anomaly_count']
        
    def test_detect_anomalies_zscore(self):
        """测试Z-score异常值检测"""
        sample_data = self.analyzer.generate_sample_data(periods=50)
        self.analyzer.load_data(sample_data, 'date', 'sentiment_score')
        
        results = self.analyzer.detect_anomalies(method='zscore', sensitivity=2.0)
        
        assert results['method'] == 'zscore'
        assert 'anomaly_count' in results
        assert 'anomaly_percentage' in results
        
    def test_generate_report(self):
        """测试报告生成"""
        sample_data = self.analyzer.generate_sample_data(periods=50)
        self.analyzer.load_data(sample_data, 'date', 'sentiment_score')
        
        # 进行一些分析
        self.analyzer.trend_analysis()
        self.analyzer.seasonal_analysis()
        self.analyzer.forecast(periods=10)
        
        # 生成报告
        report = self.analyzer.generate_report()
        
        assert isinstance(report, str)
        assert "# 时间序列分析报告" in report
        assert "数据基本信息" in report
        assert "趋势分析" in report
        assert "异常值检测" in report
        
    def test_empty_data_handling(self):
        """测试空数据处理"""
        empty_df = pd.DataFrame({'date': [], 'value': []})
        
        success = self.analyzer.load_data(empty_df, 'date', 'value')
        # 应该能加载但可能有警告
        
        if success:
            # 如果加载成功，确保后续操作不会崩溃
            assert len(self.analyzer.data) == 0
            
    def test_single_point_data(self):
        """测试单点数据处理"""
        single_point_df = pd.DataFrame({
            'date': [pd.Timestamp('2023-01-01')],
            'value': [7.5]
        })
        
        success = self.analyzer.load_data(single_point_df, 'date', 'value')
        
        if success:
            # 单点数据不应该能进行大多数分析
            with pytest.raises((ValueError, IndexError)):
                self.analyzer.trend_analysis()
                
    def test_missing_values_handling(self):
        """测试缺失值处理"""
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        values = np.random.random(20)
        values[5:8] = np.nan  # 插入缺失值
        
        data = pd.DataFrame({
            'date': dates,
            'value': values
        })
        
        self.analyzer.load_data(data, 'date', 'value')
        
        # 检查数据是否正确处理了缺失值
        assert not self.analyzer.data[self.analyzer.value_column].isna().any()
        
    def test_no_data_operations(self):
        """测试未加载数据时的操作"""
        # 未加载数据时的操作应该抛出错误
        with pytest.raises(ValueError):
            self.analyzer.trend_analysis()
            
        with pytest.raises(ValueError):
            self.analyzer.seasonal_analysis()
            
        with pytest.raises(ValueError):
            self.analyzer.forecast()
            
        with pytest.raises(ValueError):
            self.analyzer.detect_anomalies()
            
        with pytest.raises(ValueError):
            self.analyzer.generate_report()


if __name__ == "__main__":
    pytest.main([__file__])
