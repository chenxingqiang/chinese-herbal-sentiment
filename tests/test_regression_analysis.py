#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试回归分析模块
"""

import pytest
import numpy as np
import pandas as pd
import os
import tempfile
from chinese_herbal_sentiment.core.regression_analysis import SupplyChainRegression


class TestSupplyChainRegression:
    """测试供应链回归分析类"""

    def setup_method(self):
        """每个测试方法前的设置"""
        self.regressor = SupplyChainRegression()

    def test_initialization(self):
        """测试初始化"""
        assert self.regressor.model_type == 'linear'
        assert self.regressor.model is not None
        assert self.regressor.scaler is not None
        assert not self.regressor.is_fitted

    def test_different_model_types(self):
        """测试不同模型类型的初始化"""
        linear_reg = SupplyChainRegression('linear')
        ridge_reg = SupplyChainRegression('ridge')
        lasso_reg = SupplyChainRegression('lasso')

        assert linear_reg.model_type == 'linear'
        assert ridge_reg.model_type == 'ridge'
        assert lasso_reg.model_type == 'lasso'

        # 测试无效模型类型
        with pytest.raises(ValueError):
            SupplyChainRegression('invalid_type')

    def test_generate_sample_data(self):
        """测试生成示例数据"""
        data = self.regressor.generate_supply_chain_data(100)

        assert isinstance(data, pd.DataFrame)
        assert len(data) == 100
        assert 'service_quality' in data.columns
        assert 'material_quality' in data.columns

        # 检查数据范围
        assert data['service_quality'].min() >= 1.0
        assert data['service_quality'].max() <= 10.0

    def test_prepare_data(self):
        """测试数据准备"""
        # 生成测试数据
        data = self.regressor.generate_supply_chain_data(50)

        feature_columns = ['material_quality', 'technology', 'delivery_speed']
        categorical_columns = ['enterprise_size']

        X, y = self.regressor.prepare_data(
            data=data,
            target_column='service_quality',
            feature_columns=feature_columns + categorical_columns,
            categorical_columns=categorical_columns
        )

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == len(feature_columns) + len(categorical_columns)

    def test_train_model(self):
        """测试模型训练"""
        # 生成训练数据
        data = self.regressor.generate_supply_chain_data(100)

        feature_columns = ['material_quality', 'technology', 'delivery_speed']
        X, y = self.regressor.prepare_data(
            data=data,
            target_column='service_quality',
            feature_columns=feature_columns
        )

        # 训练模型
        results = self.regressor.train(X, y, test_size=0.2)

        assert self.regressor.is_fitted
        assert 'train_r2' in results
        assert 'test_r2' in results
        assert 'coefficients' in results
        assert 'intercept' in results

        # 检查系数数量
        assert len(results['coefficients']) == len(feature_columns)

    def test_predict(self):
        """测试预测功能"""
        # 训练模型
        data = self.regressor.generate_supply_chain_data(100)
        feature_columns = ['material_quality', 'technology', 'delivery_speed']
        X, y = self.regressor.prepare_data(
            data=data,
            target_column='service_quality',
            feature_columns=feature_columns
        )

        self.regressor.train(X, y)

        # 预测
        X_new = X[:5]  # 使用前5个样本
        predictions = self.regressor.predict(X_new)

        assert len(predictions) == 5
        assert all(isinstance(p, (int, float)) for p in predictions)

        # 测试置信区间预测
        predictions, lower, upper = self.regressor.predict(X_new, return_intervals=True)

        assert len(predictions) == len(lower) == len(upper) == 5
        assert all(l <= p <= u for p, l, u in zip(predictions, lower, upper))

    def test_feature_importance(self):
        """测试特征重要性"""
        # 训练模型
        data = self.regressor.generate_supply_chain_data(100)
        feature_columns = ['material_quality', 'technology', 'delivery_speed']
        X, y = self.regressor.prepare_data(
            data=data,
            target_column='service_quality',
            feature_columns=feature_columns
        )

        self.regressor.train(X, y)

        # 获取特征重要性
        importance_df = self.regressor.feature_importance()

        assert isinstance(importance_df, pd.DataFrame)
        assert len(importance_df) == len(feature_columns)
        assert 'feature' in importance_df.columns
        assert 'coefficient' in importance_df.columns
        assert 'importance_rank' in importance_df.columns

        # 检查排序
        assert importance_df['importance_rank'].tolist() == list(range(1, len(feature_columns) + 1))

    def test_save_load_model(self):
        """测试模型保存和加载"""
        # 训练模型
        data = self.regressor.generate_supply_chain_data(50)
        feature_columns = ['material_quality', 'technology']
        X, y = self.regressor.prepare_data(
            data=data,
            target_column='service_quality',
            feature_columns=feature_columns
        )

        self.regressor.train(X, y)

        # 保存模型
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            model_path = tmp_file.name

        try:
            self.regressor.save_model(model_path)
            assert os.path.exists(model_path)

            # 创建新的回归器并加载模型
            new_regressor = SupplyChainRegression()
            new_regressor.load_model(model_path)

            assert new_regressor.is_fitted
            assert new_regressor.feature_names == self.regressor.feature_names
            assert new_regressor.target_name == self.regressor.target_name

            # 测试预测一致性
            X_test = X[:3]
            pred1 = self.regressor.predict(X_test)
            pred2 = new_regressor.predict(X_test)

            np.testing.assert_array_almost_equal(pred1, pred2, decimal=5)

        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)

    def test_generate_report(self):
        """测试报告生成"""
        # 训练模型
        data = self.regressor.generate_supply_chain_data(50)
        feature_columns = ['material_quality', 'technology']
        X, y = self.regressor.prepare_data(
            data=data,
            target_column='service_quality',
            feature_columns=feature_columns
        )

        self.regressor.train(X, y)

        # 生成报告
        report = self.regressor.generate_report()

        assert isinstance(report, str)
        assert "# 供应链质量回归分析报告" in report
        assert "模型基本信息" in report
        assert "模型性能" in report
        assert "特征重要性" in report

    def test_unfitted_model_errors(self):
        """测试未训练模型的错误处理"""
        X_test = np.random.random((5, 3))

        # 未训练模型时的预测应该抛出错误
        with pytest.raises(ValueError):
            self.regressor.predict(X_test)

        with pytest.raises(ValueError):
            self.regressor.feature_importance()

        with pytest.raises(ValueError):
            self.regressor.generate_report()

        with pytest.raises(ValueError):
            self.regressor.save_model('test.pkl')


if __name__ == "__main__":
    pytest.main([__file__])
