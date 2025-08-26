#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
供应链质量回归分析模块
实现多元线性回归分析、模型诊断和预测功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
try:
    from scipy import stats
    from scipy.stats import shapiro, jarque_bera, normaltest
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
import warnings
import pickle
import os
from typing import Dict, List, Tuple, Optional, Any
import json

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class SupplyChainRegression:
    """供应链质量回归分析类"""

    def __init__(self, model_type='linear'):
        """
        初始化回归分析器

        Args:
            model_type: 模型类型 ('linear', 'ridge', 'lasso')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.target_name = ""
        self.is_fitted = False
        self.training_history = {}

        # 初始化模型
        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'ridge':
            self.model = Ridge(alpha=1.0)
        elif model_type == 'lasso':
            self.model = Lasso(alpha=1.0)
        else:
            raise ValueError("Unsupported model type. Choose from: 'linear', 'ridge', 'lasso'")

    def prepare_data(self, data: pd.DataFrame,
                    target_column: str,
                    feature_columns: Optional[List[str]] = None,
                    categorical_columns: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备回归分析数据

        Args:
            data: 输入数据框
            target_column: 目标变量列名
            feature_columns: 特征列名列表
            categorical_columns: 分类变量列名列表

        Returns:
            X, y: 特征矩阵和目标向量
        """
        df = data.copy()

        # 处理缺失值
        df = df.dropna()

        # 如果没有指定特征列，使用除目标列外的所有数值列
        if feature_columns is None:
            feature_columns = [col for col in df.select_dtypes(include=[np.number]).columns
                             if col != target_column]

        # 处理分类变量
        if categorical_columns:
            for col in categorical_columns:
                if col in df.columns:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                        df[col] = self.label_encoders[col].fit_transform(df[col])
                    else:
                        df[col] = self.label_encoders[col].transform(df[col])

        # 提取特征和目标变量
        X = df[feature_columns].values
        y = df[target_column].values

        # 保存元信息
        self.feature_names = feature_columns
        self.target_name = target_column

        return X, y

    def train(self, X: np.ndarray, y: np.ndarray,
             test_size: float = 0.2,
             random_state: int = 42,
             scale_features: bool = True) -> Dict[str, Any]:
        """
        训练回归模型

        Args:
            X: 特征矩阵
            y: 目标向量
            test_size: 测试集比例
            random_state: 随机种子
            scale_features: 是否标准化特征

        Returns:
            训练结果字典
        """
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # 特征标准化
        if scale_features:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test

        # 训练模型
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True

        # 预测
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)

        # 计算评估指标
        results = {
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'train_mse': mean_squared_error(y_train, y_train_pred),
            'test_mse': mean_squared_error(y_test, y_test_pred),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'coefficients': self.model.coef_,
            'intercept': self.model.intercept_,
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names),
            'n_samples': len(X_train)
        }

        # 交叉验证
        if len(X_train) > 50:  # 只有当样本数足够时才进行交叉验证
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
            results['cv_r2_mean'] = cv_scores.mean()
            results['cv_r2_std'] = cv_scores.std()

        # 模型诊断
        residuals = y_test - y_test_pred
        results['diagnostics'] = self._model_diagnostics(residuals, y_test_pred)

        # 保存训练历史
        self.training_history = results

        return results

    def _model_diagnostics(self, residuals: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        模型诊断

        Args:
            residuals: 残差
            y_pred: 预测值

        Returns:
            诊断结果字典
        """
        diagnostics = {}

                # 残差正态性检验
        if SCIPY_AVAILABLE:
            try:
                shapiro_stat, shapiro_p = shapiro(residuals)
                diagnostics['shapiro_test'] = {
                    'statistic': shapiro_stat,
                    'p_value': shapiro_p,
                    'is_normal': shapiro_p > 0.05
                }
            except:
                diagnostics['shapiro_test'] = None
            
            # Jarque-Bera检验
            try:
                jb_stat, jb_p = jarque_bera(residuals)
                diagnostics['jarque_bera_test'] = {
                    'statistic': jb_stat,
                    'p_value': jb_p,
                    'is_normal': jb_p > 0.05
                }
            except:
                diagnostics['jarque_bera_test'] = None
        else:
            diagnostics['shapiro_test'] = None
            diagnostics['jarque_bera_test'] = None

        # 异方差性检验 (Breusch-Pagan test的简化版本)
        try:
            correlation_coef = np.corrcoef(np.abs(residuals), y_pred)[0, 1]
            diagnostics['heteroscedasticity'] = {
                'correlation': correlation_coef,
                'has_heteroscedasticity': abs(correlation_coef) > 0.3
            }
        except:
            diagnostics['heteroscedasticity'] = None

        # 残差基本统计
        diagnostics['residuals_stats'] = {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'min': np.min(residuals),
            'max': np.max(residuals)
        }
        
        if SCIPY_AVAILABLE:
            diagnostics['residuals_stats'].update({
                'skewness': stats.skew(residuals),
                'kurtosis': stats.kurtosis(residuals)
            })

        return diagnostics

    def predict(self, X_new: np.ndarray, return_intervals: bool = False) -> np.ndarray:
        """
        使用训练好的模型进行预测

        Args:
            X_new: 新的特征数据
            return_intervals: 是否返回预测区间

        Returns:
            预测结果
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # 标准化特征
        X_new_scaled = self.scaler.transform(X_new)

        # 预测
        predictions = self.model.predict(X_new_scaled)

        if return_intervals:
            # 简单的预测区间估计（基于训练误差）
            if 'test_mse' in self.training_history:
                std_error = np.sqrt(self.training_history['test_mse'])
                lower_bound = predictions - 1.96 * std_error
                upper_bound = predictions + 1.96 * std_error
                return predictions, lower_bound, upper_bound

        return predictions

    def feature_importance(self) -> pd.DataFrame:
        """
        计算特征重要性

        Returns:
            特征重要性DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating feature importance")

        # 获取系数的绝对值作为重要性指标
        importance = np.abs(self.model.coef_)

        # 创建DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.coef_,
            'abs_coefficient': importance,
            'importance_rank': np.argsort(importance)[::-1] + 1
        })

        # 标准化重要性到0-1区间
        importance_df['normalized_importance'] = importance / np.max(importance)

        return importance_df.sort_values('abs_coefficient', ascending=False)

    def generate_supply_chain_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        生成供应链质量评价的模拟数据

        Args:
            n_samples: 样本数量

        Returns:
            模拟数据DataFrame
        """
        np.random.seed(42)

        # 生成基础特征
        data = {}

        # 上游（原料）指标
        data['material_quality'] = np.random.normal(8.0, 1.5, n_samples)  # 原料质量
        data['material_consistency'] = np.random.normal(7.8, 1.4, n_samples)  # 原料一致性
        data['material_traceability'] = np.random.normal(7.2, 1.6, n_samples)  # 可追溯性
        data['material_price'] = np.random.normal(7.5, 1.3, n_samples)  # 价格合理性
        data['supply_stability'] = np.random.normal(7.9, 1.4, n_samples)  # 供应稳定性

        # 中游（加工）指标
        data['technology'] = np.random.normal(7.7, 1.5, n_samples)  # 工艺技术
        data['production_efficiency'] = np.random.normal(7.6, 1.4, n_samples)  # 生产效率
        data['quality_standard'] = np.random.normal(7.8, 1.3, n_samples)  # 质检标准
        data['product_consistency'] = np.random.normal(7.5, 1.5, n_samples)  # 产品一致性
        data['processing_environment'] = np.random.normal(7.4, 1.6, n_samples)  # 加工环境

        # 下游（销售与物流）指标
        data['delivery_speed'] = np.random.normal(8.1, 1.3, n_samples)  # 交货速度
        data['packaging'] = np.random.normal(7.9, 1.4, n_samples)  # 包装质量
        data['order_accuracy'] = np.random.normal(8.2, 1.2, n_samples)  # 订单准确性
        data['inventory_management'] = np.random.normal(7.7, 1.5, n_samples)  # 库存管理
        data['after_sales_service'] = np.random.normal(7.3, 1.6, n_samples)  # 售后服务
        data['information_transparency'] = np.random.normal(7.1, 1.7, n_samples)  # 信息透明度

        # 分类变量
        data['enterprise_size'] = np.random.choice(['大型', '中型', '小型'], n_samples, p=[0.3, 0.4, 0.3])
        data['product_type'] = np.random.choice(['饮片', '配方颗粒', '中成药'], n_samples, p=[0.4, 0.3, 0.3])
        data['region'] = np.random.choice(['华东', '华北', '华南', '西南', '其他'], n_samples, p=[0.25, 0.20, 0.20, 0.15, 0.20])

        # 生成目标变量（综合服务质量评分）
        # 使用线性组合加上随机噪声
        service_quality = (
            0.25 * data['material_quality'] +
            0.20 * data['technology'] +
            0.15 * data['delivery_speed'] +
            0.12 * data['order_accuracy'] +
            0.10 * data['after_sales_service'] +
            0.08 * data['material_consistency'] +
            0.05 * data['packaging'] +
            0.05 * data['information_transparency'] +
            np.random.normal(0, 0.5, n_samples)  # 噪声
        )

        # 企业规模影响
        size_effect = {'大型': 0.3, '中型': 0.0, '小型': -0.2}
        for i, size in enumerate(data['enterprise_size']):
            service_quality[i] += size_effect[size]

        # 确保评分在合理范围内
        data['service_quality'] = np.clip(service_quality, 1.0, 10.0)

        # 确保所有数值在合理范围内
        for key in data.keys():
            if isinstance(data[key][0], (int, float)):
                data[key] = np.clip(data[key], 1.0, 10.0)

        return pd.DataFrame(data)

    def visualize_results(self, save_path: str = None):
        """
        可视化回归分析结果

        Args:
            save_path: 保存路径
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before visualization")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. 特征重要性
        importance_df = self.feature_importance()
        top_features = importance_df.head(10)

        axes[0, 0].barh(range(len(top_features)), top_features['abs_coefficient'])
        axes[0, 0].set_yticks(range(len(top_features)))
        axes[0, 0].set_yticklabels(top_features['feature'])
        axes[0, 0].set_xlabel('系数绝对值')
        axes[0, 0].set_title('特征重要性（前10个）')
        axes[0, 0].invert_yaxis()

        # 2. 模型性能对比
        metrics = ['训练R²', '测试R²', '训练MSE', '测试MSE']
        values = [
            self.training_history['train_r2'],
            self.training_history['test_r2'],
            self.training_history['train_mse'],
            self.training_history['test_mse']
        ]

        colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']
        axes[0, 1].bar(metrics[:2], values[:2], color=colors[:2])
        axes[0, 1].set_ylabel('R² 值')
        axes[0, 1].set_title('模型拟合度')
        axes[0, 1].set_ylim(0, 1)

        # 在柱状图上添加数值标签
        for i, v in enumerate(values[:2]):
            axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

        # 3. 残差分析（如果有诊断信息）
        if 'diagnostics' in self.training_history and self.training_history['diagnostics']:
            diag = self.training_history['diagnostics']
            if diag['residuals_stats']:
                stats_data = diag['residuals_stats']
                stats_names = ['均值', '标准差', '偏度', '峰度']
                stats_values = [
                    stats_data['mean'],
                    stats_data['std'],
                    stats_data['skewness'],
                    stats_data['kurtosis']
                ]

                axes[1, 0].bar(stats_names, stats_values, color='lightsteelblue')
                axes[1, 0].set_title('残差统计特征')
                axes[1, 0].set_ylabel('值')

                # 添加数值标签
                for i, v in enumerate(stats_values):
                    axes[1, 0].text(i, v, f'{v:.3f}', ha='center', va='bottom' if v >= 0 else 'top')

        # 4. 回归方程展示
        axes[1, 1].axis('off')
        equation_text = f"服务质量 = {self.model.intercept_:.3f}"

        # 添加主要特征的系数
        importance_df = self.feature_importance()
        for _, row in importance_df.head(5).iterrows():
            coef = row['coefficient']
            feature = row['feature']
            sign = '+' if coef >= 0 else ''
            equation_text += f"\n{sign}{coef:.3f} × {feature}"

        equation_text += "\n+ ε"

        axes[1, 1].text(0.1, 0.5, equation_text, fontsize=12,
                       verticalalignment='center', fontfamily='monospace')
        axes[1, 1].set_title('回归方程')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"回归分析结果已保存到: {save_path}")
        else:
            plt.savefig('regression_analysis_results.png', dpi=300, bbox_inches='tight')

        plt.show()

    def save_model(self, file_path: str):
        """
        保存训练好的模型

        Args:
            file_path: 保存路径
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'model_type': self.model_type,
            'training_history': self.training_history
        }

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"模型已保存到: {file_path}")

    def load_model(self, file_path: str):
        """
        加载保存的模型

        Args:
            file_path: 模型文件路径
        """
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        self.target_name = model_data['target_name']
        self.model_type = model_data['model_type']
        self.training_history = model_data['training_history']
        self.is_fitted = True

        print(f"模型已从 {file_path} 加载")

    def generate_report(self, save_path: str = None) -> str:
        """
        生成详细的分析报告

        Args:
            save_path: 报告保存路径

        Returns:
            报告内容
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating report")

        report = "# 供应链质量回归分析报告\n\n"

        # 模型基本信息
        report += "## 1. 模型基本信息\n\n"
        report += f"- **模型类型**: {self.model_type.upper()}\n"
        report += f"- **特征数量**: {len(self.feature_names)}\n"
        report += f"- **样本数量**: {self.training_history['n_samples']}\n"
        report += f"- **目标变量**: {self.target_name}\n\n"

        # 模型性能
        report += "## 2. 模型性能\n\n"
        hist = self.training_history
        report += f"- **训练集R²**: {hist['train_r2']:.4f}\n"
        report += f"- **测试集R²**: {hist['test_r2']:.4f}\n"
        report += f"- **训练集MSE**: {hist['train_mse']:.4f}\n"
        report += f"- **测试集MSE**: {hist['test_mse']:.4f}\n"
        report += f"- **训练集MAE**: {hist['train_mae']:.4f}\n"
        report += f"- **测试集MAE**: {hist['test_mae']:.4f}\n"

        if 'cv_r2_mean' in hist:
            report += f"- **交叉验证R²**: {hist['cv_r2_mean']:.4f} ± {hist['cv_r2_std']:.4f}\n"

        report += "\n"

        # 特征重要性
        report += "## 3. 特征重要性\n\n"
        importance_df = self.feature_importance()
        report += "| 排名 | 特征名称 | 系数 | 重要性 |\n"
        report += "|------|----------|------|--------|\n"

        for _, row in importance_df.head(10).iterrows():
            report += f"| {row['importance_rank']} | {row['feature']} | {row['coefficient']:.4f} | {row['normalized_importance']:.4f} |\n"

        report += "\n"

        # 模型诊断
        if 'diagnostics' in hist and hist['diagnostics']:
            report += "## 4. 模型诊断\n\n"
            diag = hist['diagnostics']

            if diag['shapiro_test']:
                shapiro = diag['shapiro_test']
                report += f"### 4.1 残差正态性检验 (Shapiro-Wilk)\n"
                report += f"- **统计量**: {shapiro['statistic']:.4f}\n"
                report += f"- **p值**: {shapiro['p_value']:.4f}\n"
                report += f"- **是否服从正态分布**: {'是' if shapiro['is_normal'] else '否'}\n\n"

            if diag['heteroscedasticity']:
                hetero = diag['heteroscedasticity']
                report += f"### 4.2 异方差性检验\n"
                report += f"- **残差与预测值相关系数**: {hetero['correlation']:.4f}\n"
                report += f"- **是否存在异方差性**: {'是' if hetero['has_heteroscedasticity'] else '否'}\n\n"

            if diag['residuals_stats']:
                resid = diag['residuals_stats']
                report += f"### 4.3 残差统计特征\n"
                report += f"- **均值**: {resid['mean']:.4f}\n"
                report += f"- **标准差**: {resid['std']:.4f}\n"
                report += f"- **偏度**: {resid['skewness']:.4f}\n"
                report += f"- **峰度**: {resid['kurtosis']:.4f}\n\n"

        # 回归方程
        report += "## 5. 回归方程\n\n"
        report += f"```\n{self.target_name} = {self.model.intercept_:.4f}"

        for i, (feature, coef) in enumerate(zip(self.feature_names, self.model.coef_)):
            sign = '+' if coef >= 0 else ''
            report += f"\n    {sign}{coef:.4f} × {feature}"

        report += "\n    + ε\n```\n\n"

        # 结论和建议
        report += "## 6. 结论和建议\n\n"

        # 模型拟合度评价
        r2 = hist['test_r2']
        if r2 > 0.8:
            fit_level = "优秀"
        elif r2 > 0.6:
            fit_level = "良好"
        elif r2 > 0.4:
            fit_level = "一般"
        else:
            fit_level = "较差"

        report += f"1. **模型拟合度**: 测试集R²为{r2:.4f}，模型拟合度{fit_level}。\n"

        # 重要特征分析
        top_feature = importance_df.iloc[0]
        report += f"2. **关键影响因素**: {top_feature['feature']}是最重要的影响因素，系数为{top_feature['coefficient']:.4f}。\n"

        # 改进建议
        report += "3. **改进建议**:\n"
        report += "   - 重点关注排名前5的重要特征\n"
        report += "   - 对系数为负的特征进行优化改进\n"
        report += "   - 收集更多数据以提高模型准确性\n"

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"报告已保存到: {save_path}")

        return report


def demo_regression_analysis():
    """演示回归分析功能"""
    print("=== 供应链质量回归分析演示 ===\n")

    # 创建回归分析器
    regressor = SupplyChainRegression(model_type='linear')

    # 生成模拟数据
    print("1. 生成供应链质量评价模拟数据...")
    data = regressor.generate_supply_chain_data(n_samples=1000)
    print(f"   数据维度: {data.shape}")
    print(f"   特征列: {list(data.columns[:-1])}")
    print(f"   目标变量: {data.columns[-1]}\n")

    # 准备数据
    print("2. 准备回归分析数据...")
    feature_columns = [col for col in data.columns if col not in ['service_quality', 'enterprise_size', 'product_type', 'region']]
    categorical_columns = ['enterprise_size', 'product_type', 'region']

    X, y = regressor.prepare_data(
        data=data,
        target_column='service_quality',
        feature_columns=feature_columns + categorical_columns,
        categorical_columns=categorical_columns
    )
    print(f"   特征矩阵形状: {X.shape}")
    print(f"   目标向量长度: {len(y)}\n")

    # 训练模型
    print("3. 训练回归模型...")
    results = regressor.train(X, y, test_size=0.2, random_state=42)

    print(f"   训练集R²: {results['train_r2']:.4f}")
    print(f"   测试集R²: {results['test_r2']:.4f}")
    print(f"   测试集MSE: {results['test_mse']:.4f}")
    if 'cv_r2_mean' in results:
        print(f"   交叉验证R²: {results['cv_r2_mean']:.4f} ± {results['cv_r2_std']:.4f}")
    print()

    # 特征重要性
    print("4. 分析特征重要性...")
    importance_df = regressor.feature_importance()
    print("   前5个重要特征:")
    for _, row in importance_df.head(5).iterrows():
        print(f"   - {row['feature']}: {row['coefficient']:.4f}")
    print()

    # 预测演示
    print("5. 预测新样本...")
    X_new = X[:5]  # 使用前5个样本作为演示
    predictions = regressor.predict(X_new)
    print(f"   预测结果: {predictions}")
    print(f"   实际值: {y[:5]}")
    print()

    # 可视化结果
    print("6. 生成可视化结果...")
    regressor.visualize_results(save_path='output/regression_analysis_results.png')

    # 保存模型
    print("7. 保存模型...")
    os.makedirs('output/models', exist_ok=True)
    regressor.save_model('output/models/supply_chain_regression_model.pkl')

    # 生成报告
    print("8. 生成分析报告...")
    report = regressor.generate_report('output/regression_analysis_report.md')

    print("=== 演示完成 ===")
    return regressor, results


if __name__ == "__main__":
    demo_regression_analysis()
