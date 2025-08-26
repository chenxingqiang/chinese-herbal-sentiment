#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
时间序列分析模块
实现情感时间序列分析、趋势预测、季节性分析等功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
import warnings
try:
    from scipy import stats
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 尝试导入时间序列专用库
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    print("Warning: statsmodels not available. Advanced time series features will be disabled.")
    STATSMODELS_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    try:
        from fbprophet import Prophet
        PROPHET_AVAILABLE = True
    except ImportError:
        print("Warning: Prophet not available. Prophet forecasting will be disabled.")
        PROPHET_AVAILABLE = False


class TimeSeriesAnalyzer:
    """时间序列分析器"""

    def __init__(self):
        """初始化时间序列分析器"""
        self.data = None
        self.time_column = None
        self.value_column = None
        self.frequency = None
        self.trend_results = {}
        self.seasonal_results = {}
        self.forecast_results = {}

    def load_data(self,
                  data: pd.DataFrame,
                  time_column: str,
                  value_column: str,
                  parse_dates: bool = True) -> bool:
        """
        加载时间序列数据

        Args:
            data: 数据框
            time_column: 时间列名
            value_column: 数值列名
            parse_dates: 是否解析日期

        Returns:
            是否加载成功
        """
        try:
            self.data = data.copy()
            self.time_column = time_column
            self.value_column = value_column

            # 解析日期
            if parse_dates:
                self.data[time_column] = pd.to_datetime(self.data[time_column])

            # 按时间排序
            self.data = self.data.sort_values(time_column)

            # 设置时间为索引
            self.data.set_index(time_column, inplace=True)

            # 检测频率
            self._detect_frequency()

            print(f"Successfully loaded time series data: {len(self.data)} records")
            print(f"Time range: {self.data.index.min()} to {self.data.index.max()}")
            print(f"Detected frequency: {self.frequency}")

            return True

        except Exception as e:
            print(f"Failed to load time series data: {e}")
            return False

    def _detect_frequency(self):
        """自动检测时间序列频率"""
        if len(self.data) < 2:
            self.frequency = None
            return

        # 计算时间差
        time_diffs = self.data.index.to_series().diff().dropna()
        most_common_diff = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else None

        if most_common_diff:
            days = most_common_diff.days
            if days == 1:
                self.frequency = 'D'  # 日
            elif days == 7:
                self.frequency = 'W'  # 周
            elif 28 <= days <= 31:
                self.frequency = 'M'  # 月
            elif 90 <= days <= 92:
                self.frequency = 'Q'  # 季度
            elif 365 <= days <= 366:
                self.frequency = 'Y'  # 年
            else:
                self.frequency = f'{days}D'  # 自定义天数
        else:
            self.frequency = None

    def generate_sample_data(self,
                           start_date: str = '2022-01-01',
                           periods: int = 365,
                           frequency: str = 'D') -> pd.DataFrame:
        """
        生成示例时间序列数据

        Args:
            start_date: 开始日期
            periods: 周期数
            frequency: 频率

        Returns:
            示例数据DataFrame
        """
        # 生成日期范围
        dates = pd.date_range(start=start_date, periods=periods, freq=frequency)

        # 生成基础趋势
        trend = np.linspace(7.0, 8.0, periods)

        # 添加季节性（年度周期）
        seasonal = 0.5 * np.sin(2 * np.pi * np.arange(periods) / (365/periods if frequency == 'D' else 12))

        # 添加周期性波动（周或月）
        if frequency == 'D':
            weekly_cycle = 0.3 * np.sin(2 * np.pi * np.arange(periods) / 7)
        else:
            weekly_cycle = 0

        # 添加随机噪声
        noise = np.random.normal(0, 0.2, periods)

        # 添加一些异常值和事件
        anomalies = np.zeros(periods)

        # 模拟节假日效应（正面）
        holiday_indices = [59, 120, 181, 242, 303, 364]  # 假设的节假日
        for idx in holiday_indices:
            if idx < periods:
                anomalies[idx-2:idx+3] += np.random.uniform(0.5, 1.0, min(5, periods-idx+2))

        # 模拟负面事件
        negative_events = [150, 200, 280]
        for idx in negative_events:
            if idx < periods:
                anomalies[idx-1:idx+2] -= np.random.uniform(0.3, 0.8, min(3, periods-idx+1))

        # 合成最终的情感得分
        sentiment_score = trend + seasonal + weekly_cycle + noise + anomalies

        # 确保得分在合理范围内
        sentiment_score = np.clip(sentiment_score, 1.0, 10.0)

        # 生成其他相关指标
        review_count = np.random.poisson(50, periods) + np.random.randint(10, 100, periods)

        # 企业类型分布
        enterprise_types = np.random.choice(['大型', '中型', '小型'], periods, p=[0.3, 0.4, 0.3])

        # 产品类别
        product_types = np.random.choice(['饮片', '配方颗粒', '中成药'], periods, p=[0.4, 0.3, 0.3])

        # 创建数据框
        data = pd.DataFrame({
            'date': dates,
            'sentiment_score': sentiment_score,
            'review_count': review_count,
            'enterprise_type': enterprise_types,
            'product_type': product_types,
            'positive_ratio': np.clip(sentiment_score / 10.0 + np.random.normal(0, 0.1, periods), 0, 1),
            'negative_ratio': np.clip((10 - sentiment_score) / 10.0 + np.random.normal(0, 0.1, periods), 0, 1)
        })

        # 确保比例和为1
        total_ratio = data['positive_ratio'] + data['negative_ratio']
        data['positive_ratio'] /= total_ratio
        data['negative_ratio'] /= total_ratio
        data['neutral_ratio'] = 1 - data['positive_ratio'] - data['negative_ratio']

        return data

    def trend_analysis(self,
                      method: str = 'linear',
                      window: Optional[int] = None) -> Dict[str, Any]:
        """
        趋势分析

        Args:
            method: 趋势分析方法 ('linear', 'polynomial', 'moving_average')
            window: 移动平均窗口大小

        Returns:
            趋势分析结果
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")

        results = {
            'method': method,
            'trend_direction': None,
            'trend_strength': None,
            'trend_significance': None,
            'trend_line': None,
            'statistics': {}
        }

        values = self.data[self.value_column].values
        time_numeric = np.arange(len(values))

        try:
            if method == 'linear':
                # 线性回归趋势
                model = LinearRegression()
                model.fit(time_numeric.reshape(-1, 1), values)
                trend_line = model.predict(time_numeric.reshape(-1, 1))

                # 趋势强度和显著性
                if SCIPY_AVAILABLE:
                    correlation, p_value = stats.pearsonr(time_numeric, values)
                else:
                    correlation = np.corrcoef(time_numeric, values)[0, 1]
                    p_value = None

                results['trend_line'] = trend_line
                results['trend_direction'] = 'increasing' if model.coef_[0] > 0 else 'decreasing'
                results['trend_strength'] = abs(correlation)
                results['trend_significance'] = p_value
                results['slope'] = model.coef_[0]
                results['intercept'] = model.intercept_

                results['statistics'] = {
                    'correlation': correlation,
                    'p_value': p_value,
                    'slope': model.coef_[0],
                    'r_squared': model.score(time_numeric.reshape(-1, 1), values)
                }

            elif method == 'polynomial':
                # 多项式趋势（二次）
                coeffs = np.polyfit(time_numeric, values, 2)
                trend_line = np.polyval(coeffs, time_numeric)

                # 计算拟合优度
                ss_res = np.sum((values - trend_line) ** 2)
                ss_tot = np.sum((values - np.mean(values)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)

                results['trend_line'] = trend_line
                results['trend_direction'] = 'complex'
                results['trend_strength'] = r_squared
                results['coefficients'] = coeffs.tolist()

                results['statistics'] = {
                    'r_squared': r_squared,
                    'coefficients': coeffs.tolist()
                }

            elif method == 'moving_average':
                # 移动平均趋势
                if window is None:
                    window = max(7, len(values) // 20)  # 默认窗口大小

                trend_line = self.data[self.value_column].rolling(window=window, center=True).mean()

                # 计算趋势方向
                first_half = trend_line[:len(trend_line)//2].mean()
                second_half = trend_line[len(trend_line)//2:].mean()

                results['trend_line'] = trend_line.values
                results['trend_direction'] = 'increasing' if second_half > first_half else 'decreasing'
                results['trend_strength'] = abs(second_half - first_half) / first_half
                results['window_size'] = window

                results['statistics'] = {
                    'first_half_mean': first_half,
                    'second_half_mean': second_half,
                    'relative_change': (second_half - first_half) / first_half
                }

            # 保存结果
            self.trend_results = results

            print(f"Trend analysis completed using {method} method")
            print(f"Trend direction: {results['trend_direction']}")
            print(f"Trend strength: {results['trend_strength']:.4f}")

            return results

        except Exception as e:
            print(f"Trend analysis failed: {e}")
            return {'error': str(e)}

    def seasonal_analysis(self, model: str = 'additive') -> Dict[str, Any]:
        """
        季节性分析

        Args:
            model: 分解模型 ('additive', 'multiplicative')

        Returns:
            季节性分析结果
        """
        if not STATSMODELS_AVAILABLE:
            return self._simple_seasonal_analysis()

        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")

        try:
            # 确保有足够的数据进行季节性分解
            if len(self.data) < 24:  # 至少需要2个完整周期
                return self._simple_seasonal_analysis()

            # 使用statsmodels进行季节性分解
            ts_data = self.data[self.value_column]

            # 自动确定季节性周期
            if self.frequency == 'D':
                period = 7  # 周季节性
            elif self.frequency == 'W':
                period = 4  # 月季节性
            elif self.frequency == 'M':
                period = 12  # 年季节性
            else:
                period = min(12, len(ts_data) // 3)  # 默认周期

            # 进行季节性分解
            decomposition = seasonal_decompose(ts_data, model=model, period=period)

            results = {
                'model': model,
                'period': period,
                'trend': decomposition.trend.dropna().values,
                'seasonal': decomposition.seasonal.values,
                'residual': decomposition.resid.dropna().values,
                'seasonal_strength': None,
                'trend_strength': None,
                'seasonality_test': None
            }

            # 计算季节性强度
            var_residual = np.var(decomposition.resid.dropna())
            var_seasonal_residual = np.var(decomposition.seasonal + decomposition.resid)
            results['seasonal_strength'] = max(0, 1 - var_residual / var_seasonal_residual)

            # 计算趋势强度
            var_detrend = np.var(decomposition.seasonal + decomposition.resid)
            var_original = np.var(ts_data)
            results['trend_strength'] = max(0, 1 - var_detrend / var_original)

            # 季节性显著性检验
            seasonal_values = decomposition.seasonal.values
            seasonal_unique = np.unique(seasonal_values)
            if len(seasonal_unique) > 1 and SCIPY_AVAILABLE:
                try:
                    f_stat, p_value = stats.f_oneway(*[seasonal_values[i::period] for i in range(period)])
                    results['seasonality_test'] = {
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'is_significant': p_value < 0.05
                    }
                except:
                    results['seasonality_test'] = None
            else:
                results['seasonality_test'] = None

            self.seasonal_results = results

            print(f"Seasonal analysis completed")
            print(f"Seasonal strength: {results['seasonal_strength']:.4f}")
            print(f"Trend strength: {results['trend_strength']:.4f}")

            return results

        except Exception as e:
            print(f"Advanced seasonal analysis failed, falling back to simple method: {e}")
            return self._simple_seasonal_analysis()

    def _simple_seasonal_analysis(self) -> Dict[str, Any]:
        """简单的季节性分析（不依赖statsmodels）"""
        if self.data is None:
            raise ValueError("No data loaded")

        values = self.data[self.value_column].values

        # 检测周期性模式
        periods_to_test = [7, 30, 90, 365]  # 周、月、季、年
        best_period = 7
        best_autocorr = 0

        for period in periods_to_test:
            if len(values) > period * 2:
                # 计算自相关
                if len(values) > period:
                    autocorr = np.corrcoef(values[:-period], values[period:])[0, 1]
                    if not np.isnan(autocorr) and abs(autocorr) > abs(best_autocorr):
                        best_autocorr = autocorr
                        best_period = period

        # 计算简单的季节性模式
        seasonal_pattern = []
        for i in range(best_period):
            period_values = values[i::best_period]
            if len(period_values) > 0:
                seasonal_pattern.append(np.mean(period_values))

        # 去除趋势的简单方法
        trend = np.convolve(values, np.ones(min(7, len(values)//4))/min(7, len(values)//4), mode='same')

        results = {
            'model': 'simple',
            'period': best_period,
            'seasonal_pattern': seasonal_pattern,
            'trend': trend,
            'best_autocorr': best_autocorr,
            'seasonal_strength': abs(best_autocorr),
            'residual': values - trend
        }

        return results

    def forecast(self,
                periods: int = 30,
                method: str = 'auto',
                confidence_interval: float = 0.95) -> Dict[str, Any]:
        """
        时间序列预测

        Args:
            periods: 预测期数
            method: 预测方法 ('auto', 'linear', 'exponential', 'arima', 'prophet')
            confidence_interval: 置信区间

        Returns:
            预测结果
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")

        values = self.data[self.value_column].values

        if method == 'auto':
            # 自动选择最佳方法
            if PROPHET_AVAILABLE and len(values) > 50:
                method = 'prophet'
            elif STATSMODELS_AVAILABLE and len(values) > 30:
                method = 'exponential'
            else:
                method = 'linear'

        try:
            if method == 'linear':
                return self._linear_forecast(periods, confidence_interval)
            elif method == 'exponential':
                return self._exponential_forecast(periods, confidence_interval)
            elif method == 'arima' and STATSMODELS_AVAILABLE:
                return self._arima_forecast(periods, confidence_interval)
            elif method == 'prophet' and PROPHET_AVAILABLE:
                return self._prophet_forecast(periods, confidence_interval)
            else:
                print(f"Method {method} not available, falling back to linear forecast")
                return self._linear_forecast(periods, confidence_interval)

        except Exception as e:
            print(f"Forecast method {method} failed: {e}")
            print("Falling back to linear forecast")
            return self._linear_forecast(periods, confidence_interval)

    def _linear_forecast(self, periods: int, confidence_interval: float) -> Dict[str, Any]:
        """线性预测"""
        values = self.data[self.value_column].values
        time_numeric = np.arange(len(values))

        # 拟合线性模型
        model = LinearRegression()
        model.fit(time_numeric.reshape(-1, 1), values)

        # 预测未来值
        future_time = np.arange(len(values), len(values) + periods)
        predictions = model.predict(future_time.reshape(-1, 1))

        # 计算预测误差
        train_pred = model.predict(time_numeric.reshape(-1, 1))
        mse = mean_squared_error(values, train_pred)
        std_error = np.sqrt(mse)

        # 置信区间
        alpha = 1 - confidence_interval
        if SCIPY_AVAILABLE:
            z_score = stats.norm.ppf(1 - alpha/2)
        else:
            z_score = 1.96  # 95%置信区间的近似值

        lower_bound = predictions - z_score * std_error
        upper_bound = predictions + z_score * std_error

        # 生成未来日期
        last_date = self.data.index[-1]
        if self.frequency:
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                       periods=periods, freq=self.frequency)
        else:
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                       periods=periods, freq='D')

        results = {
            'method': 'linear',
            'periods': periods,
            'predictions': predictions.tolist(),
            'lower_bound': lower_bound.tolist(),
            'upper_bound': upper_bound.tolist(),
            'future_dates': future_dates.tolist(),
            'model_performance': {
                'mse': mse,
                'mae': mean_absolute_error(values, train_pred),
                'r_squared': model.score(time_numeric.reshape(-1, 1), values)
            }
        }

        return results

    def _exponential_forecast(self, periods: int, confidence_interval: float) -> Dict[str, Any]:
        """指数平滑预测"""
        if not STATSMODELS_AVAILABLE:
            return self._linear_forecast(periods, confidence_interval)

        ts_data = self.data[self.value_column]

        # 指数平滑模型
        model = ExponentialSmoothing(ts_data, trend='add', seasonal=None)
        fitted_model = model.fit()

        # 预测
        forecast = fitted_model.forecast(periods)

        # 置信区间（简化版本）
        residuals = fitted_model.resid
        std_error = np.std(residuals)
        alpha = 1 - confidence_interval
        if SCIPY_AVAILABLE:
            z_score = stats.norm.ppf(1 - alpha/2)
        else:
            z_score = 1.96  # 95%置信区间的近似值

        lower_bound = forecast - z_score * std_error
        upper_bound = forecast + z_score * std_error

        # 生成未来日期
        last_date = self.data.index[-1]
        if self.frequency:
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                       periods=periods, freq=self.frequency)
        else:
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                       periods=periods, freq='D')

        results = {
            'method': 'exponential_smoothing',
            'periods': periods,
            'predictions': forecast.tolist(),
            'lower_bound': lower_bound.tolist(),
            'upper_bound': upper_bound.tolist(),
            'future_dates': future_dates.tolist(),
            'model_performance': {
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'mse': np.mean(residuals**2)
            }
        }

        return results

    def _prophet_forecast(self, periods: int, confidence_interval: float) -> Dict[str, Any]:
        """Prophet预测"""
        if not PROPHET_AVAILABLE:
            return self._linear_forecast(periods, confidence_interval)

        # 准备Prophet数据格式
        df = self.data.reset_index()
        df = df.rename(columns={self.time_column: 'ds', self.value_column: 'y'})

        # 创建和训练Prophet模型
        model = Prophet(interval_width=confidence_interval)
        model.fit(df)

        # 创建未来日期框架
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)

        # 提取预测结果
        future_forecast = forecast.tail(periods)

        results = {
            'method': 'prophet',
            'periods': periods,
            'predictions': future_forecast['yhat'].tolist(),
            'lower_bound': future_forecast['yhat_lower'].tolist(),
            'upper_bound': future_forecast['yhat_upper'].tolist(),
            'future_dates': future_forecast['ds'].tolist(),
            'trend': future_forecast['trend'].tolist(),
            'seasonal': future_forecast.get('seasonal', []),
            'model_performance': {
                'components': ['trend', 'seasonal']
            }
        }

        return results

    def detect_anomalies(self, method: str = 'iqr', sensitivity: float = 1.5) -> Dict[str, Any]:
        """
        异常值检测

        Args:
            method: 检测方法 ('iqr', 'zscore', 'isolation')
            sensitivity: 敏感度参数

        Returns:
            异常值检测结果
        """
        if self.data is None:
            raise ValueError("No data loaded")

        values = self.data[self.value_column].values
        anomaly_indices = []
        anomaly_scores = []

        if method == 'iqr':
            # 四分位距方法
            Q1 = np.percentile(values, 25)
            Q3 = np.percentile(values, 75)
            IQR = Q3 - Q1

            lower_bound = Q1 - sensitivity * IQR
            upper_bound = Q3 + sensitivity * IQR

            for i, value in enumerate(values):
                if value < lower_bound or value > upper_bound:
                    anomaly_indices.append(i)
                    # 计算异常程度
                    if value < lower_bound:
                        anomaly_scores.append((lower_bound - value) / IQR)
                    else:
                        anomaly_scores.append((value - upper_bound) / IQR)

        elif method == 'zscore':
            # Z-score方法
            mean_val = np.mean(values)
            std_val = np.std(values)

            z_scores = np.abs((values - mean_val) / std_val)
            threshold = sensitivity * 2  # 默认阈值

            for i, z_score in enumerate(z_scores):
                if z_score > threshold:
                    anomaly_indices.append(i)
                    anomaly_scores.append(z_score)

        # 创建结果
        anomaly_dates = self.data.index[anomaly_indices].tolist()
        anomaly_values = values[anomaly_indices].tolist()

        results = {
            'method': method,
            'sensitivity': sensitivity,
            'anomaly_count': len(anomaly_indices),
            'anomaly_indices': anomaly_indices,
            'anomaly_dates': anomaly_dates,
            'anomaly_values': anomaly_values,
            'anomaly_scores': anomaly_scores,
            'anomaly_percentage': len(anomaly_indices) / len(values) * 100
        }

        print(f"Detected {len(anomaly_indices)} anomalies ({results['anomaly_percentage']:.2f}%)")

        return results

    def visualize_analysis(self,
                          include_trend: bool = True,
                          include_seasonal: bool = True,
                          include_forecast: bool = True,
                          include_anomalies: bool = True,
                          save_path: str = None):
        """
        可视化时间序列分析结果

        Args:
            include_trend: 是否包含趋势分析
            include_seasonal: 是否包含季节性分析
            include_forecast: 是否包含预测结果
            include_anomalies: 是否包含异常值检测
            save_path: 保存路径
        """
        if self.data is None:
            raise ValueError("No data loaded")

        # 计算子图数量
        n_plots = 1  # 原始数据
        if include_trend:
            n_plots += 1
        if include_seasonal and self.seasonal_results:
            n_plots += 1
        if include_forecast and self.forecast_results:
            n_plots += 1

        fig, axes = plt.subplots(n_plots, 1, figsize=(15, 5 * n_plots))

        if n_plots == 1:
            axes = [axes]

        plot_idx = 0

        # 1. 原始时间序列
        axes[plot_idx].plot(self.data.index, self.data[self.value_column],
                           'b-', linewidth=1, label='原始数据')

        # 添加异常值
        if include_anomalies:
            anomalies = self.detect_anomalies()
            if anomalies['anomaly_count'] > 0:
                anomaly_dates = [self.data.index[i] for i in anomalies['anomaly_indices']]
                anomaly_values = anomalies['anomaly_values']
                axes[plot_idx].scatter(anomaly_dates, anomaly_values,
                                     color='red', s=50, alpha=0.7, label='异常值')

        axes[plot_idx].set_title('时间序列数据')
        axes[plot_idx].set_ylabel(self.value_column)
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1

        # 2. 趋势分析
        if include_trend and self.trend_results:
            axes[plot_idx].plot(self.data.index, self.data[self.value_column],
                               'b-', alpha=0.5, label='原始数据')

            if 'trend_line' in self.trend_results:
                axes[plot_idx].plot(self.data.index, self.trend_results['trend_line'],
                                   'r-', linewidth=2, label='趋势线')

            direction = self.trend_results.get('trend_direction', 'unknown')
            strength = self.trend_results.get('trend_strength', 0)
            axes[plot_idx].set_title(f'趋势分析 (方向: {direction}, 强度: {strength:.3f})')
            axes[plot_idx].set_ylabel(self.value_column)
            axes[plot_idx].legend()
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1

        # 3. 季节性分析
        if include_seasonal and self.seasonal_results:
            if 'seasonal' in self.seasonal_results:
                seasonal_data = self.seasonal_results['seasonal']
                axes[plot_idx].plot(seasonal_data, 'g-', linewidth=1)
                axes[plot_idx].set_title('季节性成分')
                axes[plot_idx].set_ylabel('季节性值')
                axes[plot_idx].grid(True, alpha=0.3)
                plot_idx += 1

        # 4. 预测结果
        if include_forecast and self.forecast_results:
            # 历史数据
            axes[plot_idx].plot(self.data.index, self.data[self.value_column],
                               'b-', label='历史数据')

            # 预测数据
            future_dates = pd.to_datetime(self.forecast_results['future_dates'])
            predictions = self.forecast_results['predictions']

            axes[plot_idx].plot(future_dates, predictions,
                               'r-', linewidth=2, label='预测值')

            # 置信区间
            if 'lower_bound' in self.forecast_results and 'upper_bound' in self.forecast_results:
                axes[plot_idx].fill_between(future_dates,
                                           self.forecast_results['lower_bound'],
                                           self.forecast_results['upper_bound'],
                                           alpha=0.3, color='red', label='置信区间')

            axes[plot_idx].set_title(f'预测结果 ({self.forecast_results["method"]})')
            axes[plot_idx].set_ylabel(self.value_column)
            axes[plot_idx].legend()
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"时间序列分析图表已保存到: {save_path}")

        plt.show()

    def generate_report(self, save_path: str = None) -> str:
        """
        生成时间序列分析报告

        Args:
            save_path: 报告保存路径

        Returns:
            报告内容
        """
        if self.data is None:
            raise ValueError("No data loaded")

        report = "# 时间序列分析报告\n\n"

        # 基本信息
        report += "## 1. 数据基本信息\n\n"
        report += f"- **数据量**: {len(self.data)} 个观测值\n"
        report += f"- **时间范围**: {self.data.index.min()} 到 {self.data.index.max()}\n"
        report += f"- **检测频率**: {self.frequency}\n"
        report += f"- **数值列**: {self.value_column}\n"

        # 描述性统计
        stats_desc = self.data[self.value_column].describe()
        report += f"- **均值**: {stats_desc['mean']:.4f}\n"
        report += f"- **标准差**: {stats_desc['std']:.4f}\n"
        report += f"- **最小值**: {stats_desc['min']:.4f}\n"
        report += f"- **最大值**: {stats_desc['max']:.4f}\n\n"

        # 趋势分析结果
        if self.trend_results:
            report += "## 2. 趋势分析\n\n"
            trend = self.trend_results
            report += f"- **分析方法**: {trend['method']}\n"
            report += f"- **趋势方向**: {trend['trend_direction']}\n"
            report += f"- **趋势强度**: {trend['trend_strength']:.4f}\n"

            if 'trend_significance' in trend:
                report += f"- **显著性**: {'显著' if trend['trend_significance'] < 0.05 else '不显著'} (p={trend['trend_significance']:.4f})\n"

            if 'slope' in trend:
                report += f"- **变化率**: {trend['slope']:.6f} 单位/期\n"

            report += "\n"

        # 季节性分析结果
        if self.seasonal_results:
            report += "## 3. 季节性分析\n\n"
            seasonal = self.seasonal_results
            report += f"- **分解模型**: {seasonal.get('model', 'simple')}\n"
            report += f"- **检测周期**: {seasonal.get('period', 'N/A')}\n"

            if 'seasonal_strength' in seasonal:
                report += f"- **季节性强度**: {seasonal['seasonal_strength']:.4f}\n"

            if 'trend_strength' in seasonal:
                report += f"- **趋势强度**: {seasonal['trend_strength']:.4f}\n"

            if 'seasonality_test' in seasonal and seasonal['seasonality_test']:
                test = seasonal['seasonality_test']
                report += f"- **季节性显著性**: {'显著' if test['is_significant'] else '不显著'} (p={test['p_value']:.4f})\n"

            report += "\n"

        # 预测结果
        if self.forecast_results:
            report += "## 4. 预测分析\n\n"
            forecast = self.forecast_results
            report += f"- **预测方法**: {forecast['method']}\n"
            report += f"- **预测期数**: {forecast['periods']}\n"

            if 'model_performance' in forecast:
                perf = forecast['model_performance']
                if 'r_squared' in perf:
                    report += f"- **模型拟合度(R²)**: {perf['r_squared']:.4f}\n"
                if 'mse' in perf:
                    report += f"- **均方误差**: {perf['mse']:.4f}\n"
                if 'mae' in perf:
                    report += f"- **平均绝对误差**: {perf['mae']:.4f}\n"

            # 预测摘要
            predictions = forecast['predictions']
            report += f"- **预测均值**: {np.mean(predictions):.4f}\n"
            report += f"- **预测范围**: {min(predictions):.4f} - {max(predictions):.4f}\n"

            report += "\n"

        # 异常值检测
        anomalies = self.detect_anomalies()
        report += "## 5. 异常值检测\n\n"
        report += f"- **检测方法**: {anomalies['method']}\n"
        report += f"- **异常值数量**: {anomalies['anomaly_count']}\n"
        report += f"- **异常值比例**: {anomalies['anomaly_percentage']:.2f}%\n"

        if anomalies['anomaly_count'] > 0:
            report += "\n**主要异常值**:\n\n"
            for i, (date, value, score) in enumerate(zip(
                anomalies['anomaly_dates'][:5],
                anomalies['anomaly_values'][:5],
                anomalies['anomaly_scores'][:5]
            )):
                report += f"- {date}: {value:.4f} (异常程度: {score:.2f})\n"

        report += "\n"

        # 结论和建议
        report += "## 6. 结论和建议\n\n"

        # 趋势结论
        if self.trend_results:
            direction = self.trend_results['trend_direction']
            strength = self.trend_results['trend_strength']

            if direction == 'increasing':
                report += "1. **趋势分析**: 数据呈现上升趋势"
            elif direction == 'decreasing':
                report += "1. **趋势分析**: 数据呈现下降趋势"
            else:
                report += "1. **趋势分析**: 数据趋势不明显"

            if strength > 0.7:
                report += "，趋势非常明显。\n"
            elif strength > 0.5:
                report += "，趋势较为明显。\n"
            else:
                report += "，但趋势强度较弱。\n"

        # 季节性结论
        if self.seasonal_results and 'seasonal_strength' in self.seasonal_results:
            seasonal_strength = self.seasonal_results['seasonal_strength']
            if seasonal_strength > 0.3:
                report += "2. **季节性分析**: 数据存在明显的季节性模式，建议在预测时考虑季节性因素。\n"
            else:
                report += "2. **季节性分析**: 数据的季节性模式不明显。\n"

        # 异常值结论
        if anomalies['anomaly_percentage'] > 5:
            report += "3. **异常值**: 数据中异常值较多，建议进一步调查异常原因。\n"
        elif anomalies['anomaly_percentage'] > 1:
            report += "3. **异常值**: 数据中存在少量异常值，属于正常范围。\n"
        else:
            report += "3. **异常值**: 数据质量良好，异常值很少。\n"

        # 预测建议
        if self.forecast_results:
            method = self.forecast_results['method']
            report += f"4. **预测建议**: 当前使用{method}方法进行预测，"

            if 'model_performance' in self.forecast_results:
                perf = self.forecast_results['model_performance']
                if 'r_squared' in perf and perf['r_squared'] > 0.8:
                    report += "模型拟合度较好。\n"
                elif 'r_squared' in perf and perf['r_squared'] > 0.6:
                    report += "模型拟合度一般，建议尝试其他方法。\n"
                else:
                    report += "模型拟合度较差，建议收集更多数据或尝试其他方法。\n"

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"时间序列分析报告已保存到: {save_path}")

        return report


def demo_time_series_analysis():
    """演示时间序列分析功能"""
    print("=== 时间序列分析演示 ===\n")

    # 创建时间序列分析器
    analyzer = TimeSeriesAnalyzer()

    # 生成示例数据
    print("1. 生成示例时间序列数据...")
    sample_data = analyzer.generate_sample_data(
        start_date='2022-01-01',
        periods=365,
        frequency='D'
    )
    print(f"   数据维度: {sample_data.shape}")
    print(f"   时间范围: {sample_data['date'].min()} 到 {sample_data['date'].max()}")
    print()

    # 加载数据
    print("2. 加载时间序列数据...")
    success = analyzer.load_data(sample_data, 'date', 'sentiment_score')
    if not success:
        print("   数据加载失败")
        return
    print()

    # 趋势分析
    print("3. 进行趋势分析...")
    trend_results = analyzer.trend_analysis(method='linear')
    print(f"   趋势方向: {trend_results['trend_direction']}")
    print(f"   趋势强度: {trend_results['trend_strength']:.4f}")
    if 'slope' in trend_results:
        print(f"   变化率: {trend_results['slope']:.6f} 单位/天")
    print()

    # 季节性分析
    print("4. 进行季节性分析...")
    seasonal_results = analyzer.seasonal_analysis()
    if 'seasonal_strength' in seasonal_results:
        print(f"   季节性强度: {seasonal_results['seasonal_strength']:.4f}")
    print(f"   检测周期: {seasonal_results.get('period', 'N/A')}")
    print()

    # 预测分析
    print("5. 进行预测分析...")
    forecast_results = analyzer.forecast(periods=30, method='auto')
    analyzer.forecast_results = forecast_results  # 保存结果用于可视化

    print(f"   预测方法: {forecast_results['method']}")
    print(f"   预测期数: {forecast_results['periods']}")
    predictions = forecast_results['predictions']
    print(f"   预测均值: {np.mean(predictions):.4f}")
    print(f"   预测范围: {min(predictions):.4f} - {max(predictions):.4f}")
    print()

    # 异常值检测
    print("6. 进行异常值检测...")
    anomalies = analyzer.detect_anomalies(method='iqr')
    print(f"   检测到 {anomalies['anomaly_count']} 个异常值")
    print(f"   异常值比例: {anomalies['anomaly_percentage']:.2f}%")

    if anomalies['anomaly_count'] > 0:
        print("   主要异常值:")
        for i, (date, value) in enumerate(zip(
            anomalies['anomaly_dates'][:3],
            anomalies['anomaly_values'][:3]
        )):
            print(f"     - {date}: {value:.4f}")
    print()

    # 可视化结果
    print("7. 生成可视化图表...")
    try:
        os.makedirs('output', exist_ok=True)
        analyzer.visualize_analysis(
            include_trend=True,
            include_seasonal=True,
            include_forecast=True,
            include_anomalies=True,
            save_path='output/time_series_analysis.png'
        )
    except Exception as e:
        print(f"   可视化失败: {e}")

    # 生成报告
    print("8. 生成分析报告...")
    try:
        report = analyzer.generate_report('output/time_series_analysis_report.md')
        print("   报告已生成")
    except Exception as e:
        print(f"   报告生成失败: {e}")

    print("\n=== 演示完成 ===")
    return analyzer, sample_data


if __name__ == "__main__":
    demo_time_series_analysis()
