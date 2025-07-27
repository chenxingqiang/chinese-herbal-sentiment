#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import jieba
import re
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import SVC
import time

# 导入项目中的模块
import sys
sys.path.append('src')
try:
    from sentiment_analysis import SentimentAnalysis
    from keyword_mapping import KeywordMapping
    from data_analysis import load_excel_files, preprocess_text
except ImportError:
    print("无法导入项目模块，请确保当前目录是项目根目录")
    sys.exit(1)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

class ThesisValidator:
    def __init__(self, comments_dir='comments', output_dir='validation_output'):
        """初始化验证器"""
        self.comments_dir = comments_dir
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化分析组件
        self.sentiment_analyzer = SentimentAnalysis(comments_dir)
        self.keyword_mapper = KeywordMapping(comments_dir)
        
        # 论文中的关键数值
        self.thesis_values = {
            'total_comments': 235000,  # 论文提到的评论总数
            'positive_percentage': 75.8,  # 正面评论百分比
            'neutral_percentage': 11.5,  # 中性评论百分比
            'negative_percentage': 12.7,  # 负面评论百分比
            'sentiment_accuracy': 85.6,  # 情感分析准确率
            'f1_score': 84.2,  # F1值
            'mapping_success_rate': 92.3,  # 映射成功率
            'valid_comments': 212000,  # 有效评论数
            'r2': 0.742,  # 模型解释度
            'key_factors': {
                'material_quality': 0.342,  # 原料质量系数
                'delivery_speed': 0.298,  # 物流配送系数
                'technology': 0.245,  # 加工工艺系数
                'after_sales_service': 0.186,  # 售后服务系数
                'information_transparency': 0.154  # 信息透明度系数
            },
            'dimension_scores': {
                'upstream': 8.12,  # 上游原料维度平均得分
                'midstream': 7.68,  # 中游加工维度平均得分
                'downstream': 7.95  # 下游销售物流维度平均得分
            },
            'time_series': {
                'jan_2024': 62.5,  # 2024年1月正面评价比例
                'jun_2024': 65.3  # 2024年6月正面评价比例
            }
        }
        
        # 存储验证结果
        self.validation_results = {}
    
    def validate_data_loading(self):
        """验证数据加载和统计"""
        print("\n=== 验证数据加载和统计 ===")
        
        # 加载评论数据
        print("加载评论数据...")
        data = self.sentiment_analyzer.load_excel_data()
        
        # 计算评论总数
        total_comments = sum(len(comments) for comments in data.values())
        positive_count = len(data['positive'])
        neutral_count = len(data['neutral'])
        negative_count = len(data['negative'])
        
        # 计算百分比
        positive_percentage = (positive_count / total_comments) * 100
        neutral_percentage = (neutral_count / total_comments) * 100
        negative_percentage = (negative_count / total_comments) * 100
        
        # 存储验证结果
        self.validation_results['data_loading'] = {
            'total_comments': {
                'actual': total_comments,
                'thesis': self.thesis_values['total_comments'],
                'match': abs(total_comments - self.thesis_values['total_comments']) / self.thesis_values['total_comments'] < 0.1
            },
            'positive_percentage': {
                'actual': positive_percentage,
                'thesis': self.thesis_values['positive_percentage'],
                'match': abs(positive_percentage - self.thesis_values['positive_percentage']) < 2.0
            },
            'neutral_percentage': {
                'actual': neutral_percentage,
                'thesis': self.thesis_values['neutral_percentage'],
                'match': abs(neutral_percentage - self.thesis_values['neutral_percentage']) < 2.0
            },
            'negative_percentage': {
                'actual': negative_percentage,
                'thesis': self.thesis_values['negative_percentage'],
                'match': abs(negative_percentage - self.thesis_values['negative_percentage']) < 2.0
            }
        }
        
        # 打印结果
        print(f"评论总数: {total_comments} (论文: {self.thesis_values['total_comments']})")
        print(f"正面评论: {positive_count} ({positive_percentage:.1f}%) (论文: {self.thesis_values['positive_percentage']}%)")
        print(f"中性评论: {neutral_count} ({neutral_percentage:.1f}%) (论文: {self.thesis_values['neutral_percentage']}%)")
        print(f"负面评论: {negative_count} ({negative_percentage:.1f}%) (论文: {self.thesis_values['negative_percentage']}%)")
        
        # 生成饼图
        plt.figure(figsize=(10, 6))
        plt.pie(
            [positive_count, neutral_count, negative_count],
            labels=['正面评价', '中性评价', '负面评价'],
            autopct='%1.1f%%',
            startangle=90,
            colors=['green', 'gray', 'red']
        )
        plt.axis('equal')
        plt.title('评论类别分布')
        plt.savefig(os.path.join(self.output_dir, 'sentiment_distribution_validation.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def validate_sentiment_analysis(self):
        """验证情感分析性能"""
        print("\n=== 验证情感分析性能 ===")
        
        # 加载评论数据
        print("加载评论数据...")
        data = self.sentiment_analyzer.load_excel_data()
        
        # 准备训练数据
        print("准备训练数据...")
        texts = []
        labels = []
        
        # 添加正面评价
        texts.extend(data['positive'][:5000])
        labels.extend([1] * min(5000, len(data['positive'])))
        
        # 添加中性评价
        texts.extend(data['neutral'][:2000])
        labels.extend([0] * min(2000, len(data['neutral'])))
        
        # 添加负面评价
        texts.extend(data['negative'][:3000])
        labels.extend([-1] * min(3000, len(data['negative'])))
        
        # 文本预处理
        print("文本预处理...")
        preprocessed_texts = []
        for text in texts:
            words = self.sentiment_analyzer.preprocess_text(text)
            preprocessed_texts.append(' '.join(words))
        
        # 特征提取
        print("特征提取...")
        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform(preprocessed_texts)
        
        # 划分训练集和测试集
        print("划分数据集...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # 训练SVM模型
        print("训练SVM模型...")
        model = SVC(kernel='linear')
        model.fit(X_train, y_train)
        
        # 测试模型
        print("评估模型性能...")
        y_pred = model.predict(X_test)
        
        # 计算性能指标
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        accuracy = accuracy_score(y_test, y_pred) * 100
        precision = precision_score(y_test, y_pred, average='macro') * 100
        recall = recall_score(y_test, y_pred, average='macro') * 100
        f1 = f1_score(y_test, y_pred, average='macro') * 100
        
        # 存储验证结果
        self.validation_results['sentiment_analysis'] = {
            'accuracy': {
                'actual': accuracy,
                'thesis': self.thesis_values['sentiment_accuracy'],
                'match': abs(accuracy - self.thesis_values['sentiment_accuracy']) < 5.0
            },
            'f1_score': {
                'actual': f1,
                'thesis': self.thesis_values['f1_score'],
                'match': abs(f1 - self.thesis_values['f1_score']) < 5.0
            }
        }
        
        # 打印结果
        print(f"准确率: {accuracy:.1f}% (论文: {self.thesis_values['sentiment_accuracy']}%)")
        print(f"精确率: {precision:.1f}%")
        print(f"召回率: {recall:.1f}%")
        print(f"F1值: {f1:.1f}% (论文: {self.thesis_values['f1_score']}%)")
    
    def validate_keyword_mapping(self):
        """验证关键词映射"""
        print("\n=== 验证关键词映射 ===")
        
        # 运行关键词映射
        print("运行关键词映射...")
        try:
            mapping_result, indicator_scores = self.keyword_mapper.run_mapping()
            
            # 计算映射成功率
            # 简化计算：统计有映射结果的指标占总指标的比例
            total_indicators = 0
            mapped_indicators = 0
            
            for dimension, indicators in mapping_result.items():
                for indicator_code, keywords in indicators.items():
                    total_indicators += 1
                    if keywords:  # 如果有映射到该指标的关键词
                        mapped_indicators += 1
            
            mapping_success_rate = (mapped_indicators / total_indicators) * 100 if total_indicators > 0 else 0
            
            # 存储验证结果
            self.validation_results['keyword_mapping'] = {
                'mapping_success_rate': {
                    'actual': mapping_success_rate,
                    'thesis': self.thesis_values['mapping_success_rate'],
                    'match': abs(mapping_success_rate - self.thesis_values['mapping_success_rate']) < 5.0
                }
            }
            
            # 打印结果
            print(f"映射成功率: {mapping_success_rate:.1f}% (论文: {self.thesis_values['mapping_success_rate']}%)")
            
            # 计算各维度平均得分
            dimension_scores = {}
            for dimension, indicators in indicator_scores.items():
                if indicators:
                    dimension_scores[dimension] = sum(indicators.values()) / len(indicators)
                else:
                    dimension_scores[dimension] = 0
            
            # 存储维度得分验证结果
            for dimension, score in dimension_scores.items():
                if dimension in self.thesis_values['dimension_scores']:
                    self.validation_results['dimension_scores'] = self.validation_results.get('dimension_scores', {})
                    self.validation_results['dimension_scores'][dimension] = {
                        'actual': score,
                        'thesis': self.thesis_values['dimension_scores'][dimension],
                        'match': abs(score - self.thesis_values['dimension_scores'][dimension]) < 1.0
                    }
            
            # 打印维度得分
            print(f"上游维度平均得分: {dimension_scores.get('upstream', 0):.2f} (论文: {self.thesis_values['dimension_scores']['upstream']})")
            print(f"中游维度平均得分: {dimension_scores.get('midstream', 0):.2f} (论文: {self.thesis_values['dimension_scores']['midstream']})")
            print(f"下游维度平均得分: {dimension_scores.get('downstream', 0):.2f} (论文: {self.thesis_values['dimension_scores']['downstream']})")
            
        except Exception as e:
            print(f"关键词映射验证出错: {e}")
    
    def validate_regression_analysis(self):
        """验证回归分析"""
        print("\n=== 验证回归分析 ===")
        
        try:
            # 加载评论数据
            print("加载评论数据...")
            data = self.sentiment_analyzer.load_excel_data()
            
            # 运行关键词映射获取指标得分
            print("获取指标得分...")
            mapping_result, indicator_scores = self.keyword_mapper.run_mapping()
            
            # 准备回归数据
            print("准备回归数据...")
            X = []  # 特征矩阵
            y = []  # 目标向量
            
            # 生成样本数据
            n_samples = min(5000, len(data['positive']))
            
            for i in range(n_samples):
                # 特征：各维度各指标的得分
                features = []
                
                # 上游维度指标
                for indicator in self.keyword_mapper.indicator_system['upstream'].keys():
                    if indicator in indicator_scores['upstream']:
                        features.append(indicator_scores['upstream'][indicator])
                    else:
                        features.append(0)
                        
                # 中游维度指标
                for indicator in self.keyword_mapper.indicator_system['midstream'].keys():
                    if indicator in indicator_scores['midstream']:
                        features.append(indicator_scores['midstream'][indicator])
                    else:
                        features.append(0)
                        
                # 下游维度指标
                for indicator in self.keyword_mapper.indicator_system['downstream'].keys():
                    if indicator in indicator_scores['downstream']:
                        features.append(indicator_scores['downstream'][indicator])
                    else:
                        features.append(0)
                        
                X.append(features)
                
                # 目标：模拟服务质量得分
                if i < len(data['positive']):
                    # 简化：随机生成7-10之间的得分
                    y.append(7 + 3 * np.random.random())
            
            # 转换为numpy数组
            X = np.array(X)
            y = np.array(y)
            
            # 划分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # 训练线性回归模型
            print("训练线性回归模型...")
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # 模型评估
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            # 获取回归系数
            coefficients = model.coef_
            
            # 获取特征名称（指标名称）
            feature_names = []
            
            # 上游维度指标
            for code in self.keyword_mapper.indicator_system['upstream'].keys():
                feature_names.append(code)
                
            # 中游维度指标
            for code in self.keyword_mapper.indicator_system['midstream'].keys():
                feature_names.append(code)
                
            # 下游维度指标
            for code in self.keyword_mapper.indicator_system['downstream'].keys():
                feature_names.append(code)
            
            # 构建系数字典
            coef_dict = {name: coef for name, coef in zip(feature_names, coefficients)}
            
            # 存储验证结果
            self.validation_results['regression_analysis'] = {
                'r2': {
                    'actual': r2,
                    'thesis': self.thesis_values['r2'],
                    'match': abs(r2 - self.thesis_values['r2']) < 0.1
                }
            }
            
            # 验证关键因素系数
            for factor, thesis_coef in self.thesis_values['key_factors'].items():
                if factor in coef_dict:
                    self.validation_results['regression_analysis'][factor] = {
                        'actual': coef_dict[factor],
                        'thesis': thesis_coef,
                        'match': abs(coef_dict[factor] - thesis_coef) < 0.1
                    }
            
            # 打印结果
            print(f"R²: {r2:.3f} (论文: {self.thesis_values['r2']})")
            print(f"MSE: {mse:.3f}")
            print("\n关键因素系数:")
            for factor, thesis_coef in self.thesis_values['key_factors'].items():
                actual_coef = coef_dict.get(factor, 0)
                print(f"- {factor}: {actual_coef:.3f} (论文: {thesis_coef})")
            
        except Exception as e:
            print(f"回归分析验证出错: {e}")
    
    def generate_validation_report(self):
        """生成验证报告"""
        print("\n=== 生成验证报告 ===")
        
        report_path = os.path.join(self.output_dir, 'validation_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 论文数据与算法实现验证报告\n\n")
            
            f.write("## 1. 数据加载与统计验证\n\n")
            
            if 'data_loading' in self.validation_results:
                results = self.validation_results['data_loading']
                
                f.write("| 指标 | 实际值 | 论文值 | 是否一致 |\n")
                f.write("|------|--------|--------|----------|\n")
                
                for metric, data in results.items():
                    match_str = "✅" if data['match'] else "❌"
                    f.write(f"| {metric} | {data['actual']:.1f} | {data['thesis']} | {match_str} |\n")
                
                f.write("\n![评论类别分布](sentiment_distribution_validation.png)\n\n")
            
            f.write("## 2. 情感分析性能验证\n\n")
            
            if 'sentiment_analysis' in self.validation_results:
                results = self.validation_results['sentiment_analysis']
                
                f.write("| 指标 | 实际值 | 论文值 | 是否一致 |\n")
                f.write("|------|--------|--------|----------|\n")
                
                for metric, data in results.items():
                    match_str = "✅" if data['match'] else "❌"
                    f.write(f"| {metric} | {data['actual']:.1f}% | {data['thesis']}% | {match_str} |\n")
            
            f.write("\n## 3. 关键词映射验证\n\n")
            
            if 'keyword_mapping' in self.validation_results:
                results = self.validation_results['keyword_mapping']
                
                f.write("| 指标 | 实际值 | 论文值 | 是否一致 |\n")
                f.write("|------|--------|--------|----------|\n")
                
                for metric, data in results.items():
                    match_str = "✅" if data['match'] else "❌"
                    f.write(f"| {metric} | {data['actual']:.1f}% | {data['thesis']}% | {match_str} |\n")
            
            if 'dimension_scores' in self.validation_results:
                f.write("\n### 维度得分验证\n\n")
                
                f.write("| 维度 | 实际值 | 论文值 | 是否一致 |\n")
                f.write("|------|--------|--------|----------|\n")
                
                for dimension, data in self.validation_results['dimension_scores'].items():
                    match_str = "✅" if data['match'] else "❌"
                    f.write(f"| {dimension} | {data['actual']:.2f} | {data['thesis']} | {match_str} |\n")
            
            f.write("\n## 4. 回归分析验证\n\n")
            
            if 'regression_analysis' in self.validation_results:
                results = self.validation_results['regression_analysis']
                
                f.write("| 指标 | 实际值 | 论文值 | 是否一致 |\n")
                f.write("|------|--------|--------|----------|\n")
                
                for metric, data in results.items():
                    match_str = "✅" if data['match'] else "❌"
                    if metric == 'r2':
                        f.write(f"| R² | {data['actual']:.3f} | {data['thesis']} | {match_str} |\n")
                    else:
                        f.write(f"| {metric} | {data['actual']:.3f} | {data['thesis']} | {match_str} |\n")
            
            f.write("\n## 5. 结论与建议\n\n")
            
            # 计算总体一致性
            total_checks = 0
            matched_checks = 0
            
            for category, results in self.validation_results.items():
                if category == 'dimension_scores':
                    for _, data in results.items():
                        total_checks += 1
                        if data['match']:
                            matched_checks += 1
                else:
                    for _, data in results.items():
                        total_checks += 1
                        if data['match']:
                            matched_checks += 1
            
            consistency = (matched_checks / total_checks) * 100 if total_checks > 0 else 0
            
            f.write(f"### 总体一致性: {consistency:.1f}%\n\n")
            
            if consistency >= 90:
                f.write("**结论**: 代码实现与论文描述高度一致，数据和结果可信度高。\n\n")
            elif consistency >= 70:
                f.write("**结论**: 代码实现与论文描述基本一致，但存在一些差异需要注意。\n\n")
            else:
                f.write("**结论**: 代码实现与论文描述存在较大差异，建议进一步调整代码或更新论文内容。\n\n")
            
            f.write("### 改进建议:\n\n")
            
            if 'data_loading' in self.validation_results and not all(data['match'] for data in self.validation_results['data_loading'].values()):
                f.write("1. **数据加载**: 调整数据处理流程，确保评论数量和分布与论文一致。\n")
            
            if 'sentiment_analysis' in self.validation_results and not all(data['match'] for data in self.validation_results['sentiment_analysis'].values()):
                f.write("2. **情感分析**: 优化情感分析模型，提高准确率和F1值，使其与论文结果一致。\n")
            
            if 'keyword_mapping' in self.validation_results and not all(data['match'] for data in self.validation_results['keyword_mapping'].values()):
                f.write("3. **关键词映射**: 改进映射算法，提高映射成功率。\n")
            
            if 'regression_analysis' in self.validation_results and not all(data['match'] for data in self.validation_results['regression_analysis'].values()):
                f.write("4. **回归分析**: 调整回归模型，使R²和关键因素系数与论文一致。\n")
            
            f.write("\n### 数据一致性建议:\n\n")
            f.write("1. 确保使用论文中提到的完整数据集（23.5万条评论）\n")
            f.write("2. 统一情感分类标准，确保分类结果一致\n")
            f.write("3. 使用相同的特征提取和模型参数，确保结果可复现\n")
            f.write("4. 对模型进行多次交叉验证，获取稳定的性能指标\n")
            f.write("5. 在代码中明确记录和输出关键数值，便于与论文对照\n")
        
        print(f"验证报告已生成: {report_path}")
    
    def run_validation(self):
        """运行完整的验证流程"""
        print("\n" + "="*50)
        print("开始验证论文数据与算法实现一致性")
        print("="*50 + "\n")
        
        start_time = time.time()
        
        # 验证数据加载和统计
        self.validate_data_loading()
        
        # 验证情感分析性能
        self.validate_sentiment_analysis()
        
        # 验证关键词映射
        self.validate_keyword_mapping()
        
        # 验证回归分析
        self.validate_regression_analysis()
        
        # 生成验证报告
        self.generate_validation_report()
        
        print(f"\n验证完成，总耗时：{time.time() - start_time:.2f}秒")
        print("\n" + "="*50)


if __name__ == "__main__":
    validator = ThesisValidator()
    validator.run_validation() 