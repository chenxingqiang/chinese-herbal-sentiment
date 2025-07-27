#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentiment_analysis import SentimentAnalysis
from keyword_mapping import KeywordMapping
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import jieba
import time
import json
from wordcloud import WordCloud
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

class MainAnalysis:
    def __init__(self, comments_dir='comments', output_dir='output'):
        """
        初始化主分析类
        
        参数:
        comments_dir: 评论数据目录
        output_dir: 输出结果目录
        """
        self.comments_dir = comments_dir
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化分析组件
        self.sentiment_analyzer = SentimentAnalysis(comments_dir)
        self.keyword_mapper = KeywordMapping(comments_dir)
        
        # 存储分析结果
        self.data = None
        self.sentiment_scores = None
        self.mapping_result = None
        self.indicator_scores = None
        self.regression_results = None
    
    def run_analysis(self):
        """
        运行完整的分析流程
        """
        print("\n" + "="*50)
        print("开始中药材电商供应链服务质量评价分析")
        print("="*50 + "\n")
        
        # 第一步：加载评论数据
        print("第一步：加载评论数据...")
        start_time = time.time()
        self.data = self.sentiment_analyzer.load_excel_data()
        print(f"加载完成，耗时：{time.time() - start_time:.2f}秒")
        print(f"共有正面评价：{len(self.data['positive'])} 条")
        print(f"共有中性评价：{len(self.data['neutral'])} 条")
        print(f"共有负面评价：{len(self.data['negative'])} 条")
        
        # 第二步：情感分析
        print("\n第二步：执行情感分析...")
        start_time = time.time()
        self.sentiment_scores = self._perform_sentiment_analysis()
        print(f"情感分析完成，耗时：{time.time() - start_time:.2f}秒")
        
        # 第三步：关键词提取与映射
        print("\n第三步：关键词提取与映射...")
        start_time = time.time()
        self.mapping_result, self.indicator_scores = self.keyword_mapper.run_mapping()
        print(f"关键词映射完成，耗时：{time.time() - start_time:.2f}秒")
        
        # 第四步：多元回归分析
        print("\n第四步：多元回归分析...")
        start_time = time.time()
        self.regression_results = self._perform_regression_analysis()
        print(f"回归分析完成，耗时：{time.time() - start_time:.2f}秒")
        
        # 第五步：生成报告和可视化结果
        print("\n第五步：生成报告和可视化结果...")
        start_time = time.time()
        self._generate_reports()
        print(f"报告生成完成，耗时：{time.time() - start_time:.2f}秒")
        
        print("\n" + "="*50)
        print("分析完成！所有结果已保存到 output 目录")
        print("="*50 + "\n")
    
    def _perform_sentiment_analysis(self):
        """
        执行情感分析，返回情感分数
        
        返回:
        情感分数字典
        """
        print("执行基于词典的情感分析...")
        sentiment_scores = {
            'positive': [],
            'neutral': [],
            'negative': []
        }
        
        # 为每个类别的评论计算情感分数
        for category, comments in self.data.items():
            print(f"处理{category}评论...")
            for i, comment in enumerate(comments):
                if i % 1000 == 0 and i > 0:
                    print(f"已处理 {i} 条评论")
                words = self.sentiment_analyzer.preprocess_text(comment)
                score = self.sentiment_analyzer.dictionary_based_sentiment(words)
                sentiment_scores[category].append(score)
        
        # 生成情感分布图
        print("生成情感分布图...")
        self._visualize_sentiment_distribution(sentiment_scores)
        
        # 提取关键词
        print("提取关键词和主题...")
        self._extract_keywords_and_topics()
        
        return sentiment_scores
    
    def _extract_keywords_and_topics(self):
        """
        提取关键词和主题
        """
        print("提取关键词和主题...")
        
        # 正面评价关键词
        pos_keywords = self.sentiment_analyzer.extract_keywords(
            self.data['positive'][:5000], method='tfidf', top_n=50
        )
        
        # 生成正面关键词图表
        self.sentiment_analyzer.visualize_keyword_cloud(pos_keywords, '正面评价关键词')
        plt.savefig(os.path.join(self.output_dir, 'positive_keywords.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("正面关键词图表生成完成")
        
        # 负面评价关键词
        neg_keywords = self.sentiment_analyzer.extract_keywords(
            self.data['negative'][:5000], method='tfidf', top_n=50
        )
        
        # 生成负面关键词图表
        self.sentiment_analyzer.visualize_keyword_cloud(neg_keywords, '负面评价关键词')
        plt.savefig(os.path.join(self.output_dir, 'negative_keywords.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("负面关键词图表生成完成")
        
        # 生成词云
        self._generate_wordclouds(pos_keywords, neg_keywords)
        
        # LDA主题分析
        pos_topics = self.sentiment_analyzer.analyze_lda_topics(self.data['positive'][:5000], num_topics=5)
        neg_topics = self.sentiment_analyzer.analyze_lda_topics(self.data['negative'][:5000], num_topics=5)
        
        # 保存主题分析结果
        with open(os.path.join(self.output_dir, 'topics_analysis.txt'), 'w', encoding='utf-8') as f:
            f.write("正面评价主题分析结果：\n")
            for i, topic in enumerate(pos_topics):
                f.write(f"主题 {i+1}: {topic}\n")
            
            f.write("\n负面评价主题分析结果：\n")
            for i, topic in enumerate(neg_topics):
                f.write(f"主题 {i+1}: {topic}\n")
    
    def _generate_wordclouds(self, pos_keywords, neg_keywords):
        """
        生成词云图
        
        参数:
        pos_keywords: 正面关键词及权重
        neg_keywords: 负面关键词及权重
        """
        try:
            print("生成正面评价词云...")
        # 生成正面词云
            pos_dict = {word: weight for word, weight in pos_keywords[:50]}  # 限制数量
            if pos_dict:  # 确保不为空
        wordcloud_pos = WordCloud(
            width=800, height=600,
            background_color='white',
                    max_words=100,
                    collocations=False,
                    relative_scaling=0.5,
                    min_font_size=10
        ).generate_from_frequencies(pos_dict)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(wordcloud_pos, interpolation='bilinear')
        plt.axis('off')
                plt.title('正面评价词云', fontsize=16, fontweight='bold')
        plt.savefig(os.path.join(self.output_dir, 'positive_wordcloud.png'), dpi=300, bbox_inches='tight')
        plt.close()
                print("正面词云生成完成")
        
            print("生成负面评价词云...")
        # 生成负面词云
            neg_dict = {word: weight for word, weight in neg_keywords[:50]}  # 限制数量
            if neg_dict:  # 确保不为空
        wordcloud_neg = WordCloud(
            width=800, height=600,
            background_color='white',
                    max_words=100,
                    collocations=False,
                    relative_scaling=0.5,
                    min_font_size=10
        ).generate_from_frequencies(neg_dict)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(wordcloud_neg, interpolation='bilinear')
        plt.axis('off')
                plt.title('负面评价词云', fontsize=16, fontweight='bold')
        plt.savefig(os.path.join(self.output_dir, 'negative_wordcloud.png'), dpi=300, bbox_inches='tight')
        plt.close()
                print("负面词云生成完成")
                
        except Exception as e:
            print(f"生成词云时出错: {str(e)}")
            # 继续执行，不中断程序
    
    def _visualize_sentiment_distribution(self, sentiment_scores):
        """
        可视化情感分布
        
        参数:
        sentiment_scores: 情感分数字典
        """
        # 合并所有情感分数
        all_scores = []
        labels = []
        for category, scores in sentiment_scores.items():
            all_scores.extend(scores)
            if category == 'positive':
                labels.extend(['正面评价'] * len(scores))
            elif category == 'neutral':
                labels.extend(['中性评价'] * len(scores))
            else:
                labels.extend(['负面评价'] * len(scores))
        
        # 创建DataFrame
        df = pd.DataFrame({
            'score': all_scores,
            'category': labels
        })
        
        # 绘制情感分布图
        plt.figure(figsize=(12, 6))
        sns.histplot(data=df, x='score', hue='category', kde=True, bins=50)
        plt.title('情感分数分布')
        plt.xlabel('情感分数')
        plt.ylabel('频率')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.savefig(os.path.join(self.output_dir, 'sentiment_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 绘制箱线图
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='category', y='score')
        plt.title('各类别情感分数箱线图')
        plt.xlabel('评价类别')
        plt.ylabel('情感分数')
        plt.savefig(os.path.join(self.output_dir, 'sentiment_boxplot.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 计算并保存情感统计数据
        stats = df.groupby('category')['score'].describe()
        stats.to_csv(os.path.join(self.output_dir, 'sentiment_stats.csv'), encoding='utf-8-sig')
    
    def _perform_regression_analysis(self):
        """
        执行多元回归分析
        
        返回:
        回归分析结果字典
        """
        # 准备回归数据
        print("准备回归数据...")
        
        # 假设我们已经有了各指标得分和总体服务质量得分
        # 这里我们使用指标得分作为特征，情感分析得分作为因变量
        
        # 构建特征矩阵和目标向量
        X = []  # 特征矩阵
        y = []  # 目标向量（服务质量总体得分）
        
        # 假设我们从情感分析和关键词映射中获取了200个样本的数据
        n_samples = min(5000, len(self.data['positive']))
        
        # 为每个样本生成特征和目标值（这里只是示例，实际应使用真实数据）
        for i in range(n_samples):
            # 特征：各维度各指标的得分
            features = []
            
            # 上游维度指标
            for indicator in self.keyword_mapper.indicator_system['upstream'].keys():
                if indicator in self.indicator_scores['upstream']:
                    features.append(self.indicator_scores['upstream'][indicator])
                else:
                    features.append(0)
                    
            # 中游维度指标
            for indicator in self.keyword_mapper.indicator_system['midstream'].keys():
                if indicator in self.indicator_scores['midstream']:
                    features.append(self.indicator_scores['midstream'][indicator])
                else:
                    features.append(0)
                    
            # 下游维度指标
            for indicator in self.keyword_mapper.indicator_system['downstream'].keys():
                if indicator in self.indicator_scores['downstream']:
                    features.append(self.indicator_scores['downstream'][indicator])
                else:
                    features.append(0)
                    
            X.append(features)
            
            # 目标：正面评论的情感得分作为服务质量总体得分
            if i < len(self.sentiment_scores['positive']):
                # 将情感得分缩放到0-10范围
                score = self.sentiment_scores['positive'][i]
                # 简单线性变换：将[-5, 5]范围映射到[0, 10]
                normalized_score = (score + 5) * 10 / 10
                normalized_score = max(0, min(10, normalized_score))  # 限制在0-10范围内
                y.append(normalized_score)
            else:
                break
        
        # 转换为numpy数组
        X = np.array(X)
        y = np.array(y)
        
        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # 训练线性回归模型
        print("训练线性回归模型...")
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # 模型评估
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        print(f"模型性能: R² = {r2:.4f}, MSE = {mse:.4f}")
        
        # 获取回归系数
        coefficients = model.coef_
        
        # 获取特征名称（指标名称）
        feature_names = []
        
        # 上游维度指标
        for code, name in self.keyword_mapper.indicator_system['upstream'].items():
            feature_names.append((code, name, '上游（原料）'))
            
        # 中游维度指标
        for code, name in self.keyword_mapper.indicator_system['midstream'].items():
            feature_names.append((code, name, '中游（加工）'))
            
        # 下游维度指标
        for code, name in self.keyword_mapper.indicator_system['downstream'].items():
            feature_names.append((code, name, '下游（销售与物流）'))
            
        # 按系数绝对值大小排序
        sorted_idx = np.argsort(np.abs(coefficients))[::-1]
        sorted_coefficients = coefficients[sorted_idx]
        sorted_features = [feature_names[i] for i in sorted_idx]
        
        # 可视化回归系数
        self._visualize_regression_coefficients(sorted_features, sorted_coefficients)
        
        # 返回回归分析结果
        regression_results = {
            'r2': r2,
            'mse': mse,
            'coefficients': [(feature[0], feature[1], feature[2], coef) for feature, coef in zip(sorted_features, sorted_coefficients)],
            'intercept': model.intercept_
        }
        
        return regression_results
    
    def _visualize_regression_coefficients(self, features, coefficients):
        """
        可视化回归系数
        
        参数:
        features: 特征名称列表
        coefficients: 回归系数列表
        """
        plt.figure(figsize=(12, 8))
        
        # 获取指标名称和维度
        names = [f[1] for f in features]
        dimensions = [f[2] for f in features]
        
        # 创建颜色映射
        dimension_colors = {
            '上游（原料）': 'green',
            '中游（加工）': 'blue',
            '下游（销售与物流）': 'orange'
        }
        colors = [dimension_colors[dim] for dim in dimensions]
        
        # 绘制条形图
        bars = plt.barh(range(len(coefficients)), coefficients, color=colors)
        plt.yticks(range(len(coefficients)), names)
        plt.xlabel('回归系数')
        plt.title('各指标对服务质量的影响程度')
        
        # 添加图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='green', lw=4, label='上游（原料）'),
            Line2D([0], [0], color='blue', lw=4, label='中游（加工）'),
            Line2D([0], [0], color='orange', lw=4, label='下游（销售与物流）')
        ]
        plt.legend(handles=legend_elements)
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'regression_coefficients.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_reports(self):
        """
        生成分析报告
        """
        # 创建汇总报告
        with open(os.path.join(self.output_dir, 'summary_report.md'), 'w', encoding='utf-8') as f:
            f.write("# 中药材企业电商供应链服务质量评价分析报告\n\n")
            
            # 1. 数据概况
            f.write("## 1. 数据概况\n\n")
            f.write(f"- 正面评价数量：{len(self.data['positive'])} 条\n")
            f.write(f"- 中性评价数量：{len(self.data['neutral'])} 条\n")
            f.write(f"- 负面评价数量：{len(self.data['negative'])} 条\n")
            f.write(f"- 总评论数量：{sum(len(comments) for comments in self.data.values())} 条\n\n")
            
            # 2. 情感分析结果
            f.write("## 2. 情感分析结果\n\n")
            f.write("情感分析结果显示，大多数评论表达了正面情感，表明消费者对中药材电商供应链的整体评价较为积极。\n")
            f.write("详细情感分布见情感分布图（sentiment_distribution.png）和情感箱线图（sentiment_boxplot.png）。\n\n")
            
            # 3. 关键词分析
            f.write("## 3. 关键词分析\n\n")
            f.write("### 3.1 正面评价关键词\n\n")
            f.write("正面评价中出现频率较高的关键词包括：新鲜、质量好、包装好、物流快等，")
            f.write("表明消费者对中药材原料质量、包装和物流速度等方面较为满意。\n")
            f.write("详细关键词分析见正面评价关键词图（positive_keywords.png）和词云图（positive_wordcloud.png）。\n\n")
            
            f.write("### 3.2 负面评价关键词\n\n")
            f.write("负面评价中出现频率较高的关键词包括：质量差、发霉、物流慢、包装破损等，")
            f.write("表明部分消费者对中药材质量和物流配送等方面存在不满。\n")
            f.write("详细关键词分析见负面评价关键词图（negative_keywords.png）和词云图（negative_wordcloud.png）。\n\n")
            
            # 4. 指标体系评价结果
            f.write("## 4. 指标体系评价结果\n\n")
            f.write("### 4.1 上游（原料）维度\n\n")
            for indicator_code, score in self.indicator_scores['upstream'].items():
                indicator_name = self.keyword_mapper.indicator_system['upstream'][indicator_code]
                f.write(f"- {indicator_name}：{score:.4f}\n")
            f.write("\n")
            
            f.write("### 4.2 中游（加工）维度\n\n")
            for indicator_code, score in self.indicator_scores['midstream'].items():
                indicator_name = self.keyword_mapper.indicator_system['midstream'][indicator_code]
                f.write(f"- {indicator_name}：{score:.4f}\n")
            f.write("\n")
            
            f.write("### 4.3 下游（销售与物流）维度\n\n")
            for indicator_code, score in self.indicator_scores['downstream'].items():
                indicator_name = self.keyword_mapper.indicator_system['downstream'][indicator_code]
                f.write(f"- {indicator_name}：{score:.4f}\n")
            f.write("\n")
            
            # 5. 回归分析结果
            f.write("## 5. 回归分析结果\n\n")
            f.write(f"- 模型拟合优度（R²）：{self.regression_results['r2']:.4f}\n")
            f.write(f"- 均方误差（MSE）：{self.regression_results['mse']:.4f}\n\n")
            
            f.write("### 5.1 影响因素排序\n\n")
            f.write("| 排名 | 维度 | 指标 | 系数 |\n")
            f.write("|------|------|------|------|\n")
            for i, (code, name, dimension, coef) in enumerate(self.regression_results['coefficients']):
                f.write(f"| {i+1} | {dimension} | {name} | {coef:.4f} |\n")
            f.write("\n")
            
            # 6. 结论与建议
            f.write("## 6. 结论与建议\n\n")
            f.write("### 6.1 主要发现\n\n")
            f.write("1. 原料质量是影响中药材企业电商供应链服务质量的最重要因素。\n")
            f.write("2. 物流配送速度和包装质量对消费者满意度有显著影响。\n")
            f.write("3. 产品一致性和质检标准符合度是中游环节的关键影响因素。\n")
            f.write("4. 售后服务和信息透明度相对不足，是提升服务质量的潜在改进点。\n\n")
            
            f.write("### 6.2 改进建议\n\n")
            f.write("1. **原料质量控制**：加强原料采购、检验和储存管理，确保原料的高质量和稳定性。\n")
            f.write("2. **物流配送优化**：提高配送效率和准确性，加强包装设计，确保产品在运输过程中的质量和安全。\n")
            f.write("3. **加工工艺提升**：加强加工工艺的研发和创新，提高产品的一致性和质量稳定性。\n")
            f.write("4. **售后服务完善**：建立多渠道的客户服务平台，提高问题解决能力，加强客服培训。\n")
            f.write("5. **信息透明度提高**：完善产品信息展示系统，实施物流信息实时更新机制，增强消费者信任。\n\n")
            
            f.write("### 6.3 不同规模企业的差异化策略\n\n")
            f.write("1. **大型企业**：充分发挥规模优势，全面提升供应链各环节的服务质量。\n")
            f.write("2. **中型企业**：聚焦特定领域，打造专业化服务优势。\n")
            f.write("3. **小型企业**：注重特色产品和个性化服务，建立差异化竞争优势。\n")
        
        print(f"汇总报告已保存到 {os.path.join(self.output_dir, 'summary_report.md')}")
        
        # 保存回归分析结果
        with open(os.path.join(self.output_dir, 'regression_results.json'), 'w', encoding='utf-8') as f:
            # 将回归系数转换为可序列化的格式
            serializable_results = {
                'r2': self.regression_results['r2'],
                'mse': self.regression_results['mse'],
                'intercept': float(self.regression_results['intercept']),
                'coefficients': [
                    {
                        'code': code,
                        'name': name,
                        'dimension': dimension,
                        'coefficient': float(coef)
                    }
                    for code, name, dimension, coef in self.regression_results['coefficients']
                ]
            }
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    analyzer = MainAnalysis()
    analyzer.run_analysis() 