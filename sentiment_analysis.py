#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
中药材电商评论情感分析系统
实现论文中提到的情感分析算法
"""

import os
import glob
import pandas as pd
import numpy as np
import jieba
import re
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 用于显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用于正常显示负号

class SentimentAnalysis:
    def __init__(self):
        # 加载停用词
        self.stopwords = self.load_stopwords()
        # 加载情感词典
        self.sentiment_dict = self.load_sentiment_dict()
        # 加载否定词
        self.negation_words = self.load_negation_words()
        # 加载程度副词
        self.degree_words = self.load_degree_words()
        # 初始化jieba分词
        jieba.initialize()
        
    def load_stopwords(self):
        """加载停用词表"""
        stopwords = set()
        try:
            # 尝试从文件加载停用词，如果文件不存在则使用默认停用词
            with open('stopwords.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    stopwords.add(line.strip())
        except:
            # 默认停用词
            default_stopwords = ['的', '了', '和', '是', '就', '都', '而', '及', '与', '着',
                               '或', '一个', '没有', '我们', '你们', '他们', '它们', '这个',
                               '那个', '这些', '那些', '这样', '那样', '之', '的话', '什么']
            stopwords = set(default_stopwords)
        return stopwords
    
    def load_sentiment_dict(self):
        """加载情感词典"""
        # 在实际应用中，应该从文件加载完整的情感词典
        # 这里为了演示，使用一个简化的情感词典
        pos_words = ['优质', '好', '不错', '满意', '喜欢', '推荐', '赞', '高质量', '实惠', '物美价廉',
                    '有效', '有用', '效果好', '正品', '值得', '放心', '专业', '快速', '周到', '耐心',
                    '新鲜', '精美', '优惠', '便宜', '划算', '良心', '贴心', '舒适', '温暖', '干净']
        
        neg_words = ['差', '贵', '慢', '不好', '不满意', '失望', '退货', '垃圾', '坑', '骗',
                    '假货', '差评', '无效', '不值', '不推荐', '难吃', '难用', '不舒服', '不专业', '敷衍',
                    '过期', '变质', '破损', '缺斤少两', '虚假', '夸大', '误导', '不靠谱', '不负责', '敷衍']
        
        sentiment_dict = {}
        for word in pos_words:
            sentiment_dict[word] = 1
        for word in neg_words:
            sentiment_dict[word] = -1
        return sentiment_dict
    
    def load_negation_words(self):
        """加载否定词"""
        return ['不', '没', '无', '非', '莫', '弗', '勿', '毋', '未', '否', '别', '无法', '不能', '难以']
    
    def load_degree_words(self):
        """加载程度副词及其权重"""
        return {
            '极其': 2.0, '非常': 2.0, '特别': 2.0, '十分': 1.6, '很': 1.6, 
            '较为': 1.2, '比较': 1.2, '稍微': 1.2, '略微': 1.2, 
            '不太': 0.6, '不怎么': 0.6, '不太': 0.6
        }
    
    def preprocess(self, text):
        """文本预处理"""
        if not isinstance(text, str):
            return ""
        # 去除特殊字符和数字
        text = re.sub(r'[^\u4e00-\u9fa5]', ' ', text)
        # 分词
        words = jieba.lcut(text)
        # 去除停用词
        words = [word for word in words if word not in self.stopwords and len(word.strip()) > 0]
        return words
    
    def dictionary_based_analysis(self, text):
        """基于词典的情感分析"""
        words = self.preprocess(text)
        if not words:
            return 0
        
        sentiment_score = 0
        negation = False
        degree = 1.0
        
        for i, word in enumerate(words):
            # 处理否定词
            if word in self.negation_words:
                negation = not negation
                continue
            
            # 处理程度副词
            if word in self.degree_words:
                degree = self.degree_words[word]
                continue
            
            # 处理情感词
            if word in self.sentiment_dict:
                score = self.sentiment_dict[word]
                # 应用否定词和程度副词的影响
                if negation:
                    score = -score
                    negation = False
                score *= degree
                degree = 1.0
                sentiment_score += score
        
        # 归一化得分到[-1, 1]区间
        if sentiment_score > 0:
            return min(sentiment_score / 5, 1)
        else:
            return max(sentiment_score / 5, -1)
    
    def extract_features(self, texts):
        """提取TF-IDF特征"""
        # 预处理文本
        preprocessed_texts = [' '.join(self.preprocess(text)) for text in texts]
        # TF-IDF向量化
        vectorizer = TfidfVectorizer(max_features=5000)
        features = vectorizer.fit_transform(preprocessed_texts)
        return features, vectorizer
    
    def train_machine_learning_model(self, X_train, y_train, model_type='svm'):
        """训练机器学习模型"""
        if model_type == 'svm':
            model = SVC(kernel='linear')
        elif model_type == 'nb':
            model = MultinomialNB()
        else:
            raise ValueError("不支持的模型类型")
        
        model.fit(X_train, y_train)
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        """评估模型性能"""
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'report': report
        }
    
    def analyze_comments(self, comments_files):
        """分析评论数据"""
        all_comments = []
        all_labels = []
        
        # 读取所有评论文件
        for file_path in comments_files:
            try:
                df = pd.read_excel(file_path)
                # 提取评论内容
                comments = df['评论内容'].tolist()
                
                # 根据文件名确定情感标签
                label = 1 if '好评' in file_path else (-1 if '差评' in file_path else 0)
                labels = [label] * len(comments)
                
                all_comments.extend(comments)
                all_labels.extend(labels)
                
                print(f"已读取 {file_path}，包含 {len(comments)} 条评论")
            except Exception as e:
                print(f"读取 {file_path} 失败: {e}")
        
        print(f"共读取 {len(all_comments)} 条评论")
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            all_comments, all_labels, test_size=0.2, random_state=42
        )
        
        # 基于词典的情感分析
        print("\n使用基于词典的情感分析方法...")
        dict_scores = []
        dict_preds = []
        for comment in X_test:
            score = self.dictionary_based_analysis(comment)
            dict_scores.append(score)
            # 将得分转换为分类标签
            if score > 0.2:
                dict_preds.append(1)  # 好评
            elif score < -0.2:
                dict_preds.append(-1)  # 差评
            else:
                dict_preds.append(0)  # 中评
        
        # 计算基于词典方法的性能
        dict_accuracy = accuracy_score(y_test, dict_preds)
        dict_precision = precision_score(y_test, dict_preds, average='weighted')
        dict_recall = recall_score(y_test, dict_preds, average='weighted')
        dict_f1 = f1_score(y_test, dict_preds, average='weighted')
        
        print(f"词典方法准确率: {dict_accuracy:.4f}")
        print(f"词典方法精确率: {dict_precision:.4f}")
        print(f"词典方法召回率: {dict_recall:.4f}")
        print(f"词典方法F1值: {dict_f1:.4f}")
        
        # 基于机器学习的情感分析
        print("\n使用基于机器学习的情感分析方法...")
        # 提取特征
        X_train_features, vectorizer = self.extract_features(X_train)
        X_test_features = vectorizer.transform(X_test)
        
        # 训练SVM模型
        print("训练SVM模型...")
        svm_model = self.train_machine_learning_model(X_train_features, y_train, 'svm')
        svm_results = self.evaluate_model(svm_model, X_test_features, y_test)
        
        print(f"SVM准确率: {svm_results['accuracy']:.4f}")
        print(f"SVM精确率: {svm_results['precision']:.4f}")
        print(f"SVM召回率: {svm_results['recall']:.4f}")
        print(f"SVM F1值: {svm_results['f1']:.4f}")
        
        # 训练朴素贝叶斯模型
        print("\n训练朴素贝叶斯模型...")
        nb_model = self.train_machine_learning_model(X_train_features, y_train, 'nb')
        nb_results = self.evaluate_model(nb_model, X_test_features, y_test)
        
        print(f"朴素贝叶斯准确率: {nb_results['accuracy']:.4f}")
        print(f"朴素贝叶斯精确率: {nb_results['precision']:.4f}")
        print(f"朴素贝叶斯召回率: {nb_results['recall']:.4f}")
        print(f"朴素贝叶斯F1值: {nb_results['f1']:.4f}")
        
        # 可视化结果
        self.visualize_results({
            '词典方法': [dict_accuracy, dict_precision, dict_recall, dict_f1],
            'SVM': [svm_results['accuracy'], svm_results['precision'], svm_results['recall'], svm_results['f1']],
            '朴素贝叶斯': [nb_results['accuracy'], nb_results['precision'], nb_results['recall'], nb_results['f1']]
        })
        
        return {
            'dictionary': {
                'accuracy': dict_accuracy,
                'precision': dict_precision,
                'recall': dict_recall,
                'f1': dict_f1
            },
            'svm': svm_results,
            'naive_bayes': nb_results
        }
    
    def analyze_comments_with_data(self, comments, labels):
        """分析直接传入的评论数据"""
        print(f"使用直接传入的 {len(comments)} 条评论进行分析")
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            comments, labels, test_size=0.2, random_state=42
        )
        
        # 基于词典的情感分析
        print("\n使用基于词典的情感分析方法...")
        dict_scores = []
        dict_preds = []
        for comment in X_test:
            score = self.dictionary_based_analysis(comment)
            dict_scores.append(score)
            # 将得分转换为分类标签
            if score > 0.2:
                dict_preds.append(1)  # 好评
            elif score < -0.2:
                dict_preds.append(-1)  # 差评
            else:
                dict_preds.append(0)  # 中评
        
        # 计算基于词典方法的性能
        dict_accuracy = accuracy_score(y_test, dict_preds)
        dict_precision = precision_score(y_test, dict_preds, average='weighted')
        dict_recall = recall_score(y_test, dict_preds, average='weighted')
        dict_f1 = f1_score(y_test, dict_preds, average='weighted')
        
        print(f"词典方法准确率: {dict_accuracy:.4f}")
        print(f"词典方法精确率: {dict_precision:.4f}")
        print(f"词典方法召回率: {dict_recall:.4f}")
        print(f"词典方法F1值: {dict_f1:.4f}")
        
        # 基于机器学习的情感分析
        print("\n使用基于机器学习的情感分析方法...")
        # 提取特征
        X_train_features, vectorizer = self.extract_features(X_train)
        X_test_features = vectorizer.transform(X_test)
        
        # 训练SVM模型
        print("训练SVM模型...")
        svm_model = self.train_machine_learning_model(X_train_features, y_train, 'svm')
        svm_results = self.evaluate_model(svm_model, X_test_features, y_test)
        
        print(f"SVM准确率: {svm_results['accuracy']:.4f}")
        print(f"SVM精确率: {svm_results['precision']:.4f}")
        print(f"SVM召回率: {svm_results['recall']:.4f}")
        print(f"SVM F1值: {svm_results['f1']:.4f}")
        
        # 训练朴素贝叶斯模型
        print("\n训练朴素贝叶斯模型...")
        nb_model = self.train_machine_learning_model(X_train_features, y_train, 'nb')
        nb_results = self.evaluate_model(nb_model, X_test_features, y_test)
        
        print(f"朴素贝叶斯准确率: {nb_results['accuracy']:.4f}")
        print(f"朴素贝叶斯精确率: {nb_results['precision']:.4f}")
        print(f"朴素贝叶斯召回率: {nb_results['recall']:.4f}")
        print(f"朴素贝叶斯F1值: {nb_results['f1']:.4f}")
        
        # 可视化结果
        self.visualize_results({
            '词典方法': [dict_accuracy, dict_precision, dict_recall, dict_f1],
            'SVM': [svm_results['accuracy'], svm_results['precision'], svm_results['recall'], svm_results['f1']],
            '朴素贝叶斯': [nb_results['accuracy'], nb_results['precision'], nb_results['recall'], nb_results['f1']]
        })
        
        return {
            'dictionary': {
                'accuracy': dict_accuracy,
                'precision': dict_precision,
                'recall': dict_recall,
                'f1': dict_f1
            },
            'svm': svm_results,
            'naive_bayes': nb_results
        }
    
    def visualize_results(self, results):
        """可视化不同算法的性能比较"""
        metrics = ['准确率', '精确率', '召回率', 'F1值']
        
        # 设置图表
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 设置柱状图的宽度和位置
        x = np.arange(len(metrics))
        width = 0.25
        multiplier = 0
        
        # 绘制每个算法的性能指标
        for algorithm, scores in results.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, scores, width, label=algorithm)
            ax.bar_label(rects, fmt='%.2f', padding=3)
            multiplier += 1
        
        # 添加标签和图例
        ax.set_title('情感分析算法性能比较')
        ax.set_xticks(x + width, metrics)
        ax.set_ylim(0, 1.2)
        ax.legend(loc='upper left', ncols=len(results.keys()))
        
        plt.tight_layout()
        plt.savefig('algorithm_comparison.png')
        plt.close()

def main():
    # 创建情感分析对象
    analyzer = SentimentAnalysis()
    
    # 获取评论文件列表
    comments_files = []
    for category in ['好评', '中评', '差评']:
        files = glob.glob(f"comments/*{category}*.xls") + glob.glob(f"comments/*{category}*.xlsx")
        # 限制每类评论的文件数量，以加快处理速度
        comments_files.extend(files[:3])
    
    # 分析评论
    results = analyzer.analyze_comments(comments_files)
    
    print("\n分析完成！结果已保存到 algorithm_comparison.png")

if __name__ == "__main__":
    main() 