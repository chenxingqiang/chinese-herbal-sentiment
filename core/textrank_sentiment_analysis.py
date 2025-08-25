#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于TextRank的情感分析模型
实现论文中提到的TextRank情感分析算法
"""

import os
import glob
import pandas as pd
import numpy as np
import jieba
import re
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 用于显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用于正常显示负号

class TextRankSentimentAnalysis:
    def __init__(self):
        # 加载停用词
        self.stopwords = self.load_stopwords()
        # 初始化jieba分词
        jieba.initialize()
        # 设置模型参数
        self.window_size = 3  # TextRank窗口大小
        self.top_n = 100  # 提取的特征数量
        self.min_df = 5  # 最小文档频率
        self.max_df = 0.8  # 最大文档频率

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
        return ' '.join(words)

    def extract_textrank_features(self, texts, top_n=None):
        """使用TextRank提取文本特征"""
        if top_n is None:
            top_n = self.top_n

        print(f"使用TextRank提取前{top_n}个关键词作为特征...")

        # 预处理所有文本
        preprocessed_texts = [self.preprocess(text) for text in texts]

        # 使用TF-IDF向量化器获取词汇表
        tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=self.min_df,
            max_df=self.max_df
        )
        tfidf_vectorizer.fit(preprocessed_texts)
        vocabulary = tfidf_vectorizer.get_feature_names_out()

        # 构建词共现矩阵
        word_graph = nx.Graph()

        # 为每个文本构建词图
        for text in preprocessed_texts:
            words = text.split()
            for i, word in enumerate(words):
                if word not in vocabulary:
                    continue

                # 在窗口大小内查找共现词
                window_start = max(0, i - self.window_size)
                window_end = min(len(words), i + self.window_size + 1)

                for j in range(window_start, window_end):
                    if i != j and j < len(words):
                        coword = words[j]
                        if coword not in vocabulary:
                            continue

                        # 更新词图
                        if word_graph.has_edge(word, coword):
                            word_graph[word][coword]['weight'] += 1
                        else:
                            word_graph.add_edge(word, coword, weight=1)

        # 使用PageRank算法计算词的重要性
        try:
            word_ranks = nx.pagerank(word_graph)
        except:
            print("警告: PageRank算法失败，可能是图为空或不连通。使用度中心性代替。")
            word_ranks = {node: score for node, score in nx.degree_centrality(word_graph).items()}

        # 选择前top_n个重要的词作为特征
        top_words = sorted(word_ranks.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_words = [word for word, _ in top_words]

        print(f"提取了 {len(top_words)} 个TextRank特征")

        # 使用计数向量化器将文本转换为特征向量
        count_vectorizer = CountVectorizer(vocabulary=top_words)
        features = count_vectorizer.fit_transform(preprocessed_texts)

        return features, top_words

    def train_model(self, X_train, y_train):
        """训练分类模型"""
        print(f"训练基于TextRank特征的SVM模型，数据量: {X_train.shape[0]}条样本，特征维度: {X_train.shape[1]}")

        # 使用LinearSVC，速度更快
        model = LinearSVC(dual=False, max_iter=1000)
        model.fit(X_train, y_train)

        return model

    def evaluate_model(self, model, X_test, y_test):
        """评估模型性能"""
        # 预测
        y_pred = model.predict(X_test)

        # 计算评估指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred)

        print(f"TextRank+SVM准确率: {accuracy:.4f}")
        print(f"TextRank+SVM精确率: {precision:.4f}")
        print(f"TextRank+SVM召回率: {recall:.4f}")
        print(f"TextRank+SVM F1值: {f1:.4f}")
        print("\n分类报告:")
        print(report)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'report': report
        }

    def visualize_textrank_features(self, top_words):
        """可视化TextRank特征"""
        plt.figure(figsize=(12, 6))

        # 获取前20个词
        top_20 = top_words[:20]

        # 绘制条形图
        y_pos = np.arange(len(top_20))
        plt.barh(y_pos, range(len(top_20), 0, -1))
        plt.yticks(y_pos, top_20)
        plt.xlabel('相对重要性')
        plt.title('TextRank提取的前20个特征词')
        plt.tight_layout()
        plt.savefig('textrank_features.png')
        plt.close()

    def analyze_comments(self, comments_files, sample_size=None):
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

        # 如果指定了样本大小，随机采样
        if sample_size and len(all_comments) > sample_size:
            indices = np.random.choice(len(all_comments), sample_size, replace=False)
            all_comments = [all_comments[i] for i in indices]
            all_labels = [all_labels[i] for i in indices]
            print(f"随机采样 {sample_size} 条评论进行分析")

        return self.analyze_comments_with_data(all_comments, all_labels)

    def analyze_comments_with_data(self, comments, labels):
        """分析直接传入的评论数据"""
        print(f"使用直接传入的 {len(comments)} 条评论进行TextRank分析")

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            comments, labels, test_size=0.2, random_state=42
        )

        print(f"训练集大小: {len(X_train)}")
        print(f"测试集大小: {len(X_test)}")

        # 提取TextRank特征
        X_train_features, top_words = self.extract_textrank_features(X_train)
        X_test_features, _ = self.extract_textrank_features(X_test, top_n=len(top_words))

        # 可视化TextRank特征
        self.visualize_textrank_features(top_words)

        # 训练模型
        model = self.train_model(X_train_features, y_train)

        # 评估模型
        results = self.evaluate_model(model, X_test_features, y_test)

        return results

def main():
    # 创建TextRank情感分析对象
    textrank_analyzer = TextRankSentimentAnalysis()

    # 获取评论文件列表
    comments_files = []
    for category in ['好评', '中评', '差评']:
        files = glob.glob(f"comments/*{category}*.xls") + glob.glob(f"comments/*{category}*.xlsx")
        comments_files.extend(files[:2])  # 每类取前两个文件进行测试

    # 分析评论
    results = textrank_analyzer.analyze_comments(comments_files)

    print("\n分析完成！结果已保存到 textrank_features.png")

if __name__ == "__main__":
    main()