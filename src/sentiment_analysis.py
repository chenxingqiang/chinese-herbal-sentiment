#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import jieba
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.font_manager as fm
from gensim import corpora, models
import warnings

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
warnings.filterwarnings('ignore')

class SentimentAnalysis:
    def __init__(self, comments_dir='comments'):
        """
        初始化情感分析类
        
        参数:
        comments_dir: 评论数据目录
        """
        self.comments_dir = comments_dir
        self.pos_dict = set()  # 正面词典
        self.neg_dict = set()  # 负面词典
        self.stop_words = set()  # 停用词
        self.degree_dict = {}  # 程度副词
        self.negation_dict = set()  # 否定词
        self.load_dictionaries()
        
    def load_dictionaries(self):
        """
        加载词典
        """
        # 这里使用简化的词典，实际应用中应使用更完整的词典
        
        # 正面词典
        pos_words = [
            '好', '优秀', '满意', '不错', '高质量', '新鲜', '推荐', '快', '方便', 
            '好评', '正品', '优质', '实惠', '便宜', '贴心', '耐心', '精美', '有效',
            '专业', '精致', '清晰', '及时', '靠谱', '满足', '效果好', '值得', '喜欢'
        ]
        self.pos_dict = set(pos_words)
        
        # 负面词典
        neg_words = [
            '差', '差评', '慢', '贵', '难', '假', '坏', '问题', '不满', '不好',
            '延迟', '退货', '差劲', '虚假', '骗', '失望', '糟糕', '生气', '后悔',
            '发霉', '变质', '过期', '不新鲜', '不值', '破损', '错误', '差距', '缺点'
        ]
        self.neg_dict = set(neg_words)
        
        # 停用词
        stop_words = [
            '的', '了', '和', '是', '就', '都', '而', '及', '与', '或', '一个',
            '那个', '这个', '一', '在', '中', '有', '对', '上', '下', '但', '也',
            '我', '你', '他', '她', '它', '们', '这', '那', '啊', '吧', '呢', '吗'
        ]
        self.stop_words = set(stop_words)
        
        # 程度副词
        degree_dict = {
            '极其': 2.0, '非常': 1.8, '特别': 1.8, '很': 1.6, '比较': 1.4, '略微': 1.2,
            '稍微': 1.2, '有点': 1.2, '不太': 0.8, '不怎么': 0.6, '不很': 0.4
        }
        self.degree_dict = degree_dict
        
        # 否定词
        negation_words = ['不', '没', '无', '非', '莫', '弗', '毋', '勿', '未', '否', '别', '无须', '并非']
        self.negation_dict = set(negation_words)
        
    def load_excel_data(self):
        """
        加载Excel评论数据
        
        返回:
        分类好的评论数据字典 {'positive': [...], 'neutral': [...], 'negative': [...]}
        """
        data = {
            'positive': [],
            'neutral': [],
            'negative': []
        }
        
        # 遍历评论目录下的所有文件
        for filename in os.listdir(self.comments_dir):
            if not (filename.endswith('.xls') or filename.endswith('.xlsx')):
                continue
            
            filepath = os.path.join(self.comments_dir, filename)
            try:
                # 根据文件名分类评论
                if '好评' in filename:
                    category = 'positive'
                elif '中评' in filename:
                    category = 'neutral'
                elif '差评' in filename:
                    category = 'negative'
                else:
                    continue
                
                # 读取Excel文件
                df = pd.read_excel(filepath)
                
                # 找到包含评论的列（可能需要根据实际数据调整）
                comment_column = None
                for col in df.columns:
                    if isinstance(col, str) and ('评论' in col or '内容' in col or 'content' in col.lower()):
                        comment_column = col
                        break
                
                if comment_column is None and len(df.columns) > 0:
                    # 如果找不到，默认使用第一列
                    comment_column = df.columns[0]
                
                if comment_column is not None:
                    # 提取评论内容
                    comments = df[comment_column].dropna().astype(str).tolist()
                    data[category].extend(comments)
                
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")
                
        return data
    
    def preprocess_text(self, text):
        """
        文本预处理：去除特殊字符，分词等
        
        参数:
        text: 输入的文本
        
        返回:
        分词列表
        """
        # 去除特殊字符
        text = re.sub(r'[^\w\s]', '', text)
        
        # 分词
        words = jieba.lcut(text)
        
        # 去除停用词
        words = [word for word in words if word not in self.stop_words and len(word.strip()) > 0]
        
        return words
    
    def dictionary_based_sentiment(self, words):
        """
        基于词典的情感分析
        
        参数:
        words: 分词列表
        
        返回:
        情感得分
        """
        score = 0
        negation = False
        degree = 1.0
        
        for i, word in enumerate(words):
            # 判断是否为否定词
            if word in self.negation_dict:
                negation = not negation
                continue
                
            # 判断是否为程度副词
            if word in self.degree_dict:
                degree = self.degree_dict[word]
                continue
                
            # 判断是否为情感词
            if word in self.pos_dict:
                # 正面词
                word_score = 1.0
            elif word in self.neg_dict:
                # 负面词
                word_score = -1.0
            else:
                # 中性词
                continue
                
            # 根据否定词和程度副词调整分数
            if negation:
                word_score *= -0.8
                negation = False
            
            word_score *= degree
            score += word_score
            degree = 1.0
            
        return score
    
    def extract_keywords(self, texts, method='tfidf', top_n=20):
        """
        提取关键词
        
        参数:
        texts: 文本列表
        method: 提取方法，'tfidf'或'textrank'
        top_n: 返回前N个关键词
        
        返回:
        关键词及其权重
        """
        if method == 'tfidf':
            # 使用TF-IDF提取关键词
            preprocessed_texts = [' '.join(self.preprocess_text(text)) for text in texts]
            
            # 创建TF-IDF向量化器
            vectorizer = TfidfVectorizer(max_features=1000)
            X = vectorizer.fit_transform(preprocessed_texts)
            
            # 获取特征名称（即词语）
            feature_names = vectorizer.get_feature_names_out()
            
            # 计算每个词的TF-IDF值的平均值
            tfidf_mean = X.mean(axis=0).A1
            
            # 构建词-权重对
            keywords = [(feature_names[i], tfidf_mean[i]) for i in range(len(feature_names))]
            
            # 按权重排序并返回前N个
            return sorted(keywords, key=lambda x: x[1], reverse=True)[:top_n]
            
        elif method == 'textrank':
            # 使用TextRank提取关键词（简化版）
            all_words = []
            for text in texts:
                all_words.extend(self.preprocess_text(text))
                
            # 计算词频
            word_freq = Counter(all_words)
            
            # 返回最常见的词作为关键词（简化的TextRank实现）
            return word_freq.most_common(top_n)
            
        else:
            raise ValueError("method must be either 'tfidf' or 'textrank'")
    
    def analyze_lda_topics(self, texts, num_topics=5, num_words=10):
        """
        使用LDA进行主题分析
        
        参数:
        texts: 文本列表
        num_topics: 主题数量
        num_words: 每个主题显示的词语数量
        
        返回:
        主题-词语矩阵
        """
        # 文本预处理
        preprocessed_texts = [self.preprocess_text(text) for text in texts]
        
        # 创建字典和语料库
        dictionary = corpora.Dictionary(preprocessed_texts)
        corpus = [dictionary.doc2bow(text) for text in preprocessed_texts]
        
        # 训练LDA模型
        lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        
        # 获取主题
        topics = lda_model.print_topics(num_words=num_words)
        
        return topics
    
    def visualize_sentiment_distribution(self, sentiment_scores):
        """
        可视化情感分布
        
        参数:
        sentiment_scores: 情感分数列表
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(sentiment_scores, kde=True, bins=50)
        plt.title('情感分数分布')
        plt.xlabel('情感分数')
        plt.ylabel('频率')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.savefig('sentiment_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_keyword_cloud(self, keywords, title):
        """
        可视化关键词云
        
        参数:
        keywords: (word, weight)元组列表
        title: 图表标题
        """
        plt.figure(figsize=(12, 8))
        
        # 提取词和权重
        words = [word for word, _ in keywords[:20]]  # 只显示前20个
        weights = [weight for _, weight in keywords[:20]]
        
        # 创建条形图
        y_pos = np.arange(len(words))
        plt.barh(y_pos, weights, align='center', color='steelblue', alpha=0.7)
        plt.yticks(y_pos, words)
        plt.xlabel('TF-IDF权重', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        
        # 调整布局
        plt.tight_layout()
        
        # 不在这里保存，让调用方决定保存位置
        # plt.close() 也不在这里调用，让调用方控制
    
    def visualize_category_distribution(self, data):
        """
        可视化评论类别分布
        
        参数:
        data: 评论数据字典
        """
        categories = ['正面评价', '中性评价', '负面评价']
        counts = [len(data['positive']), len(data['neutral']), len(data['negative'])]
        
        plt.figure(figsize=(10, 6))
        plt.pie(counts, labels=categories, autopct='%1.1f%%', startangle=90, colors=['green', 'gray', 'red'])
        plt.axis('equal')
        plt.title('评论类别分布')
        plt.savefig('category_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def train_machine_learning_model(self, data):
        """
        训练机器学习情感分析模型（简化版）
        
        参数:
        data: 评论数据字典
        
        返回:
        模型性能指标
        """
        from sklearn.svm import SVC
        
        # 准备训练数据
        texts = []
        labels = []
        
        # 添加正面评价
        texts.extend(data['positive'])
        labels.extend([1] * len(data['positive']))
        
        # 添加中性评价
        texts.extend(data['neutral'])
        labels.extend([0] * len(data['neutral']))
        
        # 添加负面评价
        texts.extend(data['negative'])
        labels.extend([-1] * len(data['negative']))
        
        # 文本预处理
        preprocessed_texts = [' '.join(self.preprocess_text(text)) for text in texts]
        
        # 特征提取
        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform(preprocessed_texts)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # 训练SVM模型
        model = SVC(kernel='linear')
        model.fit(X_train, y_train)
        
        # 测试模型
        y_pred = model.predict(X_test)
        
        # 计算性能指标
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='macro'),
            'recall': recall_score(y_test, y_pred, average='macro'),
            'f1': f1_score(y_test, y_pred, average='macro')
        }
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        
        return metrics, cm
    
    def run_full_analysis(self):
        """
        运行完整的分析流程
        """
        print("开始加载评论数据...")
        data = self.load_excel_data()
        
        # 打印数据统计
        total_comments = sum(len(comments) for comments in data.values())
        print(f"加载完成，共有 {total_comments} 条评论")
        print(f"正面评价：{len(data['positive'])} 条")
        print(f"中性评价：{len(data['neutral'])} 条")
        print(f"负面评价：{len(data['negative'])} 条")
        
        # 可视化类别分布
        print("生成评论类别分布图...")
        self.visualize_category_distribution(data)
        
        # 基于词典的情感分析
        print("执行基于词典的情感分析...")
        sentiment_scores = []
        for category, comments in data.items():
            for comment in comments[:1000]:  # 为了演示，每类仅分析1000条
                words = self.preprocess_text(comment)
                score = self.dictionary_based_sentiment(words)
                sentiment_scores.append(score)
        
        # 可视化情感分布
        print("生成情感分数分布图...")
        self.visualize_sentiment_distribution(sentiment_scores)
        
        # 提取关键词
        print("提取关键词...")
        # 正面评价关键词
        pos_keywords = self.extract_keywords(data['positive'][:2000], method='tfidf', top_n=20)
        self.visualize_keyword_cloud(pos_keywords, '正面评价关键词')
        
        # 负面评价关键词
        neg_keywords = self.extract_keywords(data['negative'][:2000], method='tfidf', top_n=20)
        self.visualize_keyword_cloud(neg_keywords, '负面评价关键词')
        
        # LDA主题分析
        print("执行LDA主题分析...")
        # 正面评价主题
        pos_topics = self.analyze_lda_topics(data['positive'][:2000], num_topics=5)
        print("\n正面评价主题：")
        for i, topic in enumerate(pos_topics):
            print(f"主题 {i+1}: {topic}")
        
        # 负面评价主题
        neg_topics = self.analyze_lda_topics(data['negative'][:2000], num_topics=5)
        print("\n负面评价主题：")
        for i, topic in enumerate(neg_topics):
            print(f"主题 {i+1}: {topic}")
        
        # 训练机器学习模型
        print("\n训练机器学习情感分析模型...")
        try:
            metrics, cm = self.train_machine_learning_model(data)
            print("\n模型性能指标：")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
                
            print("\n混淆矩阵：")
            print(cm)
        except Exception as e:
            print(f"训练模型时出错: {e}")
            
        print("\n分析完成！")


if __name__ == "__main__":
    analyzer = SentimentAnalysis()
    analyzer.run_full_analysis() 