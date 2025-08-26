#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
关键词提取算法实现
实现论文中提到的TF-IDF、TextRank和LDA关键词提取算法
"""

import os
import glob
import pandas as pd
import numpy as np
import jieba
import jieba.analyse
import re
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
# Optional gensim import
try:
    from gensim import corpora, models
    GENSIM_AVAILABLE = True
except ImportError:
    print("Warning: gensim not available. LDA topic modeling will be disabled.")
    GENSIM_AVAILABLE = False
    corpora = None
    models = None
import networkx as nx

# Optional wordcloud import
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    print("Warning: wordcloud not available. Word cloud generation will be disabled.")
    WORDCLOUD_AVAILABLE = False
    WordCloud = None
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 用于显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用于正常显示负号

class KeywordExtraction:
    def __init__(self):
        # 加载停用词
        self.stopwords = self.load_stopwords()
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

    def preprocess(self, text):
        """文本预处理"""
        if not isinstance(text, str):
            return ""
        # 去除特殊字符和数字
        text = re.sub(r'[^\u4e00-\u9fa5]', ' ', text)
        # 分词
        words = jieba.lcut(text)
        # 去除停用词
        words = [word for word in words if word not in self.stopwords and len(word.strip()) > 1]
        return words

    def tfidf_extraction(self, texts, top_k=10):
        """基于TF-IDF的关键词提取"""
        # 预处理文本
        preprocessed_texts = [' '.join(self.preprocess(text)) for text in texts]

        # TF-IDF向量化
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(preprocessed_texts)

        # 获取特征名称（词语）
        feature_names = vectorizer.get_feature_names_out()

        # 提取每个文档的关键词
        keywords_list = []
        for i in range(len(texts)):
            # 获取文档的TF-IDF向量
            tfidf_vector = tfidf_matrix[i].toarray()[0]
            # 获取词语和对应的TF-IDF值
            word_tfidf = [(feature_names[j], tfidf_vector[j]) for j in range(len(feature_names))]
            # 按TF-IDF值降序排序
            word_tfidf.sort(key=lambda x: x[1], reverse=True)
            # 提取前top_k个关键词
            keywords = [(word, score) for word, score in word_tfidf[:top_k]]
            keywords_list.append(keywords)

        return keywords_list

    def textrank_extraction(self, texts, top_k=10):
        """基于TextRank的关键词提取"""
        keywords_list = []

        for text in texts:
            # 使用jieba的TextRank实现
            keywords = jieba.analyse.textrank(text, topK=top_k, withWeight=True, allowPOS=('ns', 'n', 'vn', 'v'))
            keywords_list.append(keywords)

        return keywords_list

    def custom_textrank(self, text, window_size=5, top_k=10):
        """自定义TextRank实现"""
        words = self.preprocess(text)
        if len(words) < window_size:
            return []

        # 构建词图
        graph = nx.Graph()
        for i in range(len(words) - window_size + 1):
            window = words[i:i+window_size]
            for j in range(window_size):
                for k in range(j+1, window_size):
                    if window[j] != window[k]:
                        if not graph.has_edge(window[j], window[k]):
                            graph.add_edge(window[j], window[k], weight=1)
                        else:
                            graph[window[j]][window[k]]['weight'] += 1

        # 运行PageRank算法
        scores = nx.pagerank(graph, alpha=0.85)

        # 按得分降序排序
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # 返回前top_k个关键词
        return sorted_words[:top_k]

    def lda_extraction(self, texts, num_topics=5, top_k=10):
        """基于LDA的关键词提取"""
        if not GENSIM_AVAILABLE:
            print("Warning: gensim not available. Skipping LDA extraction.")
            return []
            
        # 预处理文本
        preprocessed_texts = [self.preprocess(text) for text in texts]

        # 创建词典
        dictionary = corpora.Dictionary(preprocessed_texts)

        # 创建文档-词语矩阵
        corpus = [dictionary.doc2bow(text) for text in preprocessed_texts]

        # 训练LDA模型
        lda = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=20)

        # 提取主题关键词
        topics = lda.print_topics(num_topics=num_topics, num_words=top_k)

        # 提取每个文档的主题分布
        doc_topics = [lda.get_document_topics(doc) for doc in corpus]

        # 提取每个文档的关键词
        keywords_list = []
        for i, doc in enumerate(doc_topics):
            # 获取文档的主题分布
            doc_topic_dist = sorted(doc, key=lambda x: x[1], reverse=True)

            # 如果文档没有明显的主题，跳过
            if not doc_topic_dist:
                keywords_list.append([])
                continue

            # 获取文档的主要主题
            main_topic_id = doc_topic_dist[0][0]

            # 获取主题的关键词
            topic_keywords = lda.show_topic(main_topic_id, top_k)
            keywords_list.append(topic_keywords)

        return keywords_list, topics

    def extract_keywords(self, comments_files, top_k=10):
        """从评论文件中提取关键词"""
        all_comments = []

        # 读取所有评论文件
        for file_path in comments_files:
            try:
                df = pd.read_excel(file_path)
                # 提取评论内容
                comments = df['评论内容'].tolist()
                all_comments.extend(comments)

                print(f"已读取 {file_path}，包含 {len(comments)} 条评论")
            except Exception as e:
                print(f"读取 {file_path} 失败: {e}")

        print(f"共读取 {len(all_comments)} 条评论")

        # 限制评论数量，以加快处理速度
        if len(all_comments) > 1000:
            all_comments = all_comments[:1000]
            print(f"为加快处理速度，使用前1000条评论")

        # 使用TF-IDF提取关键词
        print("\n使用TF-IDF提取关键词...")
        tfidf_keywords = self.tfidf_extraction(all_comments, top_k)

        # 使用TextRank提取关键词
        print("使用TextRank提取关键词...")
        textrank_keywords = self.textrank_extraction(all_comments, top_k)

        # 使用LDA提取关键词
        print("使用LDA提取关键词...")
        lda_keywords, lda_topics = self.lda_extraction(all_comments, num_topics=5, top_k=top_k)

        # 统计所有关键词
        all_keywords = []
        for keywords in tfidf_keywords:
            all_keywords.extend([word for word, _ in keywords])

        # 生成词云
        self.generate_wordcloud(all_keywords)

        # 可视化关键词提取结果
        self.visualize_keywords(tfidf_keywords, textrank_keywords, lda_topics)

        return {
            'tfidf': tfidf_keywords,
            'textrank': textrank_keywords,
            'lda': lda_keywords,
            'lda_topics': lda_topics
        }

    def extract_keywords_with_data(self, comments, labels):
        """从直接传入的评论数据中提取关键词"""
        print(f"使用直接传入的 {len(comments)} 条评论进行关键词提取")

        # 限制评论数量，以加快处理速度
        if len(comments) > 1000:
            indices = np.random.choice(len(comments), 1000, replace=False)
            comments = [comments[i] for i in indices]
            print(f"为加快处理速度，使用随机采样的1000条评论")

        # 使用TF-IDF提取关键词
        print("\n使用TF-IDF提取关键词...")
        tfidf_keywords = self.tfidf_extraction(comments, 10)

        # 使用TextRank提取关键词
        print("使用TextRank提取关键词...")
        textrank_keywords = self.textrank_extraction(comments, 10)

        # 使用LDA提取关键词
        print("使用LDA提取关键词...")
        lda_keywords, lda_topics = self.lda_extraction(comments, num_topics=5, top_k=10)

        # 统计所有关键词
        all_keywords = []
        for keywords in tfidf_keywords:
            all_keywords.extend([word for word, _ in keywords])

        # 生成词云
        self.generate_wordcloud(all_keywords)

        # 可视化关键词提取结果
        self.visualize_keywords(tfidf_keywords, textrank_keywords, lda_topics)

        return {
            'tfidf': tfidf_keywords,
            'textrank': textrank_keywords,
            'lda': lda_keywords,
            'lda_topics': lda_topics
        }

    def generate_wordcloud(self, keywords):
        """生成词云"""
        if not WORDCLOUD_AVAILABLE:
            print("Warning: wordcloud not available. Skipping word cloud generation.")
            return
            
        # 统计关键词频率
        keyword_counts = Counter(keywords)

        try:
            # 尝试使用不同的字体路径
            alibaba_fonts = [
                '/Users/xingqiangchen/Library/Fonts/AlibabaPuHuiTi-3-85-Bold.ttf',
                '/Users/xingqiangchen/Library/Fonts/AlibabaPuHuiTi-2-85-Bold.ttf',
                '/Users/xingqiangchen/Library/Fonts/Alibaba PuHuiTi Bold.OTF',
                '/Users/xingqiangchen/Library/Fonts/Alibaba Sans Bold.OTF',
                '/System/Library/Fonts/PingFang.ttc'  # 备选系统字体
            ]

            # 查找第一个存在的字体
            font_path = None
            for font in alibaba_fonts:
                if os.path.exists(font):
                    font_path = font
                    print(f"使用字体: {font}")
                    break

            if not font_path:
                print("未找到可用字体，使用默认字体")

            # 创建词云
            wordcloud = WordCloud(
                font_path=font_path,  # 使用找到的字体
                width=800,
                height=400,
                background_color='white',
                max_words=100,
                collocations=False  # 避免重复词组
            )

            # 生成词云
            wordcloud.generate_from_frequencies(keyword_counts)

            # 保存词云图
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig('keywords_wordcloud.png')
            plt.close()

            print("词云图生成成功：keywords_wordcloud.png")
        except Exception as e:
            print(f"词云图生成失败: {e}")
            print("跳过词云图生成，继续其他分析")

    def visualize_keywords(self, tfidf_keywords, textrank_keywords, lda_topics):
        """可视化关键词提取结果"""
        # 提取每种方法的前10个关键词
        tfidf_top = {}
        for keywords in tfidf_keywords[:10]:
            for word, score in keywords:
                if word in tfidf_top:
                    tfidf_top[word] += score
                else:
                    tfidf_top[word] = score

        textrank_top = {}
        for keywords in textrank_keywords[:10]:
            for word, score in keywords:
                if word in textrank_top:
                    textrank_top[word] += score
                else:
                    textrank_top[word] = score

        # 从LDA主题中提取关键词
        lda_top = {}
        for topic_id, topic in lda_topics:
            # 解析主题字符串，提取关键词和权重
            words = re.findall(r'"(.*?)"', topic)
            for word in words:
                if word in lda_top:
                    lda_top[word] += 1
                else:
                    lda_top[word] = 1

        # 按权重排序
        tfidf_top = sorted(tfidf_top.items(), key=lambda x: x[1], reverse=True)[:10]
        textrank_top = sorted(textrank_top.items(), key=lambda x: x[1], reverse=True)[:10]
        lda_top = sorted(lda_top.items(), key=lambda x: x[1], reverse=True)[:10]

        # 可视化
        fig, axes = plt.subplots(3, 1, figsize=(12, 18))

        # TF-IDF关键词
        axes[0].barh([word for word, _ in tfidf_top], [score for _, score in tfidf_top])
        axes[0].set_title('TF-IDF关键词')
        axes[0].set_xlabel('权重')
        axes[0].set_ylabel('关键词')
        axes[0].invert_yaxis()  # 倒序显示，使最高分的在顶部

        # TextRank关键词
        axes[1].barh([word for word, _ in textrank_top], [score for _, score in textrank_top])
        axes[1].set_title('TextRank关键词')
        axes[1].set_xlabel('权重')
        axes[1].set_ylabel('关键词')
        axes[1].invert_yaxis()

        # LDA关键词
        axes[2].barh([word for word, _ in lda_top], [score for _, score in lda_top])
        axes[2].set_title('LDA主题关键词')
        axes[2].set_xlabel('出现次数')
        axes[2].set_ylabel('关键词')
        axes[2].invert_yaxis()

        plt.tight_layout()
        plt.savefig('keyword_extraction_comparison.png')
        plt.close()

    def map_keywords_to_indicators(self, keywords, indicator_mapping):
        """将关键词映射到评价指标"""
        indicator_scores = {}

        for word, score in keywords:
            for indicator, related_words in indicator_mapping.items():
                if word in related_words:
                    if indicator in indicator_scores:
                        indicator_scores[indicator] += score
                    else:
                        indicator_scores[indicator] = score

        return indicator_scores

def main():
    # 创建关键词提取对象
    extractor = KeywordExtraction()

    # 获取评论文件列表
    comments_files = []
    for category in ['好评', '中评', '差评']:
        files = glob.glob(f"comments/*{category}*.xls") + glob.glob(f"comments/*{category}*.xlsx")
        # 限制每类评论的文件数量，以加快处理速度
        comments_files.extend(files[:2])

    # 提取关键词
    results = extractor.extract_keywords(comments_files)

    # 定义评价指标映射（示例）
    indicator_mapping = {
        '原料质量': ['新鲜', '优质', '原料', '品质', '材料', '质量', '好', '差'],
        '加工工艺': ['工艺', '加工', '制作', '精细', '粗糙', '做工', '技术', '生产'],
        '物流配送': ['物流', '配送', '快递', '送货', '包装', '运输', '速度', '及时', '慢'],
        '售后服务': ['售后', '服务', '态度', '解决', '问题', '退货', '换货', '客服', '响应'],
        '信息透明度': ['信息', '透明', '描述', '详细', '说明', '介绍', '标注', '标签', '虚假']
    }

    # 将关键词映射到评价指标
    print("\n将关键词映射到评价指标...")

    # 使用TF-IDF关键词进行映射
    tfidf_indicators = {}
    for doc_keywords in results['tfidf']:
        doc_indicators = extractor.map_keywords_to_indicators(doc_keywords, indicator_mapping)
        for indicator, score in doc_indicators.items():
            if indicator in tfidf_indicators:
                tfidf_indicators[indicator] += score
            else:
                tfidf_indicators[indicator] = score

    # 打印映射结果
    print("\nTF-IDF关键词映射到评价指标的结果:")
    for indicator, score in sorted(tfidf_indicators.items(), key=lambda x: x[1], reverse=True):
        print(f"{indicator}: {score:.4f}")

    print("\n分析完成！结果已保存到 keywords_wordcloud.png 和 keyword_extraction_comparison.png")

if __name__ == "__main__":
    main()