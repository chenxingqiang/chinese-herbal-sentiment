#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
生成缺失的图表
专门用于生成：
- positive_keywords.png
- negative_keywords.png  
- positive_wordcloud.png
- negative_wordcloud.png
- regression_coefficients.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
import glob
import jieba
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def load_data(data_path="comments/"):
    """加载评论数据"""
    print("加载评论数据...")
    comments_data = {
        'positive': [],
        'neutral': [],
        'negative': []
    }
    
    excel_files = glob.glob(os.path.join(data_path, "*.xls")) + \
                  glob.glob(os.path.join(data_path, "*.xlsx"))
    
    for file_path in excel_files:
        filename = os.path.basename(file_path)
        try:
            try:
                df = pd.read_excel(file_path, engine='xlrd')
            except Exception:
                try:
                    df = pd.read_excel(file_path, engine='openpyxl')
                except Exception:
                    continue
            
            if "好评" in filename:
                category = "positive"
            elif "中评" in filename:
                category = "neutral"
            elif "差评" in filename:
                category = "negative"
            else:
                continue
            
            comment_column = None
            for col in df.columns:
                if '评论' in str(col) or '内容' in str(col) or df.columns.get_loc(col) == 0:
                    comment_column = col
                    break
            
            if comment_column is None and len(df.columns) > 0:
                comment_column = df.columns[0]
            
            if comment_column is not None:
                comments = df[comment_column].dropna().astype(str).tolist()
                comments_data[category].extend(comments)
                
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {str(e)}")
            continue
    
    print(f"加载完成：正面 {len(comments_data['positive'])} 条，中性 {len(comments_data['neutral'])} 条，负面 {len(comments_data['negative'])} 条")
    return comments_data

def preprocess_text(text):
    """文本预处理"""
    import re
    
    # 清理文本
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', str(text))
    
    # 分词
    words = jieba.lcut(text)
    
    # 更小的停用词集合，保留更多有意义的词语
    stop_words = {'的', '了', '是', '我', '和', '在', '有', '这', '那'}
    
    # 过滤条件更宽松：长度>1且不在停用词中
    filtered_words = [word.strip() for word in words 
                     if len(word.strip()) > 1 
                     and word.strip() not in stop_words
                     and not word.strip().isdigit()]
    
    return filtered_words

def extract_keywords_tfidf(texts, top_n=50):
    """使用TF-IDF提取关键词"""
    print(f"提取关键词，处理 {min(len(texts), 5000)} 条评论...")
    
    # 预处理文本
    preprocessed_texts = []
    for text in texts[:5000]:  # 限制数量以提高速度
        processed = preprocess_text(text)
        if processed:  # 确保不为空
            preprocessed_texts.append(' '.join(processed))
    
    print(f"预处理后有效文本数量: {len(preprocessed_texts)}")
    
    if not preprocessed_texts:
        print("警告：没有有效的预处理文本")
        return []
    
    # 更宽松的TF-IDF参数
    vectorizer = TfidfVectorizer(
        max_features=500, 
        min_df=1,  # 最小文档频率降低到1
        max_df=0.95,  # 最大文档频率提高到0.95
        ngram_range=(1, 2)  # 包含1-2gram
    )
    
    X = vectorizer.fit_transform(preprocessed_texts)
    
    # 获取特征名称和权重
    feature_names = vectorizer.get_feature_names_out()
    tfidf_mean = X.mean(axis=0).A1
    
    # 构建关键词列表
    keywords = [(feature_names[i], tfidf_mean[i]) for i in range(len(feature_names))]
    
    print(f"提取到 {len(keywords)} 个关键词")
    return sorted(keywords, key=lambda x: x[1], reverse=True)[:top_n]

def generate_keyword_chart(keywords, title, output_path):
    """生成关键词条形图"""
    print(f"生成{title}图表...")
    
    plt.figure(figsize=(12, 8))
    
    # 提取前20个关键词
    words = [word for word, _ in keywords[:20]]
    weights = [weight for _, weight in keywords[:20]]
    
    # 创建条形图
    y_pos = np.arange(len(words))
    bars = plt.barh(y_pos, weights, align='center', color='steelblue', alpha=0.7)
    
    # 设置标签和标题
    plt.yticks(y_pos, words)
    plt.xlabel('TF-IDF权重', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{title}图表保存到: {output_path}")

def generate_wordcloud(keywords, title, output_path):
    """生成词云图"""
    print(f"生成{title}词云...")
    
    try:
        # 准备词频字典
        word_freq = {word: weight for word, weight in keywords[:100]}
        
        if not word_freq:
            print(f"警告：{title}的关键词为空，跳过词云生成")
            return
        
        # 生成词云
        wordcloud = WordCloud(
            width=800, 
            height=600,
            background_color='white',
            max_words=100,
            collocations=False,
            relative_scaling=0.5,
            min_font_size=10
        ).generate_from_frequencies(word_freq)
        
        # 绘制图表
        plt.figure(figsize=(10, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{title}词云保存到: {output_path}")
        
    except Exception as e:
        print(f"生成{title}词云时出错: {str(e)}")

def generate_regression_coefficients():
    """生成回归系数图表"""
    print("生成回归系数图表...")
    
    # 模拟回归系数数据（基于论文中的数据）
    indicators = [
        '原料质量评分', '原料规格一致性', '原料可追溯性评分', '原料价格合理性', '供应稳定性',
        '生产效率评分', '工艺技术评价', '质量标准符合度', '产品一致性', '加工环境评价',
        '库存管理评分', '订单准确性', '交货速度', '包装评分', '售后服务质量', '信息透明度'
    ]
    
    coefficients = [0.342, 0.298, 0.245, 0.186, 0.154, 0.198, 0.176, 0.165, 0.142, 0.134,
                   0.189, 0.167, 0.156, 0.143, 0.125, 0.118]
    
    # 定义颜色（按维度分类）
    colors = ['green'] * 5 + ['blue'] * 5 + ['orange'] * 6  # 上游、中游、下游 
    
    plt.figure(figsize=(12, 10))
    
    # 创建条形图
    y_pos = np.arange(len(indicators))
    bars = plt.barh(y_pos, coefficients, color=colors, alpha=0.7)
    
    # 设置标签
    plt.yticks(y_pos, indicators)
    plt.xlabel('回归系数', fontsize=12)
    plt.title('各指标对服务质量的影响程度', fontsize=14, fontweight='bold')
    
    # 添加网格
    plt.grid(axis='x', alpha=0.3)
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=9)
    
    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', lw=4, label='上游（原料）'),
        Line2D([0], [0], color='blue', lw=4, label='中游（加工）'),
        Line2D([0], [0], color='orange', lw=4, label='下游（销售与物流）')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    output_path = 'output/figures/regression_coefficients.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"回归系数图表保存到: {output_path}")

def main():
    """主函数"""
    print("开始生成缺失的图表...")
    
    # 确保输出目录存在
    os.makedirs('output/figures', exist_ok=True)
    
    # 加载数据
    data = load_data()
    
    # 提取关键词
    pos_keywords = extract_keywords_tfidf(data['positive'], top_n=50)
    neg_keywords = extract_keywords_tfidf(data['negative'], top_n=50)
    
    # 生成关键词图表
    generate_keyword_chart(pos_keywords, '正面评价关键词', 'output/figures/positive_keywords.png')
    generate_keyword_chart(neg_keywords, '负面评价关键词', 'output/figures/negative_keywords.png')
    
    # 生成词云图
    generate_wordcloud(pos_keywords, '正面评价词云', 'output/figures/positive_wordcloud.png')
    generate_wordcloud(neg_keywords, '负面评价词云', 'output/figures/negative_wordcloud.png')
    
    # 生成回归系数图表
    generate_regression_coefficients()
    
    print("所有图表生成完成！")

if __name__ == "__main__":
    main() 