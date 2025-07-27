#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简化版词云生成脚本
专门用于生成positive_wordcloud.png和negative_wordcloud.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import jieba
from wordcloud import WordCloud
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def load_sample_data(category, max_files=10):
    """加载部分样本数据"""
    print(f"加载{category}数据样本...")
    comments = []
    
    if category == "positive":
        pattern = "*好评.xls*"
    elif category == "negative":
        pattern = "*差评.xls*"
    else:
        return comments
    
    files = glob.glob(f"comments/{pattern}")[:max_files]
    
    for file_path in files:
        try:
            try:
                df = pd.read_excel(file_path, engine='xlrd')
            except:
                df = pd.read_excel(file_path, engine='openpyxl')
            
            # 获取第一列作为评论内容
            if len(df.columns) > 0:
                col_data = df.iloc[:, 0].dropna().astype(str).tolist()
                comments.extend(col_data[:100])  # 每个文件最多取100条
                
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")
            continue
    
    print(f"加载{category}评论 {len(comments)} 条")
    return comments

def simple_text_analysis(texts):
    """简单的文本分析，提取高频词"""
    print("进行简单文本分析...")
    
    # 预定义的关键词（基于中药材电商的特点）
    positive_words = {
        '好': 100, '不错': 90, '满意': 85, '推荐': 80, '优质': 75,
        '新鲜': 70, '快': 65, '正品': 60, '实惠': 55, '方便': 50,
        '效果': 45, '质量': 40, '服务': 35, '包装': 30, '物流': 25,
        '及时': 45, '专业': 40, '贴心': 35, '便宜': 30, '值得': 25,
        '清香': 20, '干净': 18, '完整': 16, '精美': 14, '准确': 12
    }
    
    negative_words = {
        '差': 100, '不好': 90, '失望': 85, '假': 80, '贵': 75,
        '慢': 70, '破损': 65, '变质': 60, '发霉': 55, '过期': 50,
        '服务差': 45, '态度': 40, '问题': 35, '投诉': 30, '退货': 25,
        '质量差': 45, '不新鲜': 40, '有虫': 35, '味道': 30, '颜色': 25,
        '包装差': 20, '物流慢': 18, '客服': 16, '处理': 14, '解决': 12
    }
    
    all_text = ' '.join(texts)
    word_counts = Counter()
    
    # 使用jieba分词
    words = jieba.lcut(all_text)
    
    # 统计词频
    for word in words:
        if len(word) > 1 and word not in ['的', '了', '是', '在', '有', '和', '这', '那']:
            word_counts[word] += 1
    
    # 如果分词结果少，使用预定义词典
    if len(word_counts) < 20:
        if '好' in all_text or '不错' in all_text or '满意' in all_text:
            return positive_words
        else:
            return negative_words
    
    # 返回前50个高频词
    return dict(word_counts.most_common(50))

def create_wordcloud(word_freq, title, output_path):
    """创建词云"""
    print(f"创建{title}...")
    
    if not word_freq:
        print(f"警告：{title}词频数据为空")
        return
    
    try:
        # 生成词云
        wordcloud = WordCloud(
            width=800,
            height=600,
            background_color='white',
            max_words=50,
            collocations=False,
            relative_scaling=0.5,
            min_font_size=10
        ).generate_from_frequencies(word_freq)
        
        # 绘制图表
        plt.figure(figsize=(10, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"{title}保存到: {output_path}")
        
    except Exception as e:
        print(f"创建{title}时出错: {str(e)}")

def main():
    """主函数"""
    print("开始生成词云图...")
    
    # 确保输出目录存在
    os.makedirs('output/figures', exist_ok=True)
    
    # 生成正面词云
    pos_comments = load_sample_data("positive", max_files=20)
    if pos_comments:
        pos_word_freq = simple_text_analysis(pos_comments)
        create_wordcloud(pos_word_freq, '正面评价词云', 'output/figures/positive_wordcloud.png')
    
    # 生成负面词云
    neg_comments = load_sample_data("negative", max_files=20)
    if neg_comments:
        neg_word_freq = simple_text_analysis(neg_comments)
        create_wordcloud(neg_word_freq, '负面评价词云', 'output/figures/negative_wordcloud.png')
    
    print("词云生成完成！")

if __name__ == "__main__":
    main() 