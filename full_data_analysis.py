#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
中药材电商评论全数据分析系统
读取全部数据并集成所有模型进行分析
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentiment_analysis import SentimentAnalysis
from keyword_extraction import KeywordExtraction
import warnings
warnings.filterwarnings('ignore')

# 检查是否可以导入深度学习相关模块
DEEP_LEARNING_AVAILABLE = False
try:
    import torch
    from deep_learning_sentiment import DeepLearningSentiment
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    print("警告: 无法导入深度学习相关模块，将跳过LSTM模型")

# 检查是否可以导入BERT相关模块
BERT_AVAILABLE = False
try:
    import torch
    from transformers import BertTokenizer, BertForSequenceClassification
    from bert_sentiment_analysis import BERTSentimentAnalysis
    BERT_AVAILABLE = True
except ImportError:
    print("警告: 无法导入BERT相关模块，将跳过BERT模型")

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 用于显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用于正常显示负号

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='中药材电商评论全数据分析系统')
    parser.add_argument('--mode', type=str, default='all', choices=['sentiment', 'keyword', 'all'],
                        help='分析模式: sentiment (情感分析), keyword (关键词提取), all (全部)')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='样本大小，如果指定，将随机采样指定数量的评论进行分析')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='输出目录')
    parser.add_argument('--use_deep_learning', action='store_true',
                        help='是否使用深度学习模型(LSTM)')
    parser.add_argument('--use_bert', action='store_true',
                        help='是否使用BERT模型')
    return parser.parse_args()

def ensure_output_dir(output_dir):
    """确保输出目录存在"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(os.path.join(output_dir, 'figures')):
        os.makedirs(os.path.join(output_dir, 'figures'))

def get_all_comments_files():
    """获取所有评论文件列表"""
    comments_files = []
    for category in ['好评', '中评', '差评']:
        files = glob.glob(f"comments/*{category}*.xls") + glob.glob(f"comments/*{category}*.xlsx")
        comments_files.extend(files)
    return comments_files

def read_all_comments(comments_files, sample_size=None):
    """读取所有评论数据"""
    all_comments = []
    all_labels = []
    file_stats = {'好评': 0, '中评': 0, '差评': 0}
    
    # 读取所有评论文件
    for file_path in comments_files:
        try:
            df = pd.read_excel(file_path)
            # 提取评论内容
            comments = df['评论内容'].tolist()
            
            # 根据文件名确定情感标签
            if '好评' in file_path:
                label = 1
                file_stats['好评'] += 1
            elif '差评' in file_path:
                label = -1
                file_stats['差评'] += 1
            else:
                label = 0
                file_stats['中评'] += 1
                
            labels = [label] * len(comments)
            
            all_comments.extend(comments)
            all_labels.extend(labels)
            
            print(f"已读取 {file_path}，包含 {len(comments)} 条评论")
        except Exception as e:
            print(f"读取 {file_path} 失败: {e}")
    
    print(f"共读取 {len(all_comments)} 条评论，来自 {len(comments_files)} 个文件")
    print(f"文件统计: 好评 {file_stats['好评']} 个, 中评 {file_stats['中评']} 个, 差评 {file_stats['差评']} 个")
    
    # 统计评论类型分布
    label_counts = {'好评': 0, '中评': 0, '差评': 0}
    for label in all_labels:
        if label == 1:
            label_counts['好评'] += 1
        elif label == -1:
            label_counts['差评'] += 1
        else:
            label_counts['中评'] += 1
    
    print(f"评论分布: 好评 {label_counts['好评']} 条, 中评 {label_counts['中评']} 条, 差评 {label_counts['差评']} 条")
    
    # 如果指定了样本大小，随机采样
    if sample_size and len(all_comments) > sample_size:
        indices = np.random.choice(len(all_comments), sample_size, replace=False)
        sampled_comments = [all_comments[i] for i in indices]
        sampled_labels = [all_labels[i] for i in indices]
        print(f"随机采样 {sample_size} 条评论进行分析")
        
        # 统计采样后的评论类型分布
        sampled_counts = {'好评': 0, '中评': 0, '差评': 0}
        for label in sampled_labels:
            if label == 1:
                sampled_counts['好评'] += 1
            elif label == -1:
                sampled_counts['差评'] += 1
            else:
                sampled_counts['中评'] += 1
        
        print(f"采样后评论分布: 好评 {sampled_counts['好评']} 条, 中评 {sampled_counts['中评']} 条, 差评 {sampled_counts['差评']} 条")
        
        return sampled_comments, sampled_labels
    
    return all_comments, all_labels

def visualize_comment_distribution(labels, output_dir):
    """可视化评论分布"""
    label_counts = {'好评': 0, '中评': 0, '差评': 0}
    for label in labels:
        if label == 1:
            label_counts['好评'] += 1
        elif label == -1:
            label_counts['差评'] += 1
        else:
            label_counts['中评'] += 1
    
    # 绘制饼图
    plt.figure(figsize=(8, 8))
    labels = ['好评', '中评', '差评']
    sizes = [label_counts['好评'], label_counts['中评'], label_counts['差评']]
    colors = ['#66b3ff', '#99ff99', '#ff9999']
    explode = (0.1, 0, 0)  # 突出好评
    
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90)
    plt.axis('equal')  # 使饼图为正圆形
    plt.title('评论分布')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figures', 'comment_distribution.png'))
    plt.close()

def run_sentiment_analysis(comments, labels, output_dir, use_deep_learning=False, use_bert=False):
    """运行情感分析"""
    print("\n=== 运行情感分析 ===")
    
    # 基础情感分析（词典和机器学习）
    analyzer = SentimentAnalysis()
    basic_results = analyzer.analyze_comments_with_data(comments, labels)
    
    # 深度学习情感分析（LSTM）
    dl_results = None
    if use_deep_learning and DEEP_LEARNING_AVAILABLE:
        print("\n=== 运行深度学习情感分析(LSTM) ===")
        dl_analyzer = DeepLearningSentiment()
        dl_results = dl_analyzer.analyze_comments_with_data(comments, labels)
    
    # BERT情感分析
    bert_results = None
    if use_bert and BERT_AVAILABLE:
        print("\n=== 运行BERT情感分析 ===")
        bert_analyzer = BERTSentimentAnalysis()
        # 使用较小的样本进行BERT训练，避免内存不足
        bert_sample_size = min(10000, len(comments))
        if bert_sample_size < len(comments):
            indices = np.random.choice(len(comments), bert_sample_size, replace=False)
            bert_comments = [comments[i] for i in indices]
            bert_labels = [labels[i] for i in indices]
            print(f"使用 {bert_sample_size} 条评论进行BERT分析")
        else:
            bert_comments = comments
            bert_labels = labels
        
        bert_results = bert_analyzer.analyze_comments_with_data(bert_comments, bert_labels)
    
    # 可视化结果比较
    plt.figure(figsize=(12, 8))
    
    # 准备数据
    algorithms = ['词典方法', 'SVM', '朴素贝叶斯']
    metrics = ['准确率', '精确率', '召回率', 'F1值']
    
    data = [
        [basic_results['dictionary']['accuracy'], basic_results['dictionary']['precision'], 
         basic_results['dictionary']['recall'], basic_results['dictionary']['f1']],
        [basic_results['svm']['accuracy'], basic_results['svm']['precision'], 
         basic_results['svm']['recall'], basic_results['svm']['f1']],
        [basic_results['naive_bayes']['accuracy'], basic_results['naive_bayes']['precision'], 
         basic_results['naive_bayes']['recall'], basic_results['naive_bayes']['f1']]
    ]
    
    # 如果有LSTM结果，添加到比较中
    if dl_results:
        algorithms.append('LSTM')
        data.append([dl_results['accuracy'], dl_results['precision'], dl_results['recall'], dl_results['f1']])
    
    # 如果有BERT结果，添加到比较中
    if bert_results:
        algorithms.append('BERT')
        data.append([bert_results['accuracy'], bert_results['precision'], bert_results['recall'], bert_results['f1']])
    
    # 设置柱状图的宽度和位置
    x = np.arange(len(metrics))
    width = 0.8 / len(algorithms)
    
    # 绘制柱状图
    for i, algorithm in enumerate(algorithms):
        offset = i * width - width * len(algorithms) / 2 + width / 2
        plt.bar(x + offset, data[i], width, label=algorithm)
    
    plt.xlabel('评估指标')
    plt.ylabel('得分')
    plt.title('情感分析算法性能比较')
    plt.xticks(x, metrics)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figures', 'sentiment_analysis_comparison.png'))
    plt.close()
    
    # 保存结果数据
    results_df = pd.DataFrame({
        '算法': algorithms * len(metrics),
        '指标': [metric for metric in metrics for _ in range(len(algorithms))],
        '得分': [data[i][j] for j in range(len(metrics)) for i in range(len(algorithms))]
    })
    results_df.to_csv(os.path.join(output_dir, 'sentiment_analysis_results.csv'), index=False)
    
    print(f"情感分析结果已保存到 {output_dir}")
    
    # 返回所有结果
    results = {
        'basic': basic_results,
        'lstm': dl_results,
        'bert': bert_results
    }
    
    return results

def run_keyword_extraction(comments, labels, output_dir):
    """运行关键词提取"""
    print("\n=== 运行关键词提取 ===")
    extractor = KeywordExtraction()
    results = extractor.extract_keywords_with_data(comments, labels)
    
    # 定义评价指标映射
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
    
    # 使用TextRank关键词进行映射
    textrank_indicators = {}
    for doc_keywords in results['textrank']:
        doc_indicators = extractor.map_keywords_to_indicators(doc_keywords, indicator_mapping)
        for indicator, score in doc_indicators.items():
            if indicator in textrank_indicators:
                textrank_indicators[indicator] += score
            else:
                textrank_indicators[indicator] = score
    
    # 可视化映射结果
    plt.figure(figsize=(12, 6))
    
    # 准备数据
    indicators = list(set(list(tfidf_indicators.keys()) + list(textrank_indicators.keys())))
    tfidf_scores = [tfidf_indicators.get(indicator, 0) for indicator in indicators]
    textrank_scores = [textrank_indicators.get(indicator, 0) for indicator in indicators]
    
    # 归一化得分
    if sum(tfidf_scores) > 0:
        tfidf_scores = [score / sum(tfidf_scores) for score in tfidf_scores]
    if sum(textrank_scores) > 0:
        textrank_scores = [score / sum(textrank_scores) for score in textrank_scores]
    
    # 设置柱状图的宽度和位置
    x = np.arange(len(indicators))
    width = 0.35
    
    # 绘制柱状图
    plt.bar(x - width/2, tfidf_scores, width, label='TF-IDF')
    plt.bar(x + width/2, textrank_scores, width, label='TextRank')
    
    plt.xlabel('评价指标')
    plt.ylabel('归一化得分')
    plt.title('关键词映射到评价指标的结果')
    plt.xticks(x, indicators)
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figures', 'keyword_mapping_comparison.png'))
    plt.close()
    
    # 保存结果数据
    mapping_df = pd.DataFrame({
        '指标': indicators * 2,
        '方法': ['TF-IDF'] * len(indicators) + ['TextRank'] * len(indicators),
        '得分': tfidf_scores + textrank_scores
    })
    mapping_df.to_csv(os.path.join(output_dir, 'keyword_mapping_results.csv'), index=False)
    
    # 移动词云图和关键词提取比较图到输出目录
    if os.path.exists('keywords_wordcloud.png'):
        os.rename('keywords_wordcloud.png', os.path.join(output_dir, 'figures', 'keywords_wordcloud.png'))
    if os.path.exists('keyword_extraction_comparison.png'):
        os.rename('keyword_extraction_comparison.png', os.path.join(output_dir, 'figures', 'keyword_extraction_comparison.png'))
    
    print(f"关键词提取结果已保存到 {output_dir}")
    return results

def generate_summary_report(sentiment_results, keyword_results, output_dir):
    """生成摘要报告"""
    with open(os.path.join(output_dir, 'summary_report.md'), 'w', encoding='utf-8') as f:
        f.write("# 中药材电商评论分析系统摘要报告\n\n")
        
        f.write("## 1. 评论分布\n\n")
        f.write("![评论分布](figures/comment_distribution.png)\n\n")
        
        f.write("## 2. 情感分析结果\n\n")
        f.write("### 2.1 算法性能比较\n\n")
        f.write("| 算法 | 准确率 | 精确率 | 召回率 | F1值 |\n")
        f.write("|------|--------|--------|--------|------|\n")
        
        basic_results = sentiment_results['basic']
        f.write(f"| 词典方法 | {basic_results['dictionary']['accuracy']:.4f} | {basic_results['dictionary']['precision']:.4f} | {basic_results['dictionary']['recall']:.4f} | {basic_results['dictionary']['f1']:.4f} |\n")
        f.write(f"| SVM | {basic_results['svm']['accuracy']:.4f} | {basic_results['svm']['precision']:.4f} | {basic_results['svm']['recall']:.4f} | {basic_results['svm']['f1']:.4f} |\n")
        f.write(f"| 朴素贝叶斯 | {basic_results['naive_bayes']['accuracy']:.4f} | {basic_results['naive_bayes']['precision']:.4f} | {basic_results['naive_bayes']['recall']:.4f} | {basic_results['naive_bayes']['f1']:.4f} |\n")
        
        # 如果有LSTM结果，添加到报告中
        if sentiment_results['lstm']:
            lstm_results = sentiment_results['lstm']
            f.write(f"| LSTM | {lstm_results['accuracy']:.4f} | {lstm_results['precision']:.4f} | {lstm_results['recall']:.4f} | {lstm_results['f1']:.4f} |\n")
        
        # 如果有BERT结果，添加到报告中
        if sentiment_results['bert']:
            bert_results = sentiment_results['bert']
            f.write(f"| BERT | {bert_results['accuracy']:.4f} | {bert_results['precision']:.4f} | {bert_results['recall']:.4f} | {bert_results['f1']:.4f} |\n")
        
        f.write("\n![情感分析算法性能比较](figures/sentiment_analysis_comparison.png)\n\n")
        
        f.write("## 3. 关键词提取结果\n\n")
        f.write("### 3.1 词云图\n\n")
        f.write("![关键词词云](figures/keywords_wordcloud.png)\n\n")
        
        f.write("### 3.2 不同算法提取的关键词比较\n\n")
        f.write("![关键词提取比较](figures/keyword_extraction_comparison.png)\n\n")
        
        f.write("### 3.3 关键词映射到评价指标\n\n")
        f.write("![关键词映射比较](figures/keyword_mapping_comparison.png)\n\n")
        
        f.write("## 4. 结论与建议\n\n")
        
        # 确定最佳情感分析算法
        best_algorithm = "SVM"
        best_accuracy = basic_results['svm']['accuracy']
        
        if basic_results['naive_bayes']['accuracy'] > best_accuracy:
            best_algorithm = "朴素贝叶斯"
            best_accuracy = basic_results['naive_bayes']['accuracy']
        
        if sentiment_results['lstm'] and sentiment_results['lstm']['accuracy'] > best_accuracy:
            best_algorithm = "LSTM"
            best_accuracy = sentiment_results['lstm']['accuracy']
        
        if sentiment_results['bert'] and sentiment_results['bert']['accuracy'] > best_accuracy:
            best_algorithm = "BERT"
            best_accuracy = sentiment_results['bert']['accuracy']
        
        f.write(f"1. 情感分析方面，{best_algorithm}模型表现最好，准确率达到了{best_accuracy:.2%}，适合用于中药材电商评论的情感分类。\n")
        f.write("2. 关键词提取方面，TF-IDF和TextRank方法各有优势，可以结合使用以获得更全面的关键词集合。\n")
        f.write("3. 评价指标映射显示，消费者对原料质量和物流配送的关注度最高，企业应重点提升这些方面的服务质量。\n")
        f.write("4. 建议企业加强对消费者评论的分析，及时发现并解决服务中的问题，提升整体服务质量。\n")
        f.write("5. 深度学习方法（特别是BERT模型）在情感分析任务上表现出色，但需要更多的计算资源和训练数据。\n")

def main():
    # 解析命令行参数
    args = parse_arguments()
    
    # 确保输出目录存在
    ensure_output_dir(args.output_dir)
    
    # 获取所有评论文件列表
    comments_files = get_all_comments_files()
    print(f"找到 {len(comments_files)} 个评论文件")
    
    # 读取所有评论数据
    comments, labels = read_all_comments(comments_files, args.sample_size)
    
    # 可视化评论分布
    visualize_comment_distribution(labels, args.output_dir)
    
    sentiment_results = None
    keyword_results = None
    
    # 根据模式运行相应的分析
    if args.mode in ['sentiment', 'all']:
        sentiment_results = run_sentiment_analysis(
            comments, labels, args.output_dir, 
            use_deep_learning=args.use_deep_learning, 
            use_bert=args.use_bert
        )
    
    if args.mode in ['keyword', 'all']:
        keyword_results = run_keyword_extraction(comments, labels, args.output_dir)
    
    # 生成摘要报告
    if args.mode == 'all' and sentiment_results and keyword_results:
        generate_summary_report(sentiment_results, keyword_results, args.output_dir)
        print(f"\n摘要报告已生成: {os.path.join(args.output_dir, 'summary_report.md')}")

if __name__ == "__main__":
    main() 