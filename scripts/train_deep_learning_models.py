#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
训练深度学习模型（LSTM和BERT）
用于中药材电商评论情感分析
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练深度学习模型（LSTM和BERT）')
    parser.add_argument('--model', type=str, default='both', choices=['lstm', 'bert', 'both'],
                        help='要训练的模型: lstm, bert, both (两者)')
    parser.add_argument('--sample_size', type=int, default=5000,
                        help='样本大小，随机采样指定数量的评论进行训练')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='输出目录')
    parser.add_argument('--epochs', type=int, default=3,
                        help='训练轮次')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批量大小')
    return parser.parse_args()

def ensure_output_dir(output_dir):
    """确保输出目录存在"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(os.path.join(output_dir, 'figures')):
        os.makedirs(os.path.join(output_dir, 'figures'))
    if not os.path.exists(os.path.join(output_dir, 'models')):
        os.makedirs(os.path.join(output_dir, 'models'))

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
    labels_text = ['好评', '中评', '差评']
    sizes = [label_counts['好评'], label_counts['中评'], label_counts['差评']]
    colors = ['#66b3ff', '#99ff99', '#ff9999']
    explode = (0.1, 0, 0)  # 突出好评
    
    plt.pie(sizes, explode=explode, labels=labels_text, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90)
    plt.axis('equal')  # 使饼图为正圆形
    plt.title('评论分布')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figures', 'comment_distribution.png'))
    plt.close()

def train_lstm_model(comments, labels, output_dir, epochs=3, batch_size=32):
    """训练LSTM模型"""
    try:
        from deep_learning_sentiment import DeepLearningSentiment
        
        print("\n=== 训练LSTM模型 ===")
        dl_analyzer = DeepLearningSentiment()
        # 修改LSTM模型参数
        dl_analyzer.epochs = epochs
        dl_analyzer.batch_size = batch_size
        
        # 训练模型
        results = dl_analyzer.analyze_comments_with_data(comments, labels)
        
        # 将结果图移到指定输出目录
        if os.path.exists('lstm_training_history.png'):
            os.rename('lstm_training_history.png', 
                     os.path.join(output_dir, 'figures', 'lstm_training_history.png'))
        
        # 将模型移到指定输出目录
        if os.path.exists('lstm_sentiment_model.h5'):
            os.rename('lstm_sentiment_model.h5',
                     os.path.join(output_dir, 'models', 'lstm_sentiment_model.h5'))
        
        print(f"LSTM模型训练完成，准确率: {results['accuracy']:.4f}")
        return results
    except ImportError as e:
        print(f"无法训练LSTM模型: {e}")
        print("请安装必要的依赖: pip install tensorflow")
        return None

def train_bert_model(comments, labels, output_dir, epochs=3, batch_size=16):
    """训练BERT模型"""
    try:
        import torch
        from transformers import BertTokenizer
        from bert_sentiment_analysis import BERTSentimentAnalysis
        
        print("\n=== 训练BERT模型 ===")
        bert_analyzer = BERTSentimentAnalysis()
        # 修改BERT模型参数
        bert_analyzer.epochs = epochs
        bert_analyzer.batch_size = batch_size
        
        # 训练模型
        results = bert_analyzer.analyze_comments_with_data(comments, labels)
        
        # 将结果图移到指定输出目录
        if os.path.exists('bert_training_stats.png'):
            os.rename('bert_training_stats.png', 
                     os.path.join(output_dir, 'figures', 'bert_training_stats.png'))
        
        # 将模型目录移到指定输出目录
        if os.path.exists('bert_sentiment_model'):
            if os.path.exists(os.path.join(output_dir, 'models', 'bert_sentiment_model')):
                import shutil
                shutil.rmtree(os.path.join(output_dir, 'models', 'bert_sentiment_model'))
            os.rename('bert_sentiment_model',
                     os.path.join(output_dir, 'models', 'bert_sentiment_model'))
        
        print(f"BERT模型训练完成，准确率: {results['accuracy']:.4f}")
        return results
    except ImportError as e:
        print(f"无法训练BERT模型: {e}")
        print("请安装必要的依赖: pip install torch transformers")
        return None

def generate_summary_report(lstm_results, bert_results, output_dir):
    """生成摘要报告"""
    report = "# 深度学习模型训练报告\n\n"
    report += f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    report += "## 1. 模型性能比较\n\n"
    report += "| 模型 | 准确率 | 精确率 | 召回率 | F1值 |\n"
    report += "|------|--------|--------|--------|------|\n"
    
    if lstm_results:
        report += f"| LSTM | {lstm_results['accuracy']:.4f} | {lstm_results['precision']:.4f} | {lstm_results['recall']:.4f} | {lstm_results['f1']:.4f} |\n"
    
    if bert_results:
        report += f"| BERT | {bert_results['accuracy']:.4f} | {bert_results['precision']:.4f} | {bert_results['recall']:.4f} | {bert_results['f1']:.4f} |\n"
    
    report += "\n## 2. 详细分类报告\n\n"
    
    if lstm_results:
        report += "### 2.1 LSTM模型\n\n"
        report += "```\n"
        report += lstm_results['report']
        report += "\n```\n\n"
    
    if bert_results:
        report += "### 2.2 BERT模型\n\n"
        report += "```\n"
        report += bert_results['report']
        report += "\n```\n\n"
    
    report += "## 3. 结论\n\n"
    
    # 确定最佳模型
    if lstm_results and bert_results:
        best_model = "BERT" if bert_results['accuracy'] > lstm_results['accuracy'] else "LSTM"
        report += f"根据实验结果，{best_model}模型在中药材评论情感分析任务上表现最好，准确率达到"
        if best_model == "BERT":
            report += f"{bert_results['accuracy']:.4f}，比LSTM模型高出{bert_results['accuracy'] - lstm_results['accuracy']:.4f}。\n\n"
        else:
            report += f"{lstm_results['accuracy']:.4f}，比BERT模型高出{lstm_results['accuracy'] - bert_results['accuracy']:.4f}。\n\n"
    elif lstm_results:
        report += f"LSTM模型在中药材评论情感分析任务上表现良好，准确率达到{lstm_results['accuracy']:.4f}。\n\n"
    elif bert_results:
        report += f"BERT模型在中药材评论情感分析任务上表现良好，准确率达到{bert_results['accuracy']:.4f}。\n\n"
    
    report += "### 训练可视化\n\n"
    
    if lstm_results:
        report += "LSTM训练历史:\n\n"
        report += "![LSTM训练历史](figures/lstm_training_history.png)\n\n"
    
    if bert_results:
        report += "BERT训练历史:\n\n"
        report += "![BERT训练历史](figures/bert_training_stats.png)\n\n"
    
    # 写入报告文件
    with open(os.path.join(output_dir, 'deep_learning_report.md'), 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"摘要报告已生成: {os.path.join(output_dir, 'deep_learning_report.md')}")

def main():
    # 解析命令行参数
    args = parse_arguments()
    
    # 确保输出目录存在
    ensure_output_dir(args.output_dir)
    
    # 获取评论文件列表
    comments_files = get_all_comments_files()
    
    # 读取评论数据
    comments, labels = read_all_comments(comments_files, args.sample_size)
    
    # 可视化评论分布
    visualize_comment_distribution(labels, args.output_dir)
    
    # 训练模型
    lstm_results = None
    bert_results = None
    
    if args.model in ['lstm', 'both']:
        lstm_results = train_lstm_model(comments, labels, args.output_dir, args.epochs, args.batch_size)
    
    if args.model in ['bert', 'both']:
        bert_results = train_bert_model(comments, labels, args.output_dir, args.epochs, args.batch_size)
    
    # 生成摘要报告
    generate_summary_report(lstm_results, bert_results, args.output_dir)

if __name__ == "__main__":
    # 设置中文显示
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 用于显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用于正常显示负号
    
    main() 