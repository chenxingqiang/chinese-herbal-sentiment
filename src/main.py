#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
生成中药材企业电商供应链服务质量评价论文的图表

该脚本生成论文中引用的所有图表，包括：
1. 情感分析结果图表
2. 关键词频率图表
3. 算法性能比较图表
4. 评价指标雷达图
5. 供应链网络图
6. 维度得分比较图
7. 回归分析结果图
8. 相关性热力图
9. LSTM模型架构图
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.font_manager import FontProperties
import networkx as nx

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.style.use('seaborn-v0_8-whitegrid')

# 创建output/figures目录
os.makedirs('output/figures', exist_ok=True)

# 设置图表尺寸和DPI
FIG_SIZE = (10, 6)
FIG_SIZE_LARGE = (12, 8)
DPI = 300

# 设置颜色方案
COLORS = ['#2878B5', '#9AC9DB', '#C82423', '#F8AC8C', '#1E963C', '#40B27D', '#F3D266', '#985F99']

def create_sentiment_distribution():
    """生成情感分布饼图"""
    labels = ['正面评论', '中性评论', '负面评论']
    sizes = [65.3, 16.0, 18.7]
    colors = ['#66b3ff', '#c2c2c2', '#ff9999']
    explode = (0.1, 0, 0)  # 突出正面评论
    
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, 
                                     autopct='%1.1f%%', startangle=90, colors=colors,
                                     textprops={'fontsize': 14})
    
    # 设置标题和属性
    ax.set_title('中药材企业电商评论情感分布', fontsize=16, pad=20)
    plt.setp(autotexts, size=12, weight="bold")
    plt.setp(texts, size=14)
    
    # 添加图例
    ax.legend(labels, loc="best", fontsize=12)
    
    plt.tight_layout()
    plt.savefig('output/figures/sentiment_distribution.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("已生成情感分布饼图")

def create_keyword_frequency():
    """生成关键词频率柱状图"""
    keywords = ['新鲜', '正品', '效果好', '包装好', '物流快', '价格合理',
                '假货', '发霉', '效果差', '物流慢', '客服差']
    
    frequencies = [85, 78, 72, 68, 65, 58, 45, 42, 38, 35, 30]
    sentiment = ['正面', '正面', '正面', '正面', '正面', '正面',
                '负面', '负面', '负面', '负面', '负面']
    
    # 创建DataFrame
    df = pd.DataFrame({
        '关键词': keywords,
        '频率': frequencies,
        '情感倾向': sentiment
    })
    
    # 设置颜色映射
    color_map = {'正面': '#66b3ff', '负面': '#ff9999'}
    
    # 绘制柱状图
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    sns.barplot(x='频率', y='关键词', hue='情感倾向', palette=color_map, data=df, ax=ax)
    
    # 设置标题和标签
    ax.set_title('中药材企业电商评论关键词频率分析', fontsize=16)
    ax.set_xlabel('出现频率', fontsize=14)
    ax.set_ylabel('关键词', fontsize=14)
    
    # 添加数值标签
    for i, v in enumerate(frequencies):
        ax.text(v + 1, i, str(v), va='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('output/figures/keyword_frequency.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("已生成关键词频率柱状图")

def create_algorithm_comparison():
    """生成算法性能比较图"""
    algorithms = ['朴素贝叶斯', 'SVM', 'LSTM', 'BERT', '混合模型']
    metrics = ['准确率', '精确率', '召回率', 'F1值']
    
    # 模拟各算法在不同指标上的表现
    data = np.array([
        [0.78, 0.75, 0.77, 0.76],  # 朴素贝叶斯
        [0.82, 0.80, 0.79, 0.80],  # SVM
        [0.87, 0.85, 0.84, 0.84],  # LSTM
        [0.89, 0.88, 0.86, 0.87],  # BERT
        [0.91, 0.90, 0.88, 0.89]   # 混合模型
    ])
    
    # 创建DataFrame
    df = pd.DataFrame(data, index=algorithms, columns=metrics)
    
    # 绘制热力图
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    sns.heatmap(df, annot=True, cmap='Blues', fmt='.2f', 
                linewidths=.5, cbar_kws={"shrink": .8}, ax=ax, annot_kws={"size": 12})
    
    # 设置标题和标签
    ax.set_title('情感分析算法性能比较', fontsize=16)
    ax.set_xlabel('评估指标', fontsize=14)
    ax.set_ylabel('算法', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('output/figures/algorithm_comparison.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("已生成算法性能比较图")

def create_indicator_radar():
    """生成评价指标雷达图"""
    # 各维度的指标
    categories = ['原料质量', '原料规格一致性', '原材料可追溯性',
                '工艺技术水平', '产品一致性', '质检标准符合度',
                '交货速度', '包装质量', '订单准确性',
                '售后响应速度', '信息透明度']
    
    # 模拟数据
    values = [8.2, 7.9, 7.1, 7.8, 7.6, 7.5, 8.1, 8.0, 8.2, 7.5, 7.3]
    
    # 将数据转换为雷达图所需格式
    values = values + [values[0]]  # 闭合雷达图
    categories = categories + [categories[0]]  # 闭合类别
    
    # 计算角度
    N = len(categories) - 1
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 闭合
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # 绘制雷达图
    ax.fill(angles, values, color=COLORS[0], alpha=0.25)
    ax.plot(angles, values, color=COLORS[0], linewidth=2, marker='o', markersize=8)
    
    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories[:-1], fontsize=12)
    
    # 设置刻度范围
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=10)
    ax.set_ylim(0, 10)
    
    # 设置标题
    ax.set_title('中药材企业电商供应链服务质量评价指标雷达图', fontsize=16, pad=20)
    
    plt.tight_layout()
    plt.savefig('output/figures/indicator_radar.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("已生成评价指标雷达图")

def create_supply_chain_network():
    """生成供应链网络图"""
    # 创建有向图
    G = nx.DiGraph()
    
    # 添加节点
    nodes = {
        '原料供应商': {'pos': (0, 3), 'color': COLORS[0]},
        '种植基地': {'pos': (0, 1), 'color': COLORS[0]},
        '加工企业': {'pos': (2, 2), 'color': COLORS[1]},
        '仓储中心': {'pos': (4, 3), 'color': COLORS[2]},
        '物流配送': {'pos': (4, 1), 'color': COLORS[2]},
        '电商平台': {'pos': (6, 2), 'color': COLORS[3]},
        '消费者': {'pos': (8, 2), 'color': COLORS[4]}
    }
    
    for node, attr in nodes.items():
        G.add_node(node, **attr)
    
    # 添加边
    edges = [
        ('原料供应商', '加工企业', {'label': '原料质量', 'width': 2}),
        ('种植基地', '加工企业', {'label': '原料新鲜度', 'width': 2}),
        ('加工企业', '仓储中心', {'label': '产品一致性', 'width': 2}),
        ('仓储中心', '物流配送', {'label': '库存管理', 'width': 1.5}),
        ('物流配送', '电商平台', {'label': '配送速度', 'width': 1.5}),
        ('电商平台', '消费者', {'label': '售后服务', 'width': 2}),
        ('电商平台', '物流配送', {'label': '订单处理', 'width': 1, 'style': 'dashed'}),
        ('加工企业', '电商平台', {'label': '质量信息', 'width': 1, 'style': 'dashed'})
    ]
    
    for u, v, attr in edges:
        G.add_edge(u, v, **attr)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=FIG_SIZE_LARGE)
    
    # 获取节点位置
    pos = nx.get_node_attributes(G, 'pos')
    
    # 绘制节点
    node_colors = [attr['color'] for node, attr in nodes.items()]
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color=node_colors, alpha=0.8, ax=ax)
    
    # 绘制边
    for u, v, attr in G.edges(data=True):
        edge_style = attr.get('style', 'solid')
        edge_width = attr.get('width', 1.0)
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=edge_width, 
                              edge_color='gray', style=edge_style, 
                              arrowsize=20, connectionstyle='arc3,rad=0.1', ax=ax)
    
    # 绘制标签
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)
    
    # 绘制边标签
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, ax=ax)
    
    # 设置图表属性
    ax.set_title('中药材企业电商供应链服务质量评价应用网络图', fontsize=16)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('output/figures/supply_chain_network.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("已生成供应链网络图")

def create_dimension_scores():
    """生成三个维度评分比较图"""
    dimensions = ['上游（原料）', '中游（加工）', '下游（销售与物流）']
    scores = [0.78, 0.82, 0.76]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    # 绘制条形图
    bars = ax.bar(dimensions, scores, color=COLORS[:3], width=0.6)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=14)
    
    # 设置标题和标签
    ax.set_title('中药材企业电商供应链三个维度评分比较', fontsize=16)
    ax.set_ylabel('评分 (0-1)', fontsize=14)
    ax.set_ylim(0, 1.0)
    
    # 添加网格线
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('output/figures/dimension_scores.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("已生成维度评分比较图")

def create_regression_results():
    """生成回归分析结果图"""
    factors = ['原料质量', '物流配送', '加工工艺', '售后服务', '信息透明度']
    coefficients = [0.352, 0.312, 0.285, 0.245, 0.218]
    
    # 按系数大小排序
    sorted_indices = np.argsort(coefficients)[::-1]  # 降序排列
    factors = [factors[i] for i in sorted_indices]
    coefficients = [coefficients[i] for i in sorted_indices]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    # 绘制水平条形图
    bars = ax.barh(factors, coefficients, color=COLORS, alpha=0.7, height=0.6)
    
    # 添加数值标签
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}', va='center', fontsize=12)
    
    # 设置标题和标签
    ax.set_title('中药材企业电商供应链服务质量影响因素回归分析', fontsize=16)
    ax.set_xlabel('回归系数', fontsize=14)
    ax.set_xlim(0, 0.4)
    
    # 添加网格线
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('output/figures/regression_results.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("已生成回归分析结果图")

def create_correlation_heatmap():
    """生成相关性热力图"""
    # 创建变量列表
    variables = ['服务质量', '原料质量', '加工工艺', '物流配送', '售后服务', '信息透明度']
    
    # 创建相关系数矩阵
    corr_matrix = np.array([
        [1.00, 0.78, 0.72, 0.75, 0.68, 0.65],  # 服务质量
        [0.78, 1.00, 0.65, 0.45, 0.42, 0.38],  # 原料质量
        [0.72, 0.65, 1.00, 0.48, 0.44, 0.40],  # 加工工艺
        [0.75, 0.45, 0.48, 1.00, 0.62, 0.52],  # 物流配送
        [0.68, 0.42, 0.44, 0.62, 1.00, 0.58],  # 售后服务
        [0.65, 0.38, 0.40, 0.52, 0.58, 1.00]   # 信息透明度
    ])
    
    # 创建DataFrame
    df = pd.DataFrame(corr_matrix, index=variables, columns=variables)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    # 绘制热力图
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(df, mask=mask, cmap=cmap, vmax=1.0, vmin=0.0, center=0.5,
                annot=True, fmt=".2f", square=True, linewidths=.5, 
                cbar_kws={"shrink": .8}, annot_kws={"size": 12}, ax=ax)
    
    # 设置标题
    ax.set_title('中药材企业电商供应链服务质量因素相关性热力图', fontsize=16)
    
    plt.tight_layout()
    plt.savefig('output/figures/correlation_heatmap.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("已生成相关性热力图")

def create_lstm_architecture():
    """生成LSTM模型架构图"""
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    # 设置背景色
    ax.set_facecolor('#f8f8f8')
    
    # 绘制输入层
    ax.add_patch(plt.Rectangle((0.1, 0.8), 0.8, 0.1, fill=True, color=COLORS[0], alpha=0.7))
    ax.text(0.5, 0.85, '输入层', ha='center', va='center', fontsize=14)
    
    # 绘制Embedding层
    ax.add_patch(plt.Rectangle((0.1, 0.65), 0.8, 0.1, fill=True, color=COLORS[1], alpha=0.7))
    ax.text(0.5, 0.7, 'Embedding层', ha='center', va='center', fontsize=14)
    
    # 绘制LSTM层
    ax.add_patch(plt.Rectangle((0.1, 0.45), 0.8, 0.15, fill=True, color=COLORS[2], alpha=0.7))
    ax.text(0.5, 0.525, 'LSTM层', ha='center', va='center', fontsize=14)
    
    # 绘制全连接层
    ax.add_patch(plt.Rectangle((0.1, 0.3), 0.8, 0.1, fill=True, color=COLORS[3], alpha=0.7))
    ax.text(0.5, 0.35, '全连接层', ha='center', va='center', fontsize=14)
    
    # 绘制输出层
    ax.add_patch(plt.Rectangle((0.1, 0.15), 0.8, 0.1, fill=True, color=COLORS[4], alpha=0.7))
    ax.text(0.5, 0.2, '输出层', ha='center', va='center', fontsize=14)
    
    # 绘制连接箭头
    arrow_props = dict(arrowstyle='->', lw=1.5, color='gray')
    ax.annotate('', xy=(0.5, 0.8), xytext=(0.5, 0.75), arrowprops=arrow_props)
    ax.annotate('', xy=(0.5, 0.65), xytext=(0.5, 0.6), arrowprops=arrow_props)
    ax.annotate('', xy=(0.5, 0.45), xytext=(0.5, 0.4), arrowprops=arrow_props)
    ax.annotate('', xy=(0.5, 0.3), xytext=(0.5, 0.25), arrowprops=arrow_props)
    
    # 添加文本说明
    ax.text(0.5, 0.9, '评论文本', ha='center', va='center', fontsize=12)
    ax.text(0.5, 0.1, '情感类别(正面/负面/中性)', ha='center', va='center', fontsize=12)
    
    # 移除坐标轴
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # 设置标题
    ax.set_title('LSTM情感分析模型架构', fontsize=16)
    
    plt.tight_layout()
    plt.savefig('output/figures/lstm_architecture.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("已生成LSTM模型架构图")

def main():
    """主函数，生成所有图表"""
    print("开始生成论文中引用的图表...")
    
    create_sentiment_distribution()  # 图4-1 中药材企业电商评论情感分布
    create_keyword_frequency()       # 图4-2 中药材企业电商评论关键词频率分析
    create_indicator_radar()         # 图4-3 中药材企业电商供应链服务质量评价指标雷达图
    create_algorithm_comparison()    # 图4-4 情感分析算法性能比较
    create_lstm_architecture()       # 图4-5 LSTM情感分析模型架构
    create_dimension_scores()        # 图5-1 中药材企业电商供应链三个维度评分比较
    create_regression_results()      # 图5-2 中药材企业电商供应链服务质量影响因素回归分析
    create_correlation_heatmap()     # 图5-3 中药材企业电商供应链服务质量因素相关性热力图
    create_supply_chain_network()    # 图3-1 中药材企业电商供应链服务质量评价应用网络图
    
    print("所有图表生成完成！")

if __name__ == "__main__":
    main() 