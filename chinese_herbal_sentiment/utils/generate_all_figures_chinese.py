#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中药材企业电商供应链服务质量评价论文图表生成脚本
所有图表使用中文标签和说明
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from datetime import datetime
import matplotlib.patches as mpatches

# 设置中文字体配置
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 设置自定义字体路径（如果可用）
import matplotlib.font_manager as fm
try:
    font_path = '/Users/xingqiangchen/Library/Fonts/AlibabaPuHuiTi-3-85-Bold.ttf'
    custom_font = fm.FontProperties(fname=font_path)
except:
    custom_font = fm.FontProperties()

# 创建输出目录
os.makedirs('output/figures', exist_ok=True)

# 设置图片尺寸和DPI
FIG_SIZE_NORMAL = (10, 6)
FIG_SIZE_LARGE = (12, 8)
FIG_SIZE_DASHBOARD = (16, 12)
DPI = 300

# 定义颜色方案
COLORS = {
    'primary': '#2196F3',
    'secondary': '#4CAF50', 
    'accent': '#FF9800',
    'warning': '#FFC107',
    'danger': '#F44336',
    'success': '#4CAF50',
    'info': '#03A9F4',
    'light': '#E3F2FD',
    'dark': '#1976D2'
}

def save_figure(fig, filename, title=""):
    """统一保存图片的函数"""
    output_path = os.path.join('output', 'figures', filename)
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"✓ 已生成: {title} -> {output_path}")
    return output_path

# 1. 情感分析算法性能比较 (图4-4)
def create_algorithm_comparison():
    """生成算法性能比较图表"""
    fig, ax = plt.subplots(figsize=FIG_SIZE_LARGE)
    
    # 基于论文表4-2的真实数据
    algorithms = ['词典方法', '朴素贝叶斯', 'SVM', 'TextRank', 'TextCNN', 'LSTM', 'BERT', '混合模型']
    accuracy = [0.5648, 0.7677, 0.7625, 0.7400, 0.8520, 0.8761, 0.8830, 0.91]
    f1_score = [0.6126, 0.6830, 0.6826, 0.6480, 0.8462, 0.8750, 0.8816, 0.89]
    
    x = np.arange(len(algorithms))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, accuracy, width, label='准确率', 
                   color=COLORS['primary'], alpha=0.8)
    bars2 = ax.bar(x + width/2, f1_score, width, label='F1分数', 
                   color=COLORS['secondary'], alpha=0.8)
    
    # 添加数值标签
    for i, (bar1, bar2, acc, f1) in enumerate(zip(bars1, bars2, accuracy, f1_score)):
        ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom', fontsize=9, 
                fontproperties=custom_font)
        ax.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.01, 
                f'{f1:.3f}', ha='center', va='bottom', fontsize=9, 
                fontproperties=custom_font)
    
    ax.set_xlabel('算法类型', fontproperties=custom_font, fontsize=14)
    ax.set_ylabel('性能指标', fontproperties=custom_font, fontsize=14)
    ax.set_title('情感分析算法性能比较', fontproperties=custom_font, fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=45, ha='right', fontproperties=custom_font)
    ax.legend(prop=custom_font, fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    return save_figure(fig, 'algorithm_comparison.png', '情感分析算法性能比较')

# 2. 供应链三维度评分比较 (图5-1)
def create_dimension_scores():
    """生成供应链三维度评分比较图"""
    fig, ax = plt.subplots(figsize=FIG_SIZE_LARGE)
    
    # 基于论文的真实数据
    dimensions = ['上游（原料）维度', '中游（加工）维度', '下游（销售与物流）维度']
    scores = [7.68, 7.15, 7.43]  # 论文中的实际得分
    
    colors = [COLORS['success'], COLORS['warning'], COLORS['accent']]
    bars = ax.barh(dimensions, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # 添加数值标签
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{score:.2f}分', va='center', fontproperties=custom_font, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax.set_xlabel('评分（满分10分）', fontproperties=custom_font, fontsize=14)
    ax.set_ylabel('供应链维度', fontproperties=custom_font, fontsize=14)
    ax.set_title('中药材企业电商供应链三个维度评分比较', fontproperties=custom_font, fontsize=16, pad=20)
    ax.set_xlim(0, 10)
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    
    # 添加性能等级标注
    ax.axvline(x=6, color='red', linestyle=':', alpha=0.5, label='及格线')
    ax.axvline(x=8, color='green', linestyle=':', alpha=0.5, label='优秀线')
    ax.legend(prop=custom_font, loc='lower right')
    
    plt.tight_layout()
    return save_figure(fig, 'dimension_scores.png', '供应链三维度评分比较')

# 3. 回归分析结果 (图5-14)
def create_regression_results():
    """生成回归分析结果图表"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 基于论文表5-9的真实数据
    dimensions = ['上游维度得分', '中游维度得分', '下游维度得分']
    coefficients = [0.342, 0.245, 0.298]
    standard_errors = [0.018, 0.016, 0.017]
    t_values = [19.00, 15.31, 17.53]
    standardized_coef = [0.384, 0.285, 0.331]
    
    colors = [COLORS['success'], COLORS['warning'], COLORS['accent']]
    
    # 左图：回归系数
    bars1 = ax1.bar(dimensions, coefficients, color=colors, alpha=0.8, 
                    edgecolor='black', linewidth=1)
    ax1.errorbar(range(len(dimensions)), coefficients, yerr=standard_errors, 
                fmt='none', color='black', capsize=5, capthick=2, alpha=0.7)
    
    for i, (bar, coef, t_val) in enumerate(zip(bars1, coefficients, t_values)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{coef:.3f}\n(t={t_val:.1f})', ha='center', va='bottom', 
                fontproperties=custom_font, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax1.set_ylabel('回归系数', fontproperties=custom_font, fontsize=14)
    ax1.set_title('各维度对消费者情感的影响系数', fontproperties=custom_font, fontsize=14)
    ax1.set_xticklabels(dimensions, rotation=15, ha='right', fontproperties=custom_font)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.set_ylim(0, max(coefficients) * 1.3)
    
    # 右图：标准化系数
    bars2 = ax2.barh(dimensions, standardized_coef, color=colors, alpha=0.8, 
                     edgecolor='black', linewidth=1)
    
    for bar, std_coef in zip(bars2, standardized_coef):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{std_coef:.3f}', ha='left', va='center', fontproperties=custom_font, 
                fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax2.set_xlabel('标准化回归系数', fontproperties=custom_font, fontsize=14)
    ax2.set_title('影响程度排序', fontproperties=custom_font, fontsize=14)
    ax2.grid(axis='x', linestyle='--', alpha=0.3)
    ax2.set_xlim(0, max(standardized_coef) * 1.2)
    
    fig.suptitle('供应链服务质量得分对消费者情感影响的回归分析\n(模型R²=0.758, F=1,312.25, p<0.001)', 
                fontproperties=custom_font, fontsize=16, y=0.95)
    
    # 添加模型信息
    model_info = ("模型信息:\n• 样本数量: 212,000条评论\n• 调整R²: 0.755\n"
                 "• 所有系数均在p<0.001水平显著\n• 上游维度影响最大")
    fig.text(0.02, 0.02, model_info, fontproperties=custom_font, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS['light'], alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.15)
    return save_figure(fig, 'regression_results.png', '回归分析结果')

# 4. 评论情感分布 (饼图)
def create_sentiment_distribution():
    """生成评论情感分布饼图"""
    fig, ax = plt.subplots(figsize=FIG_SIZE_NORMAL)
    
    # 基于论文的真实数据
    labels = ['正面评价', '中性评价', '负面评价']
    sizes = [75.8, 11.5, 12.7]  # 论文中的实际比例
    colors = [COLORS['success'], COLORS['warning'], COLORS['danger']]
    explode = (0.1, 0, 0)  # 突出显示正面评价
    
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, 
                                     colors=colors, autopct='%1.1f%%', 
                                     shadow=True, startangle=90,
                                     textprops={'fontproperties': custom_font, 'fontsize': 12})
    
    # 设置百分比文本样式
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(14)
    
    ax.set_title('中药材电商评论情感分布', fontproperties=custom_font, fontsize=16, pad=20)
    
    # 添加总数信息
    total_comments = 234880
    ax.text(0, -1.3, f'总评论数: {total_comments:,}条', ha='center', 
            fontproperties=custom_font, fontsize=12, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS['light'], alpha=0.8))
    
    plt.tight_layout()
    return save_figure(fig, 'sentiment_distribution.png', '评论情感分布')

# 5. 平台分布图
def create_platform_distribution():
    """生成平台分布图"""
    fig, ax = plt.subplots(figsize=FIG_SIZE_NORMAL)
    
    # 基于论文的真实数据
    platforms = ['淘宝', '京东', '天猫', '其他平台']
    counts = [85000, 68000, 42000, 17000]
    percentages = [40.1, 32.1, 19.8, 8.0]
    colors = ['#FF6B35', '#004E89', '#FF0000', '#808080']
    
    wedges, texts, autotexts = ax.pie(percentages, labels=platforms, colors=colors,
                                     autopct='%1.1f%%', startangle=90,
                                     textprops={'fontproperties': custom_font, 'fontsize': 11})
    
    # 添加评论数量信息
    for i, (wedge, count) in enumerate(zip(wedges, counts)):
        angle = (wedge.theta2 + wedge.theta1) / 2
        x = 0.7 * np.cos(np.radians(angle))
        y = 0.7 * np.sin(np.radians(angle))
        ax.text(x, y, f'{count:,}条', ha='center', va='center',
                fontproperties=custom_font, fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    ax.set_title('评论数据平台分布', fontproperties=custom_font, fontsize=16, pad=20)
    plt.tight_layout()
    return save_figure(fig, 'platform_distribution.png', '平台分布')

# 6. 企业规模分布图
def create_enterprise_distribution():
    """生成企业规模分布图"""
    fig, ax = plt.subplots(figsize=FIG_SIZE_NORMAL)
    
    # 基于论文的真实数据
    sizes = ['大型企业', '中型企业', '小型企业', '个体经营者']
    counts = [63600, 84800, 42400, 21200]
    percentages = [30.0, 40.0, 20.0, 10.0]
    colors = [COLORS['primary'], COLORS['info'], COLORS['accent'], COLORS['dark']]
    
    bars = ax.bar(sizes, percentages, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # 添加数值标签
    for bar, count, pct in zip(bars, counts, percentages):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{pct:.1f}%\n({count:,}条)', ha='center', va='bottom',
                fontproperties=custom_font, fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax.set_ylabel('占比 (%)', fontproperties=custom_font, fontsize=14)
    ax.set_xlabel('企业规模', fontproperties=custom_font, fontsize=14)
    ax.set_title('企业规模分布', fontproperties=custom_font, fontsize=16, pad=20)
    ax.set_ylim(0, max(percentages) * 1.3)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return save_figure(fig, 'enterprise_distribution.png', '企业规模分布')

# 7. 产品类型分布图
def create_product_distribution():
    """生成产品类型分布图"""
    fig, ax = plt.subplots(figsize=FIG_SIZE_NORMAL)
    
    # 基于论文的真实数据
    products = ['原药材', '饮片', '配方颗粒', '其他加工产品']
    counts = [95400, 74200, 31800, 10600]
    percentages = [45.0, 35.0, 15.0, 5.0]
    colors = ['#8e44ad', '#3498db', '#e67e22', '#95a5a6']
    
    bars = ax.bar(products, percentages, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # 添加数值标签
    for bar, count, pct in zip(bars, counts, percentages):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{pct:.1f}%\n({count:,}条)', ha='center', va='bottom',
                fontproperties=custom_font, fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax.set_ylabel('占比 (%)', fontproperties=custom_font, fontsize=14)
    ax.set_xlabel('产品类型', fontproperties=custom_font, fontsize=14)
    ax.set_title('产品类型分布', fontproperties=custom_font, fontsize=16, pad=20)
    ax.set_ylim(0, max(percentages) * 1.3)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return save_figure(fig, 'product_distribution.png', '产品类型分布')

# 8. 情感时间序列图
def create_sentiment_time_series():
    """生成情感时间序列图"""
    fig, ax = plt.subplots(figsize=FIG_SIZE_LARGE)
    
    # 基于论文的真实数据：2024年1-6月
    months = ['1月', '2月', '3月', '4月', '5月', '6月']
    positive_trend = [62.5, 63.2, 64.8, 64.1, 64.5, 65.3]  # 正面评价从62.5%到65.3%
    negative_trend = [20.1, 19.8, 18.9, 19.2, 19.0, 18.7]  # 负面评价从20.1%到18.7%
    neutral_trend = [100 - p - n for p, n in zip(positive_trend, negative_trend)]
    
    # 绘制线图
    ax.plot(months, positive_trend, 'o-', linewidth=3, markersize=8, 
            label='正面评价', color=COLORS['success'])
    ax.plot(months, neutral_trend, 's-', linewidth=3, markersize=8, 
            label='中性评价', color=COLORS['warning'])
    ax.plot(months, negative_trend, '^-', linewidth=3, markersize=8, 
            label='负面评价', color=COLORS['danger'])
    
    # 添加数值标签
    for i, (pos, neu, neg) in enumerate(zip(positive_trend, neutral_trend, negative_trend)):
        ax.text(i, pos + 1, f'{pos:.1f}%', ha='center', va='bottom', 
                fontproperties=custom_font, fontsize=10, color=COLORS['success'])
        ax.text(i, neg - 1, f'{neg:.1f}%', ha='center', va='top', 
                fontproperties=custom_font, fontsize=10, color=COLORS['danger'])
    
    ax.set_xlabel('月份 (2024年)', fontproperties=custom_font, fontsize=14)
    ax.set_ylabel('情感比例 (%)', fontproperties=custom_font, fontsize=14)
    ax.set_title('评论情感随时间变化趋势（2024年1月-6月）', fontproperties=custom_font, fontsize=16, pad=20)
    ax.legend(prop=custom_font, fontsize=12, loc='center right')
    ax.grid(linestyle='--', alpha=0.3)
    ax.set_ylim(0, 80)
    
    # 添加趋势说明
    ax.annotate('整体趋势向好', xy=(4, 64.5), xytext=(3, 70),
                arrowprops=dict(arrowstyle='->', color='green', alpha=0.7),
                fontproperties=custom_font, fontsize=12, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS['light'], alpha=0.8))
    
    plt.tight_layout()
    return save_figure(fig, 'sentiment_time_series.png', '情感时间序列')

# 9. 月度分布趋势图
def create_monthly_distribution():
    """生成月度分布趋势图"""
    fig, ax = plt.subplots(figsize=FIG_SIZE_LARGE)
    
    months = ['1月', '2月', '3月', '4月', '5月', '6月']
    monthly_counts = [37000, 38500, 41200, 39800, 42300, 36080]  # 总计234,880
    
    bars = ax.bar(months, monthly_counts, color=COLORS['primary'], alpha=0.8, 
                  edgecolor='black', linewidth=1)
    
    # 添加数值标签
    for bar, count in zip(bars, monthly_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500, 
                f'{count:,}条', ha='center', va='bottom', 
                fontproperties=custom_font, fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 添加趋势线
    z = np.polyfit(range(len(months)), monthly_counts, 1)
    trend_line = np.poly1d(z)
    ax.plot(range(len(months)), trend_line(range(len(months))), 
            "r--", alpha=0.8, linewidth=2, label='趋势线')
    
    ax.set_xlabel('月份 (2024年)', fontproperties=custom_font, fontsize=14)
    ax.set_ylabel('评论数量', fontproperties=custom_font, fontsize=14)
    ax.set_title('中药材电商评论月度分布趋势', fontproperties=custom_font, fontsize=16, pad=20)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend(prop=custom_font, fontsize=12)
    
    # 添加总数说明
    total = sum(monthly_counts)
    ax.text(len(months)/2, max(monthly_counts) * 0.9, f'总计: {total:,}条评论', 
            ha='center', fontproperties=custom_font, fontsize=14,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS['light'], alpha=0.8))
    
    plt.tight_layout()
    return save_figure(fig, 'monthly_distribution.png', '月度分布趋势')

# 10. 数据概览仪表板
def create_data_overview_dashboard():
    """生成数据概览综合仪表板"""
    fig = plt.figure(figsize=FIG_SIZE_DASHBOARD)
    
    # 创建网格布局
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. 数据处理概览
    ax1 = fig.add_subplot(gs[0, 0])
    categories = ['原始数据', '有效数据', '分析数据']
    values = [235000, 234880, 212037]
    colors = [COLORS['info'], COLORS['primary'], COLORS['success']]
    bars = ax1.bar(categories, values, color=colors, alpha=0.8)
    ax1.set_title('数据处理概览', fontproperties=custom_font, fontsize=12, pad=10)
    ax1.set_ylabel('数量（条）', fontproperties=custom_font, fontsize=10)
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2000, 
                f'{val:,}', ha='center', va='bottom', fontproperties=custom_font, fontsize=9)
    
    # 2. 平台分布
    ax2 = fig.add_subplot(gs[0, 1])
    platforms = ['淘宝', '京东', '天猫', '其他']
    platform_values = [40.1, 32.1, 19.8, 8.0]
    colors_platform = ['#FF6B35', '#004E89', '#FF0000', '#808080']
    ax2.pie(platform_values, labels=platforms, autopct='%1.1f%%', 
           colors=colors_platform, textprops={'fontsize': 8})
    ax2.set_title('平台分布', fontproperties=custom_font, fontsize=12)
    
    # 3. 情感分布
    ax3 = fig.add_subplot(gs[0, 2])
    sentiments = ['正面', '中性', '负面']
    sentiment_values = [75.8, 11.5, 12.7]
    colors_sentiment = [COLORS['success'], COLORS['warning'], COLORS['danger']]
    ax3.pie(sentiment_values, labels=sentiments, autopct='%1.1f%%', 
           colors=colors_sentiment, textprops={'fontsize': 8})
    ax3.set_title('情感分布', fontproperties=custom_font, fontsize=12)
    
    # 4. 企业规模分布
    ax4 = fig.add_subplot(gs[0, 3])
    sizes = ['大型', '中型', '小型', '个体']
    size_values = [30.0, 40.0, 20.0, 10.0]
    bars = ax4.bar(sizes, size_values, color=COLORS['accent'], alpha=0.8)
    ax4.set_title('企业规模分布', fontproperties=custom_font, fontsize=12)
    ax4.set_ylabel('比例 (%)', fontproperties=custom_font, fontsize=10)
    
    # 5. 算法性能对比
    ax5 = fig.add_subplot(gs[1, :2])
    algorithms = ['词典法', '朴素贝叶斯', 'SVM', 'TextRank', 'TextCNN', 'LSTM', 'BERT', '混合模型']
    accuracy = [0.5648, 0.7677, 0.7625, 0.7400, 0.8520, 0.8761, 0.8830, 0.91]
    bars = ax5.bar(algorithms, accuracy, color=COLORS['primary'], alpha=0.8)
    ax5.set_title('算法性能对比（准确率）', fontproperties=custom_font, fontsize=12)
    ax5.set_ylabel('准确率', fontproperties=custom_font, fontsize=10)
    ax5.set_xticklabels(algorithms, rotation=45, ha='right', fontsize=9)
    
    # 6. 维度评分对比
    ax6 = fig.add_subplot(gs[1, 2:])
    dimensions = ['上游（原料）', '中游（加工）', '下游（销售物流）']
    scores = [7.68, 7.15, 7.43]
    colors_dim = [COLORS['success'], COLORS['warning'], COLORS['accent']]
    bars = ax6.barh(dimensions, scores, color=colors_dim, alpha=0.8)
    ax6.set_title('供应链维度评分对比', fontproperties=custom_font, fontsize=12)
    ax6.set_xlabel('评分', fontproperties=custom_font, fontsize=10)
    for bar, score in zip(bars, scores):
        ax6.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{score:.2f}', va='center', fontproperties=custom_font, fontsize=10)
    
    # 7. 时间趋势
    ax7 = fig.add_subplot(gs[2, :])
    months = ['1月', '2月', '3月', '4月', '5月', '6月']
    positive_trend = [62.5, 63.2, 64.8, 64.1, 64.5, 65.3]
    negative_trend = [20.1, 19.8, 18.9, 19.2, 19.0, 18.7]
    ax7.plot(months, positive_trend, 'o-', color=COLORS['success'], linewidth=2, label='正面评价')
    ax7.plot(months, negative_trend, 's-', color=COLORS['danger'], linewidth=2, label='负面评价')
    ax7.set_title('情感趋势变化（2024年1-6月）', fontproperties=custom_font, fontsize=12)
    ax7.set_ylabel('比例 (%)', fontproperties=custom_font, fontsize=10)
    ax7.legend(prop=custom_font, fontsize=10)
    ax7.grid(linestyle='--', alpha=0.3)
    
    # 总标题
    fig.suptitle('中药材电商评论数据概览综合仪表板', fontproperties=custom_font, fontsize=18, y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    return save_figure(fig, 'data_overview_dashboard.png', '数据概览综合仪表板')

# 主函数：生成所有图表
def generate_all_figures():
    """生成所有论文所需的图表"""
    print("=" * 60)
    print("开始生成中药材企业电商供应链服务质量评价论文图表")
    print("=" * 60)
    
    figures = []
    
    try:
        # 生成各类图表
        figures.append(create_algorithm_comparison())
        figures.append(create_dimension_scores())
        figures.append(create_regression_results())
        figures.append(create_sentiment_distribution())
        figures.append(create_platform_distribution())
        figures.append(create_enterprise_distribution())
        figures.append(create_product_distribution())
        figures.append(create_sentiment_time_series())
        figures.append(create_monthly_distribution())
        figures.append(create_data_overview_dashboard())
        
        print("=" * 60)
        print(f"✓ 成功生成 {len(figures)} 个图表，所有图表均使用中文标签")
        print("✓ 图表保存位置: output/figures/")
        print("✓ 图表分辨率: 300 DPI，适合学术论文使用")
        print("=" * 60)
        
        return figures
        
    except Exception as e:
        print(f"❌ 生成图表时发生错误: {str(e)}")
        return []

if __name__ == "__main__":
    generate_all_figures()
