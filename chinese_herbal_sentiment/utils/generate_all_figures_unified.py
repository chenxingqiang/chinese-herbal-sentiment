#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中药材企业电商供应链服务质量评价论文图表生成脚本 (统一配色版本)
所有图表使用统一的配色方案和中文标签
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from datetime import datetime
import matplotlib.patches as mpatches

# 导入统一配色方案
from unified_color_scheme import (
    setup_unified_style, get_custom_font, get_color_palette, 
    apply_unified_theme, save_unified_figure, get_academic_config,
    UNIFIED_COLORS, COLOR_SCHEMES
)

# 设置统一样式
setup_unified_style()
custom_font = get_custom_font()

# 创建输出目录
os.makedirs('output/figures', exist_ok=True)

# 1. 情感分析算法性能比较 (图4-4)
def create_algorithm_comparison():
    """生成算法性能比较图表 - 使用统一配色"""
    # 基于论文表4-2的数据
    algorithms = ['词典方法', 'SVM', '朴素贝叶斯', 'LSTM', 'TextCNN', 'TextRank', 'BERT', '混合模型']
    accuracy = [0.5648, 0.7625, 0.7677, 0.8761, 0.8520, 0.7400, 0.8830, 0.9100]
    f1_scores = [0.6126, 0.6826, 0.6830, 0.8750, 0.8462, 0.6480, 0.8816, 0.9080]
    
    fig, ax = plt.subplots(figsize=get_academic_config('figure_sizes', 'large'))
    
    # 使用统一配色方案
    colors = get_color_palette('algorithm_performance', len(algorithms))
    
    x = np.arange(len(algorithms))
    width = 0.35
    
    # 创建柱状图
    bars1 = ax.bar(x - width/2, accuracy, width, label='准确率', 
                   color=colors, alpha=0.8, edgecolor='white', linewidth=1)
    bars2 = ax.bar(x + width/2, f1_scores, width, label='F1分数', 
                   color=colors, alpha=0.6, edgecolor='white', linewidth=1)
    
    # 添加数值标签
    for i, (bar1, bar2, acc, f1) in enumerate(zip(bars1, bars2, accuracy, f1_scores)):
        ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontproperties=custom_font, 
                fontsize=get_academic_config('font_sizes', 'annotation'))
        ax.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.01,
                f'{f1:.3f}', ha='center', va='bottom', fontproperties=custom_font,
                fontsize=get_academic_config('font_sizes', 'annotation'))
    
    ax.set_xlabel('情感分析算法', fontproperties=custom_font, 
                  fontsize=get_academic_config('font_sizes', 'label'))
    ax.set_ylabel('性能得分', fontproperties=custom_font,
                  fontsize=get_academic_config('font_sizes', 'label'))
    ax.set_title('中药材评论情感分析算法性能比较', fontproperties=custom_font, 
                 fontsize=get_academic_config('font_sizes', 'title'), pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, fontproperties=custom_font, 
                       fontsize=get_academic_config('font_sizes', 'tick'))
    ax.legend(prop=custom_font, fontsize=get_academic_config('font_sizes', 'legend'))
    
    # 应用统一主题
    apply_unified_theme(ax, grid=True, spine_style='clean')
    ax.set_ylim(0, 1.0)
    
    # 添加最佳性能标注
    best_idx = np.argmax(accuracy)
    ax.annotate(f'最佳性能: {algorithms[best_idx]}\n准确率: {accuracy[best_idx]:.1%}', 
                xy=(best_idx - width/2, accuracy[best_idx]), 
                xytext=(best_idx + 1, accuracy[best_idx] + 0.1),
                arrowprops=dict(arrowstyle='->', color=UNIFIED_COLORS['accent_green'], lw=1.5),
                fontproperties=custom_font, fontsize=get_academic_config('font_sizes', 'annotation'),
                bbox=dict(boxstyle="round,pad=0.3", facecolor=UNIFIED_COLORS['light_green'], alpha=0.8))
    
    plt.tight_layout()
    return save_unified_figure(fig, 'algorithm_comparison.png', '算法性能比较')

# 2. 供应链三维度评分比较 (图5-1)
def create_dimension_scores():
    """生成维度评分比较图表 - 使用统一配色"""
    dimensions = ['上游维度\n(供应商服务)', '中游维度\n(物流配送)', '下游维度\n(售后服务)']
    scores = [4.12, 3.87, 4.05]  # 基于论文第5章数据
    
    fig, ax = plt.subplots(figsize=get_academic_config('figure_sizes', 'medium'))
    
    # 使用分布图配色
    colors = get_color_palette('distribution', 3)
    
    bars = ax.bar(dimensions, scores, color=colors, alpha=0.8, 
                  edgecolor='white', linewidth=2, width=0.6)
    
    # 添加数值标签
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.05,
                f'{score:.2f}', ha='center', va='bottom', fontproperties=custom_font,
                fontsize=get_academic_config('font_sizes', 'annotation'), fontweight='bold')
    
    ax.set_ylabel('平均得分', fontproperties=custom_font,
                  fontsize=get_academic_config('font_sizes', 'label'))
    ax.set_title('中药材电商供应链服务质量三维度评分比较', fontproperties=custom_font,
                 fontsize=get_academic_config('font_sizes', 'title'), pad=20)
    
    # 应用统一主题
    apply_unified_theme(ax, grid=True, spine_style='minimal')
    ax.set_ylim(0, 5)
    
    # 添加评分标准线
    ax.axhline(y=4.0, color=UNIFIED_COLORS['medium_gray'], linestyle='--', alpha=0.7, linewidth=1)
    ax.text(0.02, 4.05, '良好标准线 (4.0分)', fontproperties=custom_font, 
            fontsize=get_academic_config('font_sizes', 'annotation'),
            transform=ax.get_yaxis_transform(), color=UNIFIED_COLORS['dark_gray'])
    
    plt.tight_layout()
    return save_unified_figure(fig, 'dimension_scores.png', '维度评分比较')

# 3. 回归分析结果图 (图5-14)
def create_regression_results():
    """生成回归分析结果图表 - 使用统一配色"""
    # 基于论文表5-9的数据
    variables = ['上游维度', '中游维度', '下游维度', '常数项']
    coefficients = [0.412, 0.356, 0.287, 1.128]
    std_errors = [0.026, 0.024, 0.022, 0.089]
    t_values = [15.85, 14.83, 13.05, 12.67]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=get_academic_config('figure_sizes', 'large'))
    
    # 子图1: 回归系数
    regression_colors = get_color_palette('regression')
    bars1 = ax1.barh(variables, coefficients, color=regression_colors['bars'], alpha=0.8)
    
    # 添加误差线
    ax1.errorbar(coefficients, range(len(variables)), xerr=std_errors, 
                fmt='none', ecolor=regression_colors['error_bars'], 
                capsize=5, capthick=2, alpha=0.8)
    
    # 添加数值标签
    for i, (coef, se) in enumerate(zip(coefficients, std_errors)):
        ax1.text(coef + se + 0.05, i, f'{coef:.3f}±{se:.3f}', 
                va='center', fontproperties=custom_font,
                fontsize=get_academic_config('font_sizes', 'annotation'))
    
    ax1.set_xlabel('回归系数', fontproperties=custom_font,
                   fontsize=get_academic_config('font_sizes', 'label'))
    ax1.set_title('供应链服务质量回归系数', fontproperties=custom_font,
                  fontsize=get_academic_config('font_sizes', 'title'))
    apply_unified_theme(ax1, grid=True, spine_style='clean')
    
    # 子图2: t统计量
    bars2 = ax2.barh(variables, t_values, color=regression_colors['bars'], alpha=0.6)
    
    # 添加显著性标准线
    ax2.axvline(x=1.96, color=UNIFIED_COLORS['alert_red'], linestyle='--', alpha=0.7, linewidth=2)
    ax2.text(1.96, len(variables), '95%显著性\n(t=1.96)', ha='center', va='bottom',
             fontproperties=custom_font, fontsize=get_academic_config('font_sizes', 'annotation'),
             color=UNIFIED_COLORS['alert_red'])
    
    # 添加数值标签
    for i, t_val in enumerate(t_values):
        ax2.text(t_val + 0.5, i, f'{t_val:.2f}', va='center', 
                fontproperties=custom_font,
                fontsize=get_academic_config('font_sizes', 'annotation'))
    
    ax2.set_xlabel('t统计量', fontproperties=custom_font,
                   fontsize=get_academic_config('font_sizes', 'label'))
    ax2.set_title('统计显著性检验', fontproperties=custom_font,
                  fontsize=get_academic_config('font_sizes', 'title'))
    apply_unified_theme(ax2, grid=True, spine_style='clean')
    
    # 整体标题
    fig.suptitle('中药材电商供应链服务质量影响因素回归分析', 
                 fontproperties=custom_font, fontsize=get_academic_config('font_sizes', 'title'))
    
    plt.tight_layout()
    return save_unified_figure(fig, 'regression_results.png', '回归分析结果')

# 4. 情感分布图
def create_sentiment_distribution():
    """生成情感分布图 - 使用统一配色"""
    labels = ['正面评价', '中性评价', '负面评价']
    sizes = [75.8, 11.5, 12.7]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=get_academic_config('figure_sizes', 'large'))
    
    # 使用情感分析专用配色
    sentiment_colors = get_color_palette('sentiment')
    colors = [sentiment_colors['positive'], sentiment_colors['neutral'], sentiment_colors['negative']]
    
    # 子图1: 饼图
    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                       startangle=90, textprops={'fontproperties': custom_font})
    
    # 美化百分比文字
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(get_academic_config('font_sizes', 'annotation'))
    
    ax1.set_title('中药材评论情感分布', fontproperties=custom_font,
                  fontsize=get_academic_config('font_sizes', 'title'))
    
    # 子图2: 柱状图
    bars = ax2.bar(labels, sizes, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
    
    # 添加数值标签
    for bar, size in zip(bars, sizes):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 1,
                f'{size}%\n({int(size*2349):,}条)', ha='center', va='bottom',
                fontproperties=custom_font, fontsize=get_academic_config('font_sizes', 'annotation'))
    
    ax2.set_ylabel('百分比 (%)', fontproperties=custom_font,
                   fontsize=get_academic_config('font_sizes', 'label'))
    ax2.set_title('评论数量统计', fontproperties=custom_font,
                  fontsize=get_academic_config('font_sizes', 'title'))
    apply_unified_theme(ax2, grid=True, spine_style='minimal')
    ax2.set_ylim(0, 85)
    
    plt.tight_layout()
    return save_unified_figure(fig, 'sentiment_distribution.png', '情感分布')

# 5. 平台分布图
def create_platform_distribution():
    """生成平台分布图 - 使用统一配色"""
    platforms = ['淘宝', '京东', '天猫', '其他平台']
    percentages = [40.1, 32.1, 19.8, 8.0]
    counts = [85000, 68000, 42000, 17000]
    
    fig, ax = plt.subplots(figsize=get_academic_config('figure_sizes', 'medium'))
    
    # 使用分布图配色
    colors = get_color_palette('distribution', len(platforms))
    
    bars = ax.bar(platforms, percentages, color=colors, alpha=0.8, 
                  edgecolor='white', linewidth=2)
    
    # 添加数值标签
    for bar, pct, count in zip(bars, percentages, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.8,
                f'{pct}%\n({count:,}条)', ha='center', va='bottom',
                fontproperties=custom_font, fontsize=get_academic_config('font_sizes', 'annotation'))
    
    ax.set_ylabel('占比 (%)', fontproperties=custom_font,
                  fontsize=get_academic_config('font_sizes', 'label'))
    ax.set_title('中药材评论数据平台分布', fontproperties=custom_font,
                 fontsize=get_academic_config('font_sizes', 'title'), pad=20)
    
    apply_unified_theme(ax, grid=True, spine_style='minimal')
    ax.set_ylim(0, 50)
    
    plt.tight_layout()
    return save_unified_figure(fig, 'platform_distribution.png', '平台分布')

# 6. 企业规模分布图
def create_enterprise_distribution():
    """生成企业规模分布图 - 使用统一配色"""
    enterprise_types = ['大型企业', '中型企业', '小型企业', '个体商户']
    percentages = [30, 40, 20, 10]
    
    fig, ax = plt.subplots(figsize=get_academic_config('figure_sizes', 'medium'))
    
    # 使用分布图配色
    colors = get_color_palette('distribution', len(enterprise_types))
    
    bars = ax.bar(enterprise_types, percentages, color=colors, alpha=0.8,
                  edgecolor='white', linewidth=2)
    
    # 添加数值标签
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                f'{pct}%', ha='center', va='bottom', fontproperties=custom_font,
                fontsize=get_academic_config('font_sizes', 'annotation'), fontweight='bold')
    
    ax.set_ylabel('占比 (%)', fontproperties=custom_font,
                  fontsize=get_academic_config('font_sizes', 'label'))
    ax.set_title('中药材电商企业规模分布', fontproperties=custom_font,
                 fontsize=get_academic_config('font_sizes', 'title'), pad=20)
    
    apply_unified_theme(ax, grid=True, spine_style='minimal')
    ax.set_ylim(0, 50)
    
    plt.tight_layout()
    return save_unified_figure(fig, 'enterprise_distribution.png', '企业规模分布')

# 7. 产品类型分布图
def create_product_distribution():
    """生成产品类型分布图 - 使用统一配色"""
    product_types = ['中药原材料', '中药饮片', '中药配方颗粒', '其他产品']
    percentages = [45, 35, 15, 5]
    
    fig, ax = plt.subplots(figsize=get_academic_config('figure_sizes', 'medium'))
    
    # 使用分布图配色
    colors = get_color_palette('distribution', len(product_types))
    
    bars = ax.bar(product_types, percentages, color=colors, alpha=0.8,
                  edgecolor='white', linewidth=2)
    
    # 添加数值标签
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.8,
                f'{pct}%', ha='center', va='bottom', fontproperties=custom_font,
                fontsize=get_academic_config('font_sizes', 'annotation'), fontweight='bold')
    
    ax.set_ylabel('占比 (%)', fontproperties=custom_font,
                  fontsize=get_academic_config('font_sizes', 'label'))
    ax.set_title('中药材产品类型分布', fontproperties=custom_font,
                 fontsize=get_academic_config('font_sizes', 'title'), pad=20)
    
    apply_unified_theme(ax, grid=True, spine_style='minimal')
    ax.set_ylim(0, 50)
    
    plt.tight_layout()
    return save_unified_figure(fig, 'product_distribution.png', '产品类型分布')

# 8. 时间序列图
def create_sentiment_time_series():
    """生成情感时间序列图 - 使用统一配色"""
    months = ['2024-01', '2024-02', '2024-03', '2024-04', '2024-05', '2024-06']
    positive_trend = [74.2, 75.1, 76.8, 75.5, 76.2, 75.8]
    negative_trend = [13.5, 12.8, 11.9, 12.4, 12.1, 12.7]
    
    fig, ax = plt.subplots(figsize=get_academic_config('figure_sizes', 'large'))
    
    # 使用情感分析配色
    sentiment_colors = get_color_palette('sentiment')
    
    ax.plot(months, positive_trend, marker='o', linewidth=get_academic_config('line_widths', 'thick'),
            markersize=get_academic_config('marker_sizes', 'medium'), 
            color=sentiment_colors['positive'], label='正面情感', alpha=0.9)
    ax.plot(months, negative_trend, marker='s', linewidth=get_academic_config('line_widths', 'thick'),
            markersize=get_academic_config('marker_sizes', 'medium'), 
            color=sentiment_colors['negative'], label='负面情感', alpha=0.9)
    
    ax.set_xlabel('时间 (月份)', fontproperties=custom_font,
                  fontsize=get_academic_config('font_sizes', 'label'))
    ax.set_ylabel('情感倾向占比 (%)', fontproperties=custom_font,
                  fontsize=get_academic_config('font_sizes', 'label'))
    ax.set_title('中药材评论情感趋势变化 (2024年1-6月)', fontproperties=custom_font,
                 fontsize=get_academic_config('font_sizes', 'title'), pad=20)
    
    ax.legend(prop=custom_font, fontsize=get_academic_config('font_sizes', 'legend'))
    apply_unified_theme(ax, grid=True, spine_style='clean')
    
    # 旋转x轴标签
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return save_unified_figure(fig, 'sentiment_time_series.png', '情感时间序列')

# 9. 月度分布图
def create_monthly_distribution():
    """生成月度分布趋势图 - 使用统一配色"""
    months = ['1月', '2月', '3月', '4月', '5月', '6月']
    comment_counts = [35420, 28350, 42180, 39650, 44280, 45000]
    
    fig, ax = plt.subplots(figsize=get_academic_config('figure_sizes', 'large'))
    
    # 使用主要蓝色配色
    color = UNIFIED_COLORS['primary_blue']
    
    bars = ax.bar(months, comment_counts, color=color, alpha=0.8,
                  edgecolor='white', linewidth=2)
    
    # 添加数值标签
    for bar, count in zip(bars, comment_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 500,
                f'{count:,}', ha='center', va='bottom', fontproperties=custom_font,
                fontsize=get_academic_config('font_sizes', 'annotation'))
    
    ax.set_ylabel('评论数量', fontproperties=custom_font,
                  fontsize=get_academic_config('font_sizes', 'label'))
    ax.set_title('2024年中药材评论月度分布趋势', fontproperties=custom_font,
                 fontsize=get_academic_config('font_sizes', 'title'), pad=20)
    
    apply_unified_theme(ax, grid=True, spine_style='minimal')
    
    # 格式化y轴
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))
    
    plt.tight_layout()
    return save_unified_figure(fig, 'monthly_distribution.png', '月度分布趋势')

# 10. 综合数据概览仪表板
def create_data_overview_dashboard():
    """生成数据概览综合仪表板 - 使用统一配色"""
    fig, axes = plt.subplots(2, 4, figsize=get_academic_config('figure_sizes', 'dashboard'))
    axes = axes.flatten()
    
    # 获取各种配色方案
    sentiment_colors = get_color_palette('sentiment')
    distribution_colors = get_color_palette('distribution', 4)
    
    # 1. 情感分布饼图
    ax = axes[0]
    sizes = [75.8, 11.5, 12.7]
    colors = [sentiment_colors['positive'], sentiment_colors['neutral'], sentiment_colors['negative']]
    ax.pie(sizes, labels=['正面', '中性', '负面'], colors=colors, autopct='%1.1f%%',
           textprops={'fontproperties': custom_font, 'fontsize': 9})
    ax.set_title('情感分布', fontproperties=custom_font, fontsize=11, pad=10)
    
    # 2. 平台分布
    ax = axes[1]
    platforms = ['淘宝', '京东', '天猫', '其他']
    platform_data = [40.1, 32.1, 19.8, 8.0]
    bars = ax.bar(platforms, platform_data, color=distribution_colors, alpha=0.8)
    ax.set_title('平台分布 (%)', fontproperties=custom_font, fontsize=11, pad=10)
    ax.set_xticklabels(platforms, fontproperties=custom_font, fontsize=8)
    apply_unified_theme(ax, grid=False, spine_style='minimal')
    
    # 3. 企业规模
    ax = axes[2]
    enterprise_data = [30, 40, 20, 10]
    enterprise_labels = ['大型', '中型', '小型', '个体']
    bars = ax.bar(enterprise_labels, enterprise_data, color=distribution_colors, alpha=0.8)
    ax.set_title('企业规模 (%)', fontproperties=custom_font, fontsize=11, pad=10)
    ax.set_xticklabels(enterprise_labels, fontproperties=custom_font, fontsize=8)
    apply_unified_theme(ax, grid=False, spine_style='minimal')
    
    # 4. 产品类型
    ax = axes[3]
    product_data = [45, 35, 15, 5]
    product_labels = ['原材料', '饮片', '颗粒', '其他']
    bars = ax.bar(product_labels, product_data, color=distribution_colors, alpha=0.8)
    ax.set_title('产品类型 (%)', fontproperties=custom_font, fontsize=11, pad=10)
    ax.set_xticklabels(product_labels, fontproperties=custom_font, fontsize=8, rotation=45)
    apply_unified_theme(ax, grid=False, spine_style='minimal')
    
    # 5. 算法性能对比
    ax = axes[4]
    algorithms = ['BERT', '混合', 'LSTM', 'TextCNN']
    performance = [0.883, 0.910, 0.876, 0.852]
    colors_perf = get_color_palette('algorithm_performance', 4)
    bars = ax.bar(algorithms, performance, color=colors_perf, alpha=0.8)
    ax.set_title('算法性能 (准确率)', fontproperties=custom_font, fontsize=11, pad=10)
    ax.set_xticklabels(algorithms, fontproperties=custom_font, fontsize=8, rotation=45)
    apply_unified_theme(ax, grid=False, spine_style='minimal')
    ax.set_ylim(0.8, 0.95)
    
    # 6. 维度评分
    ax = axes[5]
    dimensions = ['上游', '中游', '下游']
    dim_scores = [4.12, 3.87, 4.05]
    bars = ax.bar(dimensions, dim_scores, color=distribution_colors[:3], alpha=0.8)
    ax.set_title('维度评分', fontproperties=custom_font, fontsize=11, pad=10)
    ax.set_xticklabels(dimensions, fontproperties=custom_font, fontsize=9)
    apply_unified_theme(ax, grid=False, spine_style='minimal')
    ax.set_ylim(3.5, 4.5)
    
    # 7. 月度趋势
    ax = axes[6]
    months = ['1月', '2月', '3月', '4月', '5月', '6月']
    trend = [35.4, 28.4, 42.2, 39.7, 44.3, 45.0]
    ax.plot(months, trend, marker='o', color=UNIFIED_COLORS['primary_blue'], 
            linewidth=2, markersize=4)
    ax.set_title('月度趋势 (K条)', fontproperties=custom_font, fontsize=11, pad=10)
    ax.set_xticklabels(months, fontproperties=custom_font, fontsize=8, rotation=45)
    apply_unified_theme(ax, grid=False, spine_style='minimal')
    
    # 8. 关键统计信息
    ax = axes[7]
    ax.axis('off')
    stats_text = f"""数据统计摘要

📊 总评论数: 234,880条
📈 数据时间: 2024年1-6月
🏪 覆盖平台: 4个主要平台
🏢 企业数量: 1,200+家
💊 产品类型: 4大类别

🎯 最佳算法: 混合模型
   准确率: 91.0%
   F1分数: 90.8%

📋 质量评分:
   上游维度: 4.12分
   中游维度: 3.87分  
   下游维度: 4.05分

✨ 正面评价率: 75.8%"""
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontproperties=custom_font,
            fontsize=9, verticalalignment='top', linespacing=1.2,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=UNIFIED_COLORS['light_blue'], alpha=0.3))
    
    # 总标题
    fig.suptitle('中药材企业电商供应链服务质量评价 - 数据概览仪表板', 
                 fontproperties=custom_font, fontsize=16, y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    return save_unified_figure(fig, 'data_overview_dashboard.png', '数据概览仪表板')

def generate_all_figures():
    """生成所有图表"""
    print("=" * 60)
    print("开始生成所有图表 (使用统一配色方案)")
    print("=" * 60)
    
    figures = []
    
    # 生成所有图表
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
    print(f"✓ 所有图表生成完成！共生成 {len(figures)} 个图表")
    print("✓ 所有图表使用统一的配色方案")
    print("✓ 图表保存在: output/figures/")
    print("=" * 60)
    
    return figures

if __name__ == "__main__":
    generate_all_figures()
