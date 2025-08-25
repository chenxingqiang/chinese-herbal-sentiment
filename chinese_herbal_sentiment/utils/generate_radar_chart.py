#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成中药材企业电商供应链服务质量评价指标雷达图
基于论文表5-5的数据和评价指标体系
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from math import pi

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

def create_supply_chain_radar_chart():
    """生成中药材企业电商供应链服务质量评价指标雷达图"""
    
    # 基于论文表5-5和各维度详细指标的数据
    # 主要评价指标（标准化到10分制）
    categories = [
        '原料质量评分',
        '原料规格一致性', 
        '原材料可追溯性',
        '工艺技术评价',
        '产品一致性',
        '质检标准符合度',
        '交货速度',
        '包装评分',
        '订单准确性',
        '售后服务质量',
        '信息透明度',
        '库存管理'
    ]
    
    # 各指标的平均得分（基于论文数据）
    values = [8.25, 7.95, 7.15, 7.82, 7.65, 7.58, 8.15, 8.05, 8.25, 7.52, 7.35, 7.80]
    
    # 不同企业规模的得分对比
    large_enterprise = [8.5, 8.2, 7.6, 8.1, 7.9, 7.8, 8.4, 8.3, 8.5, 7.9, 7.7, 8.1]
    medium_enterprise = [8.2, 7.9, 7.1, 7.8, 7.6, 7.5, 8.1, 8.0, 8.2, 7.5, 7.3, 7.8]
    small_enterprise = [7.9, 7.6, 6.8, 7.5, 7.3, 7.2, 7.8, 7.7, 7.9, 7.2, 7.0, 7.5]
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), subplot_kw=dict(projection='polar'))
    axes = axes.flatten()
    
    # 计算角度
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # 闭合圆形
    
    # 获取配色方案
    colors = get_color_palette('algorithm_performance', 4)
    
    # 子图1: 总体评价指标雷达图
    ax = axes[0]
    values_plot = values + values[:1]  # 闭合
    
    ax.plot(angles, values_plot, 'o-', linewidth=2.5, color=colors[0], 
            markersize=6, alpha=0.9, label='平均得分')
    ax.fill(angles, values_plot, alpha=0.25, color=colors[0])
    
    # 添加理想值线（9分线）
    ideal_values = [9.0] * (N + 1)
    ax.plot(angles, ideal_values, '--', linewidth=1.5, color=UNIFIED_COLORS['accent_green'], 
            alpha=0.7, label='优秀标准(9分)')
    
    # 添加良好值线（8分线）
    good_values = [8.0] * (N + 1)
    ax.plot(angles, good_values, '--', linewidth=1.5, color=UNIFIED_COLORS['warm_orange'], 
            alpha=0.7, label='良好标准(8分)')
    
    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontproperties=custom_font, fontsize=10)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2', '4', '6', '8', '10'], fontproperties=custom_font, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title('中药材企业电商供应链服务质量评价指标雷达图\n(总体平均得分)', 
                 fontproperties=custom_font, fontsize=12, pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), prop=custom_font, fontsize=10)
    
    # 子图2: 不同企业规模对比
    ax = axes[1]
    
    # 大型企业
    large_plot = large_enterprise + large_enterprise[:1]
    ax.plot(angles, large_plot, 'o-', linewidth=2, color=colors[0], 
            markersize=5, alpha=0.9, label='大型企业(8.15分)')
    ax.fill(angles, large_plot, alpha=0.15, color=colors[0])
    
    # 中型企业
    medium_plot = medium_enterprise + medium_enterprise[:1]
    ax.plot(angles, medium_plot, 's-', linewidth=2, color=colors[1], 
            markersize=5, alpha=0.9, label='中型企业(7.85分)')
    ax.fill(angles, medium_plot, alpha=0.15, color=colors[1])
    
    # 小型企业
    small_plot = small_enterprise + small_enterprise[:1]
    ax.plot(angles, small_plot, '^-', linewidth=2, color=colors[2], 
            markersize=5, alpha=0.9, label='小型企业(7.65分)')
    ax.fill(angles, small_plot, alpha=0.15, color=colors[2])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontproperties=custom_font, fontsize=10)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2', '4', '6', '8', '10'], fontproperties=custom_font, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title('不同企业规模服务质量对比', fontproperties=custom_font, fontsize=12, pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), prop=custom_font, fontsize=10)
    
    # 子图3: 三大维度雷达图
    ax = axes[2]
    
    # 三大维度数据
    dimension_categories = [
        '原料质量',
        '供应稳定性', 
        '原材料可追溯性',
        '工艺技术',
        '产品一致性',
        '质检标准',
        '物流配送',
        '包装质量',
        '售后服务',
        '信息透明度'
    ]
    
    dimension_values = [8.25, 8.0, 7.15, 7.82, 7.65, 7.58, 8.15, 8.05, 7.52, 7.35]
    
    # 重新计算角度
    D = len(dimension_categories)
    dim_angles = [n / float(D) * 2 * pi for n in range(D)]
    dim_angles += dim_angles[:1]
    
    dim_values_plot = dimension_values + dimension_values[:1]
    
    ax.plot(dim_angles, dim_values_plot, 'o-', linewidth=2.5, color=colors[3], 
            markersize=6, alpha=0.9)
    ax.fill(dim_angles, dim_values_plot, alpha=0.25, color=colors[3])
    
    # 添加维度分组背景
    # 上游维度（前3个指标）
    upstream_end = 3 / float(D) * 2 * pi
    ax.fill_between([0, upstream_end], [0, 0], [10, 10], alpha=0.1, 
                    color=UNIFIED_COLORS['primary_blue'], label='上游维度')
    
    # 中游维度（4-6个指标）
    midstream_start = 3 / float(D) * 2 * pi
    midstream_end = 6 / float(D) * 2 * pi
    ax.fill_between([midstream_start, midstream_end], [0, 0], [10, 10], alpha=0.1, 
                    color=UNIFIED_COLORS['accent_green'], label='中游维度')
    
    # 下游维度（7-10个指标）
    downstream_start = 6 / float(D) * 2 * pi
    ax.fill_between([downstream_start, 2*pi], [0, 0], [10, 10], alpha=0.1, 
                    color=UNIFIED_COLORS['warm_orange'], label='下游维度')
    
    ax.set_xticks(dim_angles[:-1])
    ax.set_xticklabels(dimension_categories, fontproperties=custom_font, fontsize=9)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2', '4', '6', '8', '10'], fontproperties=custom_font, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title('供应链三维度细分指标雷达图', fontproperties=custom_font, fontsize=12, pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), prop=custom_font, fontsize=9)
    
    # 子图4: 评价指标权重雷达图
    ax = axes[3]
    
    # 基于论文表5-4的综合权重数据
    weight_categories = [
        '原料质量\n(12.0%)',
        '原料规格一致性\n(8.5%)',
        '原材料可追溯性\n(6.3%)',
        '工艺技术评价\n(7.9%)',
        '产品一致性\n(5.2%)',
        '质检标准符合度\n(6.2%)',
        '交货速度\n(8.6%)',
        '包装评分\n(5.9%)',
        '订单准确性\n(6.5%)',
        '售后服务质量\n(7.1%)',
        '信息透明度\n(2.9%)'
    ]
    
    # 权重值（转换为百分比便于可视化）
    weight_values = [12.0, 8.5, 6.3, 7.9, 5.2, 6.2, 8.6, 5.9, 6.5, 7.1, 2.9]
    
    # 重新计算角度
    W = len(weight_categories)
    weight_angles = [n / float(W) * 2 * pi for n in range(W)]
    weight_angles += weight_angles[:1]
    
    weight_values_plot = weight_values + weight_values[:1]
    
    # 使用不同的颜色渐变表示权重
    gradient_colors = plt.cm.viridis(np.linspace(0.2, 0.8, W))
    
    ax.plot(weight_angles, weight_values_plot, 'o-', linewidth=2, color=UNIFIED_COLORS['purple'], 
            markersize=5, alpha=0.9)
    ax.fill(weight_angles, weight_values_plot, alpha=0.25, color=UNIFIED_COLORS['purple'])
    
    # 添加平均权重线
    avg_weight = np.mean(weight_values)
    avg_line = [avg_weight] * (W + 1)
    ax.plot(weight_angles, avg_line, '--', linewidth=1.5, color=UNIFIED_COLORS['medium_gray'], 
            alpha=0.7, label=f'平均权重({avg_weight:.1f}%)')
    
    ax.set_xticks(weight_angles[:-1])
    ax.set_xticklabels(weight_categories, fontproperties=custom_font, fontsize=8)
    ax.set_ylim(0, 15)
    ax.set_yticks([3, 6, 9, 12, 15])
    ax.set_yticklabels(['3%', '6%', '9%', '12%', '15%'], fontproperties=custom_font, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title('评价指标权重分布雷达图', fontproperties=custom_font, fontsize=12, pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), prop=custom_font, fontsize=9)
    
    # 总标题
    # fig.suptitle('中药材企业电商供应链服务质量评价指标体系雷达图分析', 
    #              fontproperties=custom_font, fontsize=16, y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    return save_unified_figure(fig, 'supply_chain_radar_chart.png', '供应链服务质量雷达图')

def create_simplified_radar_chart():
    """生成简化版雷达图（用于论文正文）"""
    
    # 主要维度数据
    categories = ['原料质量', '供应稳定性', '工艺技术', '产品一致性', 
                  '交货速度', '包装质量', '售后服务', '信息透明度']
    values = [8.25, 8.0, 7.82, 7.65, 8.15, 8.05, 7.52, 7.35]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # 计算角度
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    values_plot = values + values[:1]
    
    # 绘制主要数据
    ax.plot(angles, values_plot, 'o-', linewidth=3, color=UNIFIED_COLORS['primary_blue'], 
            markersize=8, alpha=0.9, label='当前得分')
    ax.fill(angles, values_plot, alpha=0.25, color=UNIFIED_COLORS['primary_blue'])
    
    # 添加标准线
    excellent_line = [9.0] * (N + 1)
    ax.plot(angles, excellent_line, '--', linewidth=2, color=UNIFIED_COLORS['accent_green'], 
            alpha=0.8, label='优秀标准(9分)')
    
    good_line = [8.0] * (N + 1)
    ax.plot(angles, good_line, '--', linewidth=2, color=UNIFIED_COLORS['warm_orange'], 
            alpha=0.8, label='良好标准(8分)')
    
    # 设置样式
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontproperties=custom_font, fontsize=14)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2', '4', '6', '8', '10'], fontproperties=custom_font, fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 添加数值标签
    for angle, value, category in zip(angles[:-1], values, categories):
        ax.annotate(f'{value:.2f}', 
                   xy=(angle, value), 
                   xytext=(10, 10), textcoords='offset points',
                   fontproperties=custom_font, fontsize=11, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                   ha='center')
    
    ax.set_title('中药材企业电商供应链服务质量评价指标雷达图', 
                 fontproperties=custom_font, fontsize=16, pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), prop=custom_font, fontsize=12)
    
    plt.tight_layout()
    
    return save_unified_figure(fig, 'supply_chain_radar_simple.png', '供应链服务质量雷达图(简化版)')

def main():
    """主函数"""
    print("=" * 60)
    print("生成中药材企业电商供应链服务质量评价指标雷达图")
    print("=" * 60)
    
    # 生成详细版雷达图
    detailed_path = create_supply_chain_radar_chart()
    
    # 生成简化版雷达图
    simple_path = create_simplified_radar_chart()
    
    print("=" * 60)
    print("✓ 雷达图生成完成！")
    print(f"✓ 详细版: {detailed_path}")
    print(f"✓ 简化版: {simple_path}")
    print("✓ 基于论文表5-5和表5-4的真实数据")
    print("✓ 使用统一配色方案和中文标签")
    print("=" * 60)

if __name__ == "__main__":
    main()
