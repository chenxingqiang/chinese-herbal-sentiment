#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一的配色方案配置
为论文中所有图表定义统一的颜色标准
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# 统一的学术论文配色方案
UNIFIED_COLORS = {
    # 主要颜色（基于学术论文标准）
    'primary_blue': '#1976D2',      # 主蓝色 - 用于主要数据系列
    'secondary_blue': '#42A5F5',    # 辅助蓝色 - 用于次要数据系列
    'accent_green': '#388E3C',      # 强调绿色 - 用于成功/正向指标
    'warm_orange': '#F57C00',       # 暖橙色 - 用于警告/中性指标
    'alert_red': '#D32F2F',         # 警示红色 - 用于错误/负向指标
    
    # 扩展颜色调色板（用于多系列图表）
    'purple': '#7B1FA2',            # 紫色
    'teal': '#00796B',              # 青绿色
    'indigo': '#303F9F',            # 靛蓝色
    'brown': '#5D4037',             # 棕色
    'pink': '#C2185B',              # 粉红色
    
    # 背景和辅助颜色
    'light_blue': '#E3F2FD',        # 浅蓝背景
    'light_green': '#E8F5E8',       # 浅绿背景
    'light_orange': '#FFF3E0',      # 浅橙背景
    'light_red': '#FFEBEE',         # 浅红背景
    'light_gray': '#F5F5F5',        # 浅灰背景
    'medium_gray': '#BDBDBD',       # 中灰色
    'dark_gray': '#424242',         # 深灰色
    
    # 特殊用途颜色
    'grid_color': '#E0E0E0',        # 网格线颜色
    'text_color': '#212121',        # 主文本颜色
    'annotation_color': '#757575'   # 注释文本颜色
}

# 预定义的配色方案组合
COLOR_SCHEMES = {
    # 算法性能比较图配色
    'algorithm_performance': [
        UNIFIED_COLORS['primary_blue'],
        UNIFIED_COLORS['accent_green'], 
        UNIFIED_COLORS['warm_orange'],
        UNIFIED_COLORS['purple'],
        UNIFIED_COLORS['teal'],
        UNIFIED_COLORS['alert_red'],
        UNIFIED_COLORS['indigo'],
        UNIFIED_COLORS['brown']
    ],
    
    # 训练过程图配色
    'training_process': {
        'train_line': UNIFIED_COLORS['primary_blue'],
        'val_line': UNIFIED_COLORS['accent_green'],
        'loss_line': UNIFIED_COLORS['warm_orange'],
        'accuracy_line': UNIFIED_COLORS['purple'],
        'f1_line': UNIFIED_COLORS['teal']
    },
    
    # 分布图配色
    'distribution': [
        UNIFIED_COLORS['primary_blue'],
        UNIFIED_COLORS['accent_green'],
        UNIFIED_COLORS['warm_orange'],
        UNIFIED_COLORS['purple'],
        UNIFIED_COLORS['teal']
    ],
    
    # 情感分析配色
    'sentiment': {
        'positive': UNIFIED_COLORS['accent_green'],
        'neutral': UNIFIED_COLORS['medium_gray'],
        'negative': UNIFIED_COLORS['alert_red']
    },
    
    # 回归分析配色
    'regression': {
        'bars': UNIFIED_COLORS['primary_blue'],
        'error_bars': UNIFIED_COLORS['dark_gray'],
        'significance': UNIFIED_COLORS['accent_green']
    }
}

def setup_unified_style():
    """设置统一的matplotlib样式"""
    # 基本字体配置
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 图表样式配置
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = UNIFIED_COLORS['dark_gray']
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['axes.labelcolor'] = UNIFIED_COLORS['text_color']
    plt.rcParams['text.color'] = UNIFIED_COLORS['text_color']
    
    # 网格配置
    plt.rcParams['grid.color'] = UNIFIED_COLORS['grid_color']
    plt.rcParams['grid.linestyle'] = '-'
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['grid.alpha'] = 0.7
    
    # 图例配置
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.facecolor'] = 'white'
    plt.rcParams['legend.edgecolor'] = UNIFIED_COLORS['medium_gray']
    plt.rcParams['legend.borderaxespad'] = 0.5
    
    # 坐标轴配置
    plt.rcParams['xtick.color'] = UNIFIED_COLORS['dark_gray']
    plt.rcParams['ytick.color'] = UNIFIED_COLORS['dark_gray']
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'

def get_custom_font():
    """获取自定义字体"""
    try:
        font_path = '/Users/xingqiangchen/Library/Fonts/AlibabaPuHuiTi-3-85-Bold.ttf'
        return fm.FontProperties(fname=font_path)
    except:
        return fm.FontProperties()

def get_color_palette(scheme_name, n_colors=None):
    """获取指定配色方案的颜色列表"""
    if scheme_name in COLOR_SCHEMES:
        colors = COLOR_SCHEMES[scheme_name]
        if isinstance(colors, list):
            if n_colors and n_colors > len(colors):
                # 如果需要更多颜色，循环使用
                return (colors * ((n_colors // len(colors)) + 1))[:n_colors]
            return colors[:n_colors] if n_colors else colors
        else:
            return colors
    else:
        # 默认返回算法性能配色
        return COLOR_SCHEMES['algorithm_performance'][:n_colors] if n_colors else COLOR_SCHEMES['algorithm_performance']

def apply_unified_theme(ax, grid=True, spine_style='default'):
    """为单个axes应用统一主题"""
    # 设置网格
    if grid:
        ax.grid(True, alpha=0.3, color=UNIFIED_COLORS['grid_color'])
        ax.set_axisbelow(True)
    
    # 设置边框样式
    if spine_style == 'minimal':
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(UNIFIED_COLORS['medium_gray'])
        ax.spines['bottom'].set_color(UNIFIED_COLORS['medium_gray'])
    elif spine_style == 'clean':
        for spine in ax.spines.values():
            spine.set_color(UNIFIED_COLORS['medium_gray'])
            spine.set_linewidth(0.8)
    
    # 设置刻度样式
    ax.tick_params(axis='both', colors=UNIFIED_COLORS['dark_gray'], labelsize=10)

def save_unified_figure(fig, filename, title="", dpi=300):
    """使用统一标准保存图片"""
    import os
    
    output_path = os.path.join('output', 'figures', filename)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"✓ 已生成 (统一配色): {title} -> {output_path}")
    return output_path

# 学术论文图表标准配置
ACADEMIC_CONFIG = {
    'figure_sizes': {
        'small': (8, 5),
        'medium': (10, 6),
        'large': (12, 8),
        'dashboard': (16, 12)
    },
    'dpi': 300,
    'font_sizes': {
        'title': 14,
        'label': 12,
        'legend': 11,
        'tick': 10,
        'annotation': 9
    },
    'line_widths': {
        'thin': 1.0,
        'medium': 1.5,
        'thick': 2.0,
        'extra_thick': 2.5
    },
    'marker_sizes': {
        'small': 4,
        'medium': 6,
        'large': 8
    }
}

def get_academic_config(key, subkey=None):
    """获取学术论文配置参数"""
    if subkey:
        return ACADEMIC_CONFIG.get(key, {}).get(subkey)
    return ACADEMIC_CONFIG.get(key)

# 预览配色方案的函数
def preview_color_schemes():
    """生成配色方案预览图"""
    setup_unified_style()
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    custom_font = get_custom_font()
    
    # 1. 算法性能配色
    ax = axes[0]
    colors = get_color_palette('algorithm_performance', 8)
    bars = ax.bar(range(8), [1]*8, color=colors, alpha=0.8)
    ax.set_title('算法性能比较配色', fontproperties=custom_font, fontsize=12)
    ax.set_xticks(range(8))
    ax.set_xticklabels(['算法1', '算法2', '算法3', '算法4', '算法5', '算法6', '算法7', '算法8'], 
                       fontproperties=custom_font, fontsize=9, rotation=45)
    apply_unified_theme(ax)
    
    # 2. 训练过程配色
    ax = axes[1]
    x = range(20)
    training_colors = get_color_palette('training_process')
    ax.plot(x, np.sin(np.array(x)*0.3), color=training_colors['train_line'], linewidth=2, label='训练')
    ax.plot(x, np.cos(np.array(x)*0.3), color=training_colors['val_line'], linewidth=2, label='验证')
    ax.set_title('训练过程配色', fontproperties=custom_font, fontsize=12)
    ax.legend(prop=custom_font)
    apply_unified_theme(ax)
    
    # 3. 分布图配色
    ax = axes[2]
    colors = get_color_palette('distribution', 5)
    sizes = [30, 25, 20, 15, 10]
    ax.pie(sizes, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title('分布图配色', fontproperties=custom_font, fontsize=12)
    
    # 4. 情感分析配色
    ax = axes[3]
    sentiment_colors = get_color_palette('sentiment')
    sentiments = ['正面', '中性', '负面']
    values = [75.8, 11.5, 12.7]
    colors = [sentiment_colors['positive'], sentiment_colors['neutral'], sentiment_colors['negative']]
    bars = ax.bar(sentiments, values, color=colors, alpha=0.8)
    ax.set_title('情感分析配色', fontproperties=custom_font, fontsize=12)
    ax.set_ylabel('百分比 (%)', fontproperties=custom_font)
    for i, (bar, value) in enumerate(zip(bars, values)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{value}%', ha='center', va='bottom', fontproperties=custom_font, fontsize=10)
    apply_unified_theme(ax)
    
    # 5. 回归分析配色
    ax = axes[4]
    regression_colors = get_color_palette('regression')
    variables = ['上游', '中游', '下游']
    coefficients = [0.412, 0.356, 0.287]
    bars = ax.bar(variables, coefficients, color=regression_colors['bars'], alpha=0.8)
    ax.set_title('回归分析配色', fontproperties=custom_font, fontsize=12)
    ax.set_ylabel('回归系数', fontproperties=custom_font)
    apply_unified_theme(ax)
    
    # 6. 颜色代码表
    ax = axes[5]
    ax.axis('off')
    color_info = f"""统一配色方案代码

🔵 主要蓝色: {UNIFIED_COLORS['primary_blue']}
🔷 辅助蓝色: {UNIFIED_COLORS['secondary_blue']}
🟢 强调绿色: {UNIFIED_COLORS['accent_green']}
🟠 暖橙色: {UNIFIED_COLORS['warm_orange']}
🔴 警示红色: {UNIFIED_COLORS['alert_red']}

🟣 紫色: {UNIFIED_COLORS['purple']}
🔶 青绿色: {UNIFIED_COLORS['teal']}
🟦 靛蓝色: {UNIFIED_COLORS['indigo']}
🟤 棕色: {UNIFIED_COLORS['brown']}
🌸 粉红色: {UNIFIED_COLORS['pink']}

📐 学术论文标准:
• 300 DPI 高分辨率
• 白色背景
• 统一字体
• 专业配色"""
    
    ax.text(0.1, 0.9, color_info, transform=ax.transAxes, fontproperties=custom_font,
            fontsize=10, verticalalignment='top', linespacing=1.3,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=UNIFIED_COLORS['light_blue'], alpha=0.3))
    
    plt.tight_layout()
    
    # 保存预览图
    preview_path = save_unified_figure(fig, 'color_scheme_preview.png', '配色方案预览')
    return preview_path

if __name__ == "__main__":
    print("=" * 60)
    print("统一配色方案配置")
    print("=" * 60)
    
    # 生成配色预览
    preview_path = preview_color_schemes()
    
    print("=" * 60)
    print("✓ 统一配色方案配置完成！")
    print(f"✓ 配色预览图: {preview_path}")
    print("✓ 可在其他脚本中导入使用：")
    print("  from unified_color_scheme import *")
    print("=" * 60)
