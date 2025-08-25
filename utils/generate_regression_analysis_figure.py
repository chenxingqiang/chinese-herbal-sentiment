import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Set font configuration
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Alibaba PuHuiTi']
plt.rcParams['axes.unicode_minus'] = False  # Correctly display minus sign

# Set the custom font path
import matplotlib.font_manager as fm
font_path = '/Users/xingqiangchen/Library/Fonts/AlibabaPuHuiTi-3-85-Bold.ttf'
custom_font = fm.FontProperties(fname=font_path)

# Create output directory if it doesn't exist
os.makedirs('output/figures', exist_ok=True)

# Set the figure size and DPI for high quality figures
FIG_SIZE_LARGE = (12, 8)
DPI = 300

def create_regression_analysis_figure():
    """
    Generate 图5-14 供应链服务质量得分对消费者情感影响的回归分析
    Using ACTUAL DATA from Table 5-9 in the thesis
    """
    plt.figure(figsize=FIG_SIZE_LARGE)
    
    # ACTUAL DATA from Table 5-9 in thesis (分维度服务质量对消费者情感影响的回归结果)
    dimensions = ['上游维度得分', '中游维度得分', '下游维度得分']
    coefficients = [0.342, 0.245, 0.298]  # 回归系数 from Table 5-9
    standard_errors = [0.018, 0.016, 0.017]  # 标准误 from Table 5-9
    t_values = [19.00, 15.31, 17.53]  # t值 from Table 5-9
    standardized_coef = [0.384, 0.285, 0.331]  # 标准化系数 from Table 5-9
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left subplot: Regression coefficients
    colors = ['#4CAF50', '#FFC107', '#FF9800']  # Green, Yellow, Orange for upstream, midstream, downstream
    bars1 = ax1.bar(dimensions, coefficients, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add error bars
    ax1.errorbar(dimensions, coefficients, yerr=standard_errors, fmt='none', 
                color='black', capsize=5, capthick=2, alpha=0.7)
    
    # Add labels and title for left subplot
    ax1.set_ylabel('回归系数', fontproperties=custom_font, fontsize=14)
    ax1.set_title('供应链各维度对消费者情感的影响系数', fontproperties=custom_font, fontsize=14)
    ax1.set_xticklabels(dimensions, fontproperties=custom_font, rotation=15, ha='right')
    ax1.tick_params(axis='y', labelsize=12)
    
    # Add value labels on bars
    for i, (bar, coef, t_val) in enumerate(zip(bars1, coefficients, t_values)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{coef:.3f}\n(t={t_val:.2f})', 
                ha='center', va='bottom', fontproperties=custom_font, fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.set_ylim(0, max(coefficients) * 1.3)
    
    # Right subplot: Standardized coefficients comparison
    bars2 = ax2.barh(dimensions, standardized_coef, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add labels and title for right subplot
    ax2.set_xlabel('标准化回归系数', fontproperties=custom_font, fontsize=14)
    ax2.set_title('标准化系数对比（影响程度排序）', fontproperties=custom_font, fontsize=14)
    ax2.set_yticklabels(dimensions, fontproperties=custom_font)
    ax2.tick_params(axis='x', labelsize=12)
    
    # Add value labels on bars
    for i, (bar, std_coef) in enumerate(zip(bars2, standardized_coef)):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{std_coef:.3f}', 
                ha='left', va='center', fontproperties=custom_font, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax2.grid(axis='x', linestyle='--', alpha=0.7)
    ax2.set_xlim(0, max(standardized_coef) * 1.2)
    
    # Add overall title
    fig.suptitle('供应链服务质量得分对消费者情感影响的回归分析\n(模型R²=0.758, F=1,312.25, p<0.001)', 
                fontproperties=custom_font, fontsize=16, y=0.95)
    
    # Add model information text box
    model_info = (
        "模型信息:\n"
        "• 样本数量: 212,000条评论\n"
        "• 调整R²: 0.755\n"
        "• 所有系数均在p<0.001水平显著\n"
        "• 上游维度影响最大(β=0.342)\n"
        "• 下游维度次之(β=0.298)\n"
        "• 中游维度相对较小(β=0.245)"
    )
    
    fig.text(0.02, 0.02, model_info, fontproperties=custom_font, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
             verticalalignment='bottom')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.2)
    
    # Save the figure
    output_path = os.path.join('output', 'figures', 'regression_results.png')
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Successfully generated 图5-14: {output_path}")
    return output_path

def create_additional_regression_figure():
    """
    Create an additional figure showing the total quality score impact
    Using data from Table 5-8 (总体服务质量对消费者情感影响)
    """
    plt.figure(figsize=(10, 6))
    
    # Data from Table 5-8 - including control variables
    variables = ['总体服务质量得分', '企业规模', '平台类型', '商品价格(ln)', '销量(ln)']
    coefficients = [0.742, 0.203, 0.156, 0.085, 0.032]
    t_values = [49.47, 7.25, 6.24, 7.08, 4.00]
    
    # Create bar chart
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#607D8B']
    bars = plt.bar(variables, coefficients, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add labels and title
    plt.ylabel('回归系数', fontproperties=custom_font, fontsize=14)
    plt.title('总体服务质量及控制变量对消费者情感的影响\n(模型R²=0.742)', fontproperties=custom_font, fontsize=16)
    plt.xticks(rotation=45, ha='right', fontproperties=custom_font)
    plt.yticks(fontproperties=custom_font)
    
    # Add value labels on bars
    for i, (bar, coef, t_val) in enumerate(zip(bars, coefficients, t_values)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{coef:.3f}\n(t={t_val:.2f})', 
                ha='center', va='bottom', fontproperties=custom_font, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join('output', 'figures', 'total_quality_regression.png')
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Successfully generated additional regression figure: {output_path}")
    return output_path

def generate_all_regression_figures():
    """Generate all regression analysis figures"""
    print("Generating regression analysis figures with ACTUAL DATA from thesis...")
    
    figures = {
        "图5-14 供应链服务质量得分对消费者情感影响的回归分析": create_regression_analysis_figure(),
        "总体服务质量回归分析": create_additional_regression_figure()
    }
    
    print(f"Successfully generated {len(figures)} regression figures:")
    for name, path in figures.items():
        print(f"- {name}: {path}")
    
    return figures

if __name__ == "__main__":
    generate_all_regression_figures()
