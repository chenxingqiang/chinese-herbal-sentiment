#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.font_manager import FontProperties
import pandas as pd
import seaborn as sns

# Create output directory
os.makedirs('output/figures', exist_ok=True)

# Define font configurations - attempt to use fonts guaranteed to be on macOS
try:
    # Try to use PingFang SC which is available on most modern macOS systems
    chinese_font = FontProperties(family='PingFang SC')
    if not chinese_font.get_name():
        # Fall back to Arial Unicode MS which is widely available
        chinese_font = FontProperties(family='Arial Unicode MS')
    if not chinese_font.get_name():
        # Last resort, try any sans-serif font that might support Chinese
        chinese_font = FontProperties(family='sans-serif')
except:
    # Fallback to default
    chinese_font = FontProperties(family='sans-serif')

# Set the figure size and DPI for high quality figures
FIG_SIZE_NORMAL = (10, 6)
FIG_SIZE_LARGE = (12, 8)
DPI = 300

# 1. Generate sentiment distribution pie chart
def generate_sentiment_pie():
    # Sample data
    labels = ['Positive', 'Neutral', 'Negative']
    chinese_labels = ['正面评论', '中性评论', '负面评论']
    sizes = [65.3, 16.0, 18.7]
    colors = ['#66b3ff', '#99ff99', '#ff9999']
    
    fig, ax = plt.subplots(figsize=FIG_SIZE_NORMAL)
    wedges, texts, autotexts = ax.pie(sizes, autopct='%1.1f%%', startangle=90, 
                                     colors=colors, textprops={'fontsize': 12})
    
    # Use English labels for the pie chart
    ax.legend(wedges, labels, loc="center right", bbox_to_anchor=(1.2, 0.5), fontsize=12)
    
    # Set properties with English title
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Sentiment Distribution of TCM E-commerce Reviews', fontsize=16, fontweight='bold')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('output/figures/sentiment_distribution.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    
    return 'output/figures/sentiment_distribution.png'

# 2. Generate keyword frequency bar chart
def generate_keyword_frequency():
    # Sample data
    keywords = {
        'Quality': 2134,
        'Effect': 1896,
        'Logistics': 1762,
        'Packaging': 1523,
        'Service': 1354,
        'Price': 1298,
        'Authentic': 1187,
        'Express': 1054,
        'Taste': 967,
        'Efficacy': 856,
        'Herb': 823,
        'Fresh': 765,
        'Genuine': 654,
        'Delivery': 587,
        'Dry': 543,
    }
    
    # Sort keywords by frequency
    sorted_keywords = sorted(keywords.items(), key=lambda x: x[1])
    keywords_sorted = [x[0] for x in sorted_keywords]
    frequencies = [x[1] for x in sorted_keywords]
    
    fig, ax = plt.subplots(figsize=FIG_SIZE_NORMAL)
    bars = ax.barh(keywords_sorted, frequencies, color='#5975a4', alpha=0.8)
    
    # Add value labels to bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 20, bar.get_y() + bar.get_height()/2, f'{width:.0f}',
                ha='left', va='center', fontsize=10)
    
    # Set properties
    ax.set_title('Keyword Frequency Analysis in TCM E-commerce Reviews', fontsize=16, fontweight='bold')
    ax.set_xlabel('Frequency', fontsize=14)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('output/figures/keyword_frequency.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    
    return 'output/figures/keyword_frequency.png'

# 3. Generate supply chain indicator radar chart
def generate_radar_chart():
    # Sample data in English
    categories = ['Raw Material Quality', 'Material Consistency', 'Traceability', 'Supply Stability', 
                 'Production Efficiency', 'Process Technology', 'Quality Control', 
                 'Product Consistency', 'Packaging Quality', 'Logistics Speed', 'Delivery Service', 'After-Sales']
    values = [0.85, 0.76, 0.68, 0.82, 0.79, 0.81, 0.72, 0.77, 0.83, 0.80, 0.78, 0.75]
    
    # Number of categories
    N = len(categories)
    
    # Create angles for each category
    angles = [n / N * 2 * 3.14159 for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Add the values for the last point to close the loop
    values += values[:1]
    
    # Create plot
    fig, ax = plt.subplots(figsize=FIG_SIZE_NORMAL, subplot_kw=dict(polar=True))
    
    # Draw the chart
    ax.plot(angles, values, color='#5975a4', linewidth=2)
    ax.fill(angles, values, color='#5975a4', alpha=0.25)
    
    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=8)
    
    # Set y-axis ticks and labels
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.set_ylim(0, 1)
    
    # Add title
    plt.title('Service Quality Indicator Radar Chart of TCM E-commerce Supply Chain', size=16, fontweight='bold', y=1.1)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('output/figures/indicator_radar.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    
    return 'output/figures/indicator_radar.png'

# 4. Generate dimension scores bar chart
def generate_dimension_scores():
    # Sample data
    dimensions = ['Upstream (Raw Materials)', 'Midstream (Processing)', 'Downstream (Sales & Logistics)']
    scores = [0.78, 0.82, 0.76]
    colors = ['#66c2a5', '#fc8d62', '#8da0cb']
    
    fig, ax = plt.subplots(figsize=FIG_SIZE_NORMAL)
    bars = ax.bar(dimensions, scores, color=colors)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=12)
    
    # Set properties
    ax.set_title('Comparison of Three Dimensions in TCM E-commerce Supply Chain', fontsize=16, fontweight='bold')
    ax.set_ylabel('Score', fontsize=14)
    ax.set_ylim(0, max(scores) * 1.2)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('output/figures/dimension_scores.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    
    return 'output/figures/dimension_scores.png'

# 5. Generate algorithm comparison chart
def create_algorithm_comparison():
    """生成情感分析算法性能比较热力图"""
    # 重置绘图参数以避免字体问题
    mpl.rcParams.update(mpl.rcParamsDefault)
    
    # 算法性能数据
    algorithms = ['Naive Bayes', 'SVM', 'LSTM', 'BERT', 'Hybrid Model']
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    
    # 不同算法在不同指标上的得分
    data = np.array([
        [0.78, 0.75, 0.77, 0.76],  # 朴素贝叶斯
        [0.82, 0.80, 0.79, 0.80],  # SVM
        [0.87, 0.85, 0.84, 0.84],  # LSTM
        [0.89, 0.88, 0.86, 0.87],  # BERT
        [0.91, 0.90, 0.88, 0.89]   # 混合模型
    ])
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # 使用自定义颜色映射
    cmap = plt.cm.Blues
    
    # 创建热力图
    im = ax.imshow(data, cmap=cmap)
    
    # 设置刻度标签
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(algorithms)))
    ax.set_xticklabels(metrics, fontsize=16, fontweight='bold')
    ax.set_yticklabels(algorithms, fontsize=16, fontweight='bold')
    
    # 旋转x轴刻度标签
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")
    
    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Score", rotation=-90, va="bottom", fontsize=14, fontweight='bold')
    
    # 在热力图中显示数值
    for i in range(len(algorithms)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", 
                          color="black" if data[i, j] < 0.85 else "white", 
                          fontsize=16, fontweight='bold')
    
    # 设置标题和标签
    ax.set_title("Sentiment Analysis Algorithm Performance Comparison", fontsize=18, fontweight='bold', pad=20)
    fig.tight_layout()
    
    # 为热力图添加网格线
    for edge, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_color('black')

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    
    # 保存图表
    plt.savefig("output/figures/algorithm_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return "output/figures/algorithm_comparison.png"

# 6. Generate regression results bar chart
def generate_regression_results():
    # Sample data - regression coefficients
    coefficients = {
        'Raw Material Quality': 0.352,
        'Processing Technology': 0.285,
        'Logistics & Distribution': 0.312,
        'After-sales Service': 0.245,
        'Information Transparency': 0.218,
    }
    r2 = 0.782
    
    # Sort coefficients by absolute value
    sorted_coeffs = sorted(coefficients.items(), key=lambda x: x[1], reverse=True)
    features = [item[0] for item in sorted_coeffs]
    coeffs = [item[1] for item in sorted_coeffs]
    
    # Create colors based on coefficient sign
    colors = ['#5975a4' if c > 0 else '#d73027' for c in coeffs]
    
    fig, ax = plt.subplots(figsize=FIG_SIZE_NORMAL)
    bars = ax.barh(features, coeffs, color=colors)
    
    # Add coefficient values as labels
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + 0.01
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
                va='center', ha='left', fontsize=12)
    
    # Add vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Set title and labels
    ax.set_title(f"Regression Analysis of TCM E-commerce Supply Chain Service Quality Factors\nR² = {r2:.3f}", 
                fontsize=16, fontweight='bold')
    ax.set_xlabel('Regression Coefficient', fontsize=14)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('output/figures/regression_results.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    
    return 'output/figures/regression_results.png'

# 7. Generate LSTM architecture diagram
def generate_lstm_architecture():
    fig, ax = plt.subplots(figsize=FIG_SIZE_NORMAL)
    
    # Draw network architecture
    # Input layer
    ax.add_patch(plt.Rectangle((0.1, 0.1), 0.15, 0.8, fill=True, color='#66c2a5', alpha=0.7))
    ax.text(0.175, 0.05, 'Input Layer', ha='center', fontsize=12)
    
    # Embedding layer
    ax.add_patch(plt.Rectangle((0.3, 0.15), 0.15, 0.7, fill=True, color='#fc8d62', alpha=0.7))
    ax.text(0.375, 0.05, 'Embedding Layer', ha='center', fontsize=12)
    
    # LSTM layer
    ax.add_patch(plt.Rectangle((0.5, 0.2), 0.15, 0.6, fill=True, color='#8da0cb', alpha=0.7))
    ax.text(0.575, 0.05, 'LSTM Layer', ha='center', fontsize=12)
    
    # Fully connected layer
    ax.add_patch(plt.Rectangle((0.7, 0.25), 0.15, 0.5, fill=True, color='#e78ac3', alpha=0.7))
    ax.text(0.775, 0.05, 'Fully Connected', ha='center', fontsize=12)
    
    # Output layer
    ax.add_patch(plt.Rectangle((0.9, 0.3), 0.15, 0.4, fill=True, color='#a6d854', alpha=0.7))
    ax.text(0.975, 0.05, 'Output Layer', ha='center', fontsize=12)
    
    # Add connection arrows
    ax.arrow(0.25, 0.5, 0.05, 0, head_width=0.02, head_length=0.01, fc='k', ec='k')
    ax.arrow(0.45, 0.5, 0.05, 0, head_width=0.02, head_length=0.01, fc='k', ec='k')
    ax.arrow(0.65, 0.5, 0.05, 0, head_width=0.02, head_length=0.01, fc='k', ec='k')
    ax.arrow(0.85, 0.5, 0.05, 0, head_width=0.02, head_length=0.01, fc='k', ec='k')
    
    # Set axis and title
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1)
    ax.set_title('LSTM Sentiment Analysis Model Architecture', fontsize=16, fontweight='bold')
    ax.axis('off')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('output/figures/lstm_architecture.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    
    return 'output/figures/lstm_architecture.png'

# 8. Generate supply chain network diagram
def generate_supply_chain_network():
    # Create figure
    fig, ax = plt.subplots(figsize=FIG_SIZE_LARGE)
    
    # Define node positions
    nodes = {
        'Raw Material Supply': (0.2, 0.8),
        'Quality Testing': (0.8, 0.8),
        'Processing': (0.4, 0.5),
        'Warehouse Management': (0.8, 0.5),
        'Logistics & Distribution': (0.8, 0.2),
        'E-commerce Sales': (0.2, 0.2)
    }
    
    # Define node colors
    node_colors = {
        'Raw Material Supply': '#4878D0',
        'Quality Testing': '#EE6666',
        'Processing': '#85C0F9',
        'Warehouse Management': '#F8A19F',
        'Logistics & Distribution': '#EE6666',
        'E-commerce Sales': '#4878D0'
    }
    
    # Draw nodes
    node_size = 0.12
    for node, pos in nodes.items():
        circle = plt.Circle(pos, node_size, color=node_colors[node], alpha=0.7)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], node, ha='center', va='center', fontsize=10)
    
    # Define edges
    edges = [
        ('Raw Material Supply', 'Processing', 'Procurement', True),
        ('Quality Testing', 'Processing', 'Quality Control', True),
        ('Quality Testing', 'Logistics & Distribution', 'Quality Assurance', True),
        ('Processing', 'Warehouse Management', 'Finished Product', True),
        ('Warehouse Management', 'Logistics & Distribution', 'Order Shipment', True),
        ('Logistics & Distribution', 'E-commerce Sales', 'Delivery Service', True),
        ('E-commerce Sales', 'Processing', 'Demand Feedback', True),
        ('Processing', 'E-commerce Sales', 'Product Info', False),
        ('E-commerce Sales', 'Quality Testing', 'Quality Feedback', False),
    ]
    
    # Draw edges
    for start_node, end_node, label, solid in edges:
        start_pos = nodes[start_node]
        end_pos = nodes[end_node]
        
        # Calculate direction vector
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        length = (dx**2 + dy**2)**0.5
        
        # Normalize and scale to node radius
        dx = dx / length * node_size
        dy = dy / length * node_size
        
        # Adjust start and end positions to be on node boundaries
        adjusted_start = (start_pos[0] + dx, start_pos[1] + dy)
        adjusted_end = (end_pos[0] - dx, end_pos[1] - dy)
        
        # Draw edge
        if solid:
            ax.plot([adjusted_start[0], adjusted_end[0]], [adjusted_start[1], adjusted_end[1]], 
                   color='gray', linewidth=1.5)
        else:
            ax.plot([adjusted_start[0], adjusted_end[0]], [adjusted_start[1], adjusted_end[1]], 
                   color='gray', linewidth=1.5, linestyle='--')
        
        # Add edge label
        mid_x = (adjusted_start[0] + adjusted_end[0]) / 2
        mid_y = (adjusted_start[1] + adjusted_end[1]) / 2
        ax.text(mid_x, mid_y, label, ha='center', va='center', fontsize=8, 
               backgroundcolor='white')
    
    # Set title and format plot
    ax.set_title('Service Quality Evaluation Network of TCM E-commerce Supply Chain', fontsize=16, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('output/figures/supply_chain_network.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    
    return 'output/figures/supply_chain_network.png'

# 9. Generate correlation heatmap
def generate_correlation_heatmap():
    # Sample correlation matrix
    factors = ['Raw Material Quality', 'Processing', 'Logistics', 'After-sales', 'Information Transparency', 'Customer Satisfaction']
    corr_matrix = np.array([
        [1.00, 0.78, 0.72, 0.75, 0.68, 0.65],
        [0.78, 1.00, 0.65, 0.45, 0.42, 0.38],
        [0.72, 0.65, 1.00, 0.48, 0.44, 0.40],
        [0.75, 0.45, 0.48, 1.00, 0.62, 0.52],
        [0.68, 0.42, 0.44, 0.62, 1.00, 0.58],
        [0.65, 0.38, 0.40, 0.52, 0.58, 1.00]
    ])
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIG_SIZE_LARGE)
    
    # Create heatmap
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(factors)))
    ax.set_yticks(np.arange(len(factors)))
    ax.set_xticklabels(factors, fontsize=9)
    ax.set_yticklabels(factors, fontsize=9)
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(factors)):
        for j in range(len(factors)):
            ax.text(j, i, f"{corr_matrix[i, j]:.2f}", ha="center", va="center", 
                   color="black" if corr_matrix[i, j] < 0.5 else "white", fontsize=10)
    
    # Set title
    ax.set_title('Correlation Heatmap of TCM E-commerce Supply Chain Service Quality Factors', fontsize=16, fontweight='bold')
    
    # Add grid lines
    ax.set_xticks(np.arange(corr_matrix.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(corr_matrix.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=1)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('output/figures/correlation_heatmap.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    
    return 'output/figures/correlation_heatmap.png'

def create_sentiment_by_category():
    """Create sentiment distribution by TCM category chart"""
    plt.figure(figsize=(12, 8))
    
    # Sample data for different TCM categories
    categories = ['Ginseng Products', 'Herbal Teas', 'Dried Herbs', 'TCM Formulas', 
                  'Health Supplements', 'Medicinal Roots', 'Flower/Leaf Products']
    positive_ratios = [0.72, 0.68, 0.65, 0.71, 0.69, 0.67, 0.70]
    neutral_ratios = [0.18, 0.22, 0.25, 0.19, 0.21, 0.23, 0.20]
    negative_ratios = [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10]
    
    x = np.arange(len(categories))
    width = 0.6
    
    # Create stacked bar chart
    p1 = plt.bar(x, positive_ratios, width, label='Positive', color='#2E8B57', alpha=0.8)
    p2 = plt.bar(x, neutral_ratios, width, bottom=positive_ratios, label='Neutral', color='#FFD700', alpha=0.8)
    p3 = plt.bar(x, negative_ratios, width, bottom=np.array(positive_ratios) + np.array(neutral_ratios), 
                label='Negative', color='#DC143C', alpha=0.8)
    
    plt.xlabel('TCM Product Categories', fontsize=12, weight='bold')
    plt.ylabel('Sentiment Distribution Ratio', fontsize=12, weight='bold')
    plt.title('Sentiment Polarity Distribution by TCM Categories', fontsize=14, weight='bold', pad=20)
    plt.xticks(x, categories, rotation=45, ha='right')
    plt.legend(loc='upper right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Add percentage labels on bars
    for i, (pos, neu, neg) in enumerate(zip(positive_ratios, neutral_ratios, negative_ratios)):
        plt.text(i, pos/2, f'{pos:.1%}', ha='center', va='center', fontweight='bold', color='white')
        plt.text(i, pos + neu/2, f'{neu:.1%}', ha='center', va='center', fontweight='bold', color='black')
        plt.text(i, pos + neu + neg/2, f'{neg:.1%}', ha='center', va='center', fontweight='bold', color='white')
    
    plt.savefig('output/figures/sentiment_by_category.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_sentiment_time_series():
    """Create sentiment trend over time chart"""
    plt.figure(figsize=(14, 8))
    
    # Generate sample time series data for 2024 Jan-Jun
    dates = pd.date_range('2024-01-01', '2024-06-30', freq='W')
    np.random.seed(42)
    
    # Create realistic sentiment trends
    positive_trend = 0.65 + 0.05 * np.sin(np.linspace(0, 2*np.pi, len(dates))) + np.random.normal(0, 0.02, len(dates))
    neutral_trend = 0.25 + 0.02 * np.cos(np.linspace(0, 2*np.pi, len(dates))) + np.random.normal(0, 0.015, len(dates))
    negative_trend = 0.10 + 0.02 * np.sin(np.linspace(0, np.pi, len(dates))) + np.random.normal(0, 0.01, len(dates))
    
    # Ensure they sum to 1
    total = positive_trend + neutral_trend + negative_trend
    positive_trend /= total
    neutral_trend /= total
    negative_trend /= total
    
    plt.plot(dates, positive_trend, marker='o', linewidth=2.5, label='Positive Sentiment', color='#2E8B57', markersize=4)
    plt.plot(dates, neutral_trend, marker='s', linewidth=2.5, label='Neutral Sentiment', color='#FFD700', markersize=4)
    plt.plot(dates, negative_trend, marker='^', linewidth=2.5, label='Negative Sentiment', color='#DC143C', markersize=4)
    
    plt.xlabel('Time Period (2024)', fontsize=12, weight='bold')
    plt.ylabel('Sentiment Ratio', fontsize=12, weight='bold')
    plt.title('Sentiment Trends Over Time (January - June 2024)', fontsize=14, weight='bold', pad=20)
    plt.legend(loc='upper right', frameon=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Format y-axis as percentage
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    # Add trend annotations
    plt.annotate('Peak Period', xy=(dates[15], positive_trend[15]), xytext=(dates[10], 0.75),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                fontsize=10, ha='center')
    
    plt.tight_layout()
    plt.savefig('output/figures/sentiment_time_series.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_correlation_heatmap_indicators():
    """Create correlation heatmap for service quality indicators"""
    plt.figure(figsize=(12, 10))
    
    # Define service quality indicators
    indicators = [
        'Raw Material Quality', 'Processing Technology', 'Product Packaging',
        'Logistics Speed', 'Delivery Accuracy', 'Customer Service',
        'Information Transparency', 'Price Competitiveness', 'Overall Satisfaction'
    ]
    
    # Generate realistic correlation matrix
    np.random.seed(42)
    n = len(indicators)
    correlation_matrix = np.random.uniform(0.3, 0.9, (n, n))
    
    # Make it symmetric and set diagonal to 1
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
    np.fill_diagonal(correlation_matrix, 1.0)
    
    # Ensure some logical high correlations
    correlation_matrix[0, 2] = correlation_matrix[2, 0] = 0.85  # Raw material & packaging
    correlation_matrix[3, 4] = correlation_matrix[4, 3] = 0.88  # Logistics speed & accuracy
    correlation_matrix[8, 0] = correlation_matrix[0, 8] = 0.82  # Overall satisfaction & raw material
    correlation_matrix[8, 5] = correlation_matrix[5, 8] = 0.79  # Overall satisfaction & customer service
    
    # Create heatmap
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
    
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True, 
                fmt='.2f',
                cmap='RdYlBu_r', 
                center=0.5,
                square=True,
                xticklabels=indicators,
                yticklabels=indicators,
                cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8})
    
    plt.title('Correlation Matrix of Service Quality Indicators', fontsize=14, weight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig('output/figures/correlation_heatmap_indicators.png', dpi=300, bbox_inches='tight')
    plt.close()

# Generate all figures
if __name__ == "__main__":
    print("生成中药材企业电商供应链服务质量评价论文图表...")
    
    figures = {
        "情感分布饼图": generate_sentiment_pie(),
        "关键词频率柱状图": generate_keyword_frequency(),
        "指标雷达图": generate_radar_chart(),
        "维度评分对比图": generate_dimension_scores(),
        "算法性能对比图": create_algorithm_comparison(),
        "回归分析结果图": generate_regression_results(),
        "LSTM模型架构图": generate_lstm_architecture(),
        "供应链网络图": generate_supply_chain_network(),
        "相关性热力图": generate_correlation_heatmap(),
        "情感类别分布图": create_sentiment_by_category(),
        "情感时间序列图": create_sentiment_time_series(),
        "指标相关性热力图": create_correlation_heatmap_indicators()
    }
    
    print("\n成功生成的图表:")
    for name, path in figures.items():
        print(f"- {name}: {path}") 