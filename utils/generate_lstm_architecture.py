#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成LSTM模型架构图
基于论文中的LSTM模型架构描述生成详细的架构图
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# 设置中文字体配置
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置自定义字体路径（如果可用）
import matplotlib.font_manager as fm
try:
    font_path = '/Users/xingqiangchen/Library/Fonts/AlibabaPuHuiTi-3-85-Bold.ttf'
    custom_font = fm.FontProperties(fname=font_path)
except:
    custom_font = fm.FontProperties()

# 创建输出目录
os.makedirs('output/figures', exist_ok=True)

def generate_lstm_architecture():
    """生成LSTM模型架构图"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 设置画布
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 定义颜色
    colors = {
        'input': '#E3F2FD',
        'embedding': '#BBDEFB',
        'lstm': '#2196F3',
        'bilstm': '#1976D2',
        'dense': '#FF9800',
        'output': '#4CAF50',
        'arrow': '#666666'
    }
    
    # 1. 输入层
    input_rect = patches.Rectangle((1, 8.5), 2, 0.8, linewidth=2, edgecolor='black', 
                                  facecolor=colors['input'], alpha=0.8)
    ax.add_patch(input_rect)
    ax.text(2, 8.9, '输入序列\n(评论文本)', ha='center', va='center', 
            fontproperties=custom_font, fontsize=10, fontweight='bold')
    ax.text(2, 8.2, '最大长度: 100', ha='center', va='center', 
            fontproperties=custom_font, fontsize=8)
    
    # 2. 词嵌入层
    embed_rect = patches.Rectangle((1, 7), 2, 0.8, linewidth=2, edgecolor='black', 
                                  facecolor=colors['embedding'], alpha=0.8)
    ax.add_patch(embed_rect)
    ax.text(2, 7.4, '词嵌入层\n(Embedding)', ha='center', va='center', 
            fontproperties=custom_font, fontsize=10, fontweight='bold')
    ax.text(2, 6.7, '维度: 100\n词汇表: 10,000', ha='center', va='center', 
            fontproperties=custom_font, fontsize=8)
    
    # 3. 双向LSTM层
    bilstm_rect = patches.Rectangle((0.5, 5.5), 3, 0.8, linewidth=2, edgecolor='black', 
                                   facecolor=colors['bilstm'], alpha=0.8)
    ax.add_patch(bilstm_rect)
    ax.text(2, 5.9, '双向LSTM层\n(Bidirectional LSTM)', ha='center', va='center', 
            fontproperties=custom_font, fontsize=10, fontweight='bold', color='white')
    ax.text(2, 5.2, '隐藏维度: 128\n返回序列: True', ha='center', va='center', 
            fontproperties=custom_font, fontsize=8, color='white')
    
    # 双向LSTM的前向和后向箭头
    ax.annotate('', xy=(3.2, 5.7), xytext=(0.8, 5.7), 
                arrowprops=dict(arrowstyle='->', lw=2, color='white'))
    ax.annotate('', xy=(0.8, 6.1), xytext=(3.2, 6.1), 
                arrowprops=dict(arrowstyle='->', lw=2, color='white'))
    ax.text(0.3, 5.7, '前向', ha='center', va='center', fontproperties=custom_font, fontsize=8)
    ax.text(3.7, 6.1, '后向', ha='center', va='center', fontproperties=custom_font, fontsize=8)
    
    # 4. 单向LSTM层
    lstm_rect = patches.Rectangle((1, 4), 2, 0.8, linewidth=2, edgecolor='black', 
                                 facecolor=colors['lstm'], alpha=0.8)
    ax.add_patch(lstm_rect)
    ax.text(2, 4.4, 'LSTM层', ha='center', va='center', 
            fontproperties=custom_font, fontsize=10, fontweight='bold', color='white')
    ax.text(2, 3.7, '隐藏维度: 64\n返回序列: False', ha='center', va='center', 
            fontproperties=custom_font, fontsize=8, color='white')
    
    # 5. 全连接层
    dense_rect = patches.Rectangle((1, 2.5), 2, 0.8, linewidth=2, edgecolor='black', 
                                  facecolor=colors['dense'], alpha=0.8)
    ax.add_patch(dense_rect)
    ax.text(2, 2.9, '全连接层\n(Dense)', ha='center', va='center', 
            fontproperties=custom_font, fontsize=10, fontweight='bold', color='white')
    ax.text(2, 2.2, '单元数: 3\n激活函数: Softmax', ha='center', va='center', 
            fontproperties=custom_font, fontsize=8, color='white')
    
    # 6. 输出层
    output_rect = patches.Rectangle((1, 1), 2, 0.8, linewidth=2, edgecolor='black', 
                                   facecolor=colors['output'], alpha=0.8)
    ax.add_patch(output_rect)
    ax.text(2, 1.4, '输出\n(情感分类)', ha='center', va='center', 
            fontproperties=custom_font, fontsize=10, fontweight='bold', color='white')
    ax.text(2, 0.7, '正面/中性/负面', ha='center', va='center', 
            fontproperties=custom_font, fontsize=8, color='white')
    
    # 连接箭头
    arrows = [
        ((2, 8.5), (2, 7.8)),  # 输入 -> 嵌入
        ((2, 7), (2, 6.3)),    # 嵌入 -> 双向LSTM
        ((2, 5.5), (2, 4.8)),  # 双向LSTM -> LSTM
        ((2, 4), (2, 3.3)),    # LSTM -> 全连接
        ((2, 2.5), (2, 1.8))   # 全连接 -> 输出
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start, 
                   arrowprops=dict(arrowstyle='->', lw=3, color=colors['arrow']))
    
    # 右侧详细信息
    info_x = 5
    
    # 模型信息框
    info_rect = patches.Rectangle((info_x, 6), 8, 3.5, linewidth=2, edgecolor='black', 
                                 facecolor='#F5F5F5', alpha=0.9)
    ax.add_patch(info_rect)
    
    ax.text(info_x + 4, 9.2, 'LSTM情感分析模型架构详细信息', ha='center', va='center', 
            fontproperties=custom_font, fontsize=14, fontweight='bold')
    
    model_info = """
模型配置参数:
• 词汇表大小: 10,000个词
• 词嵌入维度: 100维
• 双向LSTM隐藏维度: 128维
• 单向LSTM隐藏维度: 64维
• 序列最大长度: 100个词
• 批量大小: 32
• 优化器: Adam
• 损失函数: Categorical Crossentropy
• 评价指标: Accuracy, Precision, Recall, F1

训练数据:
• 训练集: 150,323条评论
• 验证集: 37,581条评论
• 测试集: 46,976条评论
• 数据来源: 淘宝、京东、天猫平台

模型性能:
• 最终测试准确率: 87.61%
• 测试精确率: 87.41%
• 测试召回率: 87.61%
• 测试F1值: 87.50%
"""
    
    ax.text(info_x + 0.2, 8.8, model_info, ha='left', va='top', 
            fontproperties=custom_font, fontsize=9, linespacing=1.3)
    
    # 数据流程说明
    flow_rect = patches.Rectangle((info_x, 2), 8, 3.5, linewidth=2, edgecolor='black', 
                                 facecolor='#E8F5E8', alpha=0.9)
    ax.add_patch(flow_rect)
    
    ax.text(info_x + 4, 5.2, '数据处理流程', ha='center', va='center', 
            fontproperties=custom_font, fontsize=14, fontweight='bold')
    
    flow_info = """
1. 文本预处理:
   • 去除特殊字符和标点符号
   • 分词处理
   • 去除停用词
   • 文本长度标准化

2. 序列编码:
   • 词汇表映射
   • 序列填充(Padding)
   • 截断长序列

3. 模型训练:
   • 10个训练轮次
   • 早停策略(patience=3)
   • 模型检查点保存
   • 学习率调度

4. 性能评估:
   • 混淆矩阵分析
   • 分类报告生成
   • ROC曲线绘制
   • 交叉验证
"""
    
    ax.text(info_x + 0.2, 4.8, flow_info, ha='left', va='top', 
            fontproperties=custom_font, fontsize=9, linespacing=1.3)
    
    # 标题
    ax.text(7, 9.7, 'LSTM情感分析模型架构图', ha='center', va='center', 
            fontproperties=custom_font, fontsize=18, fontweight='bold')
    
    # 保存图片
    output_path = os.path.join('output', 'figures', 'lstm_architecture.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✓ 已生成: LSTM模型架构图 -> {output_path}")
    return output_path

if __name__ == "__main__":
    generate_lstm_architecture()
