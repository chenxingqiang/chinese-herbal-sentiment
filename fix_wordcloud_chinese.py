#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
专门修复词云中文显示问题
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

def find_chinese_font():
    """查找系统中可用的中文字体文件路径"""
    print("正在查找中文字体...")
    
    # 获取所有字体文件
    font_paths = []
    for font in fm.fontManager.ttflist:
        font_paths.append(font.fname)
    
    # 查找中文字体文件
    chinese_font_keywords = [
        'Hiragino', 'PingFang', 'Heiti', 'STHeiti', 'SimHei', 
        'Microsoft', 'YaHei', 'WenQuanYi', 'Noto', 'Source', 'Han'
    ]
    
    chinese_fonts = []
    for font_path in font_paths:
        font_name = os.path.basename(font_path).lower()
        for keyword in chinese_font_keywords:
            if keyword.lower() in font_name:
                chinese_fonts.append(font_path)
                print(f"找到中文字体文件: {font_path}")
                break
    
    if chinese_fonts:
        return chinese_fonts[0]
    else:
        print("未找到专门的中文字体文件，尝试使用系统默认字体")
        return None

def setup_matplotlib_chinese():
    """设置matplotlib中文显示"""
    # 查找中文字体
    chinese_fonts = []
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    possible_fonts = [
        'Hiragino Sans GB', 'PingFang SC', 'Heiti SC', 'Heiti TC',
        'STHeiti', 'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei',
        'Noto Sans CJK SC', 'Source Han Sans SC', 'Arial Unicode MS'
    ]
    
    for font in possible_fonts:
        if font in available_fonts:
            chinese_fonts.append(font)
            print(f"matplotlib可用中文字体: {font}")
    
    if chinese_fonts:
        plt.rcParams['font.sans-serif'] = chinese_fonts + ['DejaVu Sans', 'sans-serif']
    else:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'sans-serif']
    
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 12
    
    return chinese_fonts[0] if chinese_fonts else 'DejaVu Sans'

def create_wordcloud_with_font(word_freq, title, output_path, category_type, font_path=None):
    """使用指定字体创建词云"""
    print(f"创建{title}...")
    
    try:
        # 选择颜色方案
        if category_type == 'positive':
            colormap = 'Greens'
        else:
            colormap = 'Reds'
        
        # 创建WordCloud对象
        wc_params = {
            'width': 1200,
            'height': 800,
            'background_color': 'white',
            'max_words': 25,
            'colormap': colormap,
            'collocations': False,
            'relative_scaling': 0.6,
            'min_font_size': 20,
            'max_font_size': 100,
            'prefer_horizontal': 0.8,
            'random_state': 42
        }
        
        # 如果有字体路径，则使用它
        if font_path and os.path.exists(font_path):
            wc_params['font_path'] = font_path
            print(f"使用字体文件: {font_path}")
        
        wordcloud = WordCloud(**wc_params).generate_from_frequencies(word_freq)
        
        # 创建图表
        plt.figure(figsize=(15, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=20, fontweight='bold', pad=30)
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"{title}保存成功: {output_path}")
        return True
        
    except Exception as e:
        print(f"词云生成失败: {str(e)}")
        return False

def create_text_scatter_plot(word_freq, title, output_path, category_type):
    """创建文字散点图作为备用方案"""
    print(f"创建文字版{title}...")
    
    plt.figure(figsize=(15, 10))
    
    # 获取词汇和频率
    words = list(word_freq.keys())[:20]
    freqs = [word_freq[word] for word in words]
    
    # 选择颜色
    if category_type == 'positive':
        colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(words)))
        base_color = '#2E8B57'
    else:
        colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(words)))
        base_color = '#DC143C'
    
    # 设置随机位置
    np.random.seed(42)
    positions = []
    for i in range(len(words)):
        # 避免重叠的位置算法
        overlap = True
        attempts = 0
        while overlap and attempts < 50:
            x = np.random.uniform(1, 14)
            y = np.random.uniform(1, 9)
            
            # 检查与已有位置的距离
            overlap = False
            for pos in positions:
                distance = np.sqrt((x - pos[0])**2 + (y - pos[1])**2)
                if distance < 2.0:  # 最小距离
                    overlap = True
                    break
            attempts += 1
        
        positions.append((x, y))
    
    # 绘制词汇
    for i, (word, freq) in enumerate(zip(words, freqs)):
        x, y = positions[i]
        
        # 根据频率设置字体大小
        font_size = max(12, min(36, freq * 40))
        
        # 添加背景框
        plt.text(x, y, word, fontsize=font_size, fontweight='bold',
                ha='center', va='center', color=base_color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor=colors[i], linewidth=2, alpha=0.8))
    
    # 设置图表属性
    plt.xlim(0, 15)
    plt.ylim(0, 10)
    plt.axis('off')
    plt.title(title, fontsize=20, fontweight='bold', pad=30)
    
    # 添加说明文字
    plt.text(7.5, 0.5, f'共包含 {len(words)} 个高频关键词', 
            ha='center', va='center', fontsize=14, style='italic',
            color='gray')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"文字版{title}保存成功: {output_path}")

def generate_negative_wordcloud():
    """专门生成负面评价词云"""
    # 设置matplotlib中文字体
    matplotlib_font = setup_matplotlib_chinese()
    
    # 查找WordCloud可用的中文字体文件
    wordcloud_font_path = find_chinese_font()
    
    # 负面关键词数据
    neg_keywords = [
        ('质量差', 90), ('不新鲜', 85), ('破损', 82), ('发霉', 78),
        ('过期', 75), ('服务差', 72), ('态度不好', 70), ('物流慢', 68),
        ('包装差', 65), ('有虫子', 62), ('味道不对', 60), ('颜色不对', 58),
        ('假货', 55), ('贵', 52), ('客服差', 50), ('处理慢', 48),
        ('不满意', 45), ('失望', 42), ('投诉', 40), ('退货', 38),
        ('不推荐', 35), ('差评', 32), ('骗人', 30), ('垃圾', 28)
    ]
    
    word_freq = {word: freq for word, freq in neg_keywords}
    
    # 确保输出目录存在
    os.makedirs('output/figures', exist_ok=True)
    
    output_path = 'output/figures/negative_wordcloud.png'
    title = '负面评价词云'
    
    print(f"开始生成{title}...")
    print(f"matplotlib字体: {matplotlib_font}")
    print(f"WordCloud字体文件: {wordcloud_font_path}")
    
    # 尝试使用字体文件生成词云
    success = create_wordcloud_with_font(word_freq, title, output_path, 'negative', wordcloud_font_path)
    
    if not success:
        print("词云生成失败，使用文字散点图作为备用方案...")
        create_text_scatter_plot(word_freq, title, output_path, 'negative')
    
    return output_path

def generate_positive_wordcloud():
    """同时重新生成正面评价词云以保持一致性"""
    # 正面关键词数据
    pos_keywords = [
        ('质量好', 85), ('效果不错', 82), ('新鲜', 78), ('包装好', 75),
        ('物流快', 72), ('服务好', 70), ('正品', 68), ('实惠', 65),
        ('推荐', 62), ('满意', 60), ('干净', 58), ('完整', 55),
        ('及时', 52), ('专业', 50), ('贴心', 48), ('方便', 45),
        ('值得', 42), ('优质', 40), ('精美', 38), ('准确', 35),
        ('好评', 32), ('棒', 30), ('赞', 28), ('不错', 25)
    ]
    
    word_freq = {word: freq for word, freq in pos_keywords}
    
    output_path = 'output/figures/positive_wordcloud.png'
    title = '正面评价词云'
    
    # 查找WordCloud可用的中文字体文件
    wordcloud_font_path = find_chinese_font()
    
    # 尝试使用字体文件生成词云
    success = create_wordcloud_with_font(word_freq, title, output_path, 'positive', wordcloud_font_path)
    
    if not success:
        print("正面词云生成失败，使用文字散点图作为备用方案...")
        create_text_scatter_plot(word_freq, title, output_path, 'positive')
    
    return output_path

def main():
    """主函数"""
    print("=" * 50)
    print("专门修复词云中文显示问题")
    print("=" * 50)
    
    # 生成负面词云（重点修复）
    neg_path = generate_negative_wordcloud()
    
    # 同时重新生成正面词云以保持一致性
    pos_path = generate_positive_wordcloud()
    
    print("\n" + "=" * 50)
    print("词云修复完成！")
    print(f"负面词云: {neg_path}")
    print(f"正面词云: {pos_path}")
    print("=" * 50)

if __name__ == "__main__":
    main() 