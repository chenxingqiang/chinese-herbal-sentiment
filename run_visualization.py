#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import matplotlib as mpl

# 确保输出目录存在
os.makedirs('output/figures', exist_ok=True)

print("正在生成论文级别的可视化图表...")

try:
    # 导入并执行可视化代码
    from src.visualization import ThesisVisualizer, generate_sample_data
    
    # 生成示例数据
    sample_data = generate_sample_data()
    
    # 创建可视化对象
    visualizer = ThesisVisualizer()
    
    # 生成各种图表
    sentiment_fig = visualizer.sentiment_distribution(sample_data['sentiment_data'])
    keyword_fig = visualizer.keyword_frequency(sample_data['keywords'])
    radar_fig = visualizer.supply_chain_indicator_radar(sample_data['indicators'])
    network_fig = visualizer.supply_chain_network()
    heatmap_fig = visualizer.correlation_heatmap(sample_data['correlation_matrix'])
    regression_fig = visualizer.regression_results(
        sample_data['regression_coefficients'], 
        sample_data['r2_value']
    )
    dimension_fig = visualizer.dimension_scores(sample_data['dimension_scores'])
    lstm_fig = visualizer.draw_lstm_architecture()
    algo_compare_fig = visualizer.algorithm_performance_comparison()
    
    print("✅ 所有图表已成功生成并保存到 output/figures 目录")
    print("生成的图表文件：")
    for file in os.listdir('output/figures'):
        if file.endswith('.png'):
            print(f"- {file}")
            
except ImportError as e:
    print(f"❌ 导入错误: {str(e)}")
    print("请确保已安装所有必要的依赖包：matplotlib, seaborn, numpy, pandas, jieba, wordcloud, networkx等")
    sys.exit(1)
except Exception as e:
    print(f"❌ 生成图表时出错: {str(e)}")
    sys.exit(1) 