#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速生成论文图表脚本
运行命令: python generate_figures.py
"""

import sys
import os

# 添加utils目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# 导入绘图模块
try:
    from generate_all_figures_chinese import generate_all_figures
    
    print("🎨 开始生成中药材企业电商供应链服务质量评价论文图表...")
    print("📊 所有图表将使用中文标签和说明")
    print("-" * 60)
    
    # 生成所有图表
    figures = generate_all_figures()
    
    print("-" * 60)
    print("🎉 图表生成完成！")
    print(f"📁 图表保存位置: output/figures/")
    print(f"📈 共生成 {len(figures)} 个图表文件")
    print("✅ 所有图表均为300 DPI高分辨率，适合学术论文使用")
    
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保 utils/generate_all_figures_chinese.py 文件存在")
    
except Exception as e:
    print(f"❌ 生成图表时发生错误: {e}")
    sys.exit(1)
