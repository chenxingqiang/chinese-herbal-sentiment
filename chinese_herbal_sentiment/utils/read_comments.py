#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
import glob

# 读取一个样例文件
sample_file = "comments/100001642346-好评.xls"
if not os.path.exists(sample_file):
    # 如果指定文件不存在，找一个存在的文件
    files = glob.glob("comments/*.xls")
    if files:
        sample_file = files[0]
    else:
        print("找不到评论文件")
        exit(1)

try:
    # 尝试读取Excel文件
    df = pd.read_excel(sample_file)
    print(f"成功读取文件: {sample_file}")
    print(f"数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    print("\n前5行数据:")
    print(df.head())
except Exception as e:
    print(f"读取文件出错: {e}") 