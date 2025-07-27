#!/bin/bash

# 确保src目录存在
mkdir -p src

# 确保output目录存在
mkdir -p output

# 安装依赖
pip install -r requirements.txt

# 运行分析脚本
cd src && python main_analysis.py

echo "分析完成！结果已保存到output目录。"

# 列出生成的文件
echo "生成的文件:"
ls -la ../output/

# 自动打开报告（如果在支持的系统上）
if [[ "$OSTYPE" == "darwin"* ]]; then
  # macOS
  open ../output/summary_report.md
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
  # Linux
  if command -v xdg-open &> /dev/null; then
    xdg-open ../output/summary_report.md
  fi
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
  # Windows
  start ../output/summary_report.md
fi 