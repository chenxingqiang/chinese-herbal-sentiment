#!/bin/bash

# 激活虚拟环境
source venv/bin/activate

# 安装基础依赖
pip install pandas numpy matplotlib scikit-learn jieba gensim networkx wordcloud

# 安装深度学习依赖
pip install tensorflow
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers

echo "所有依赖安装完成！" 