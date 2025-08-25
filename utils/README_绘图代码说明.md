# 中药材企业电商供应链服务质量评价论文 - 绘图代码说明

## 📊 主要绘图脚本

### 1. 主要脚本文件
- **`generate_all_figures_chinese.py`** - 主要绘图脚本，生成所有论文图表
- **`generate_regression_analysis_figure.py`** - 专门的回归分析图表脚本
- **`generate_figures_with_real_data.py`** - 使用真实数据的图表生成脚本

### 2. 快速生成所有图表
```bash
cd /Users/xingqiangchen/TASK/master-thesis
python utils/generate_all_figures_chinese.py
```

## 📈 生成的图表列表

### 核心图表（论文中引用）
1. **algorithm_comparison.png** - 情感分析算法性能比较（图4-4）
2. **dimension_scores.png** - 供应链三维度评分比较（图5-1）
3. **regression_results.png** - 回归分析结果（图5-14）
4. **sentiment_distribution.png** - 评论情感分布
5. **platform_distribution.png** - 平台分布图
6. **enterprise_distribution.png** - 企业规模分布
7. **product_distribution.png** - 产品类型分布
8. **sentiment_time_series.png** - 情感时间序列图
9. **monthly_distribution.png** - 月度分布趋势
10. **data_overview_dashboard.png** - 数据概览综合仪表板

### 补充图表
- **total_quality_regression.png** - 总体质量回归分析
- **bert_training_stats.png** - BERT训练统计
- **lstm_training_history.png** - LSTM训练历史

### 已清理的图表
项目已清理了18个重复或不必要的图表文件，保留了13个核心图表。

## 🎨 图表设计特点

### 中文支持
- 所有标签、标题、说明均使用中文
- 支持中文字体渲染
- 统一的字体配置

### 数据来源
- 基于论文中的真实数据
- 表4-2：算法性能数据
- 表5-8、5-9：回归分析数据
- 论文第5章：各类分布数据

### 视觉设计
- 统一的颜色方案
- 高分辨率输出（300 DPI）
- 适合学术论文的简洁风格
- 包含数值标签和统计信息

## 🔧 技术参数

### 图片规格
- **分辨率**: 300 DPI
- **格式**: PNG
- **背景**: 白色
- **字体**: 支持中文显示

### 尺寸规范
- **标准图表**: 10×6 英寸
- **大型图表**: 12×8 英寸  
- **仪表板**: 16×12 英寸

### 颜色方案
```python
COLORS = {
    'primary': '#2196F3',    # 主色调（蓝色）
    'secondary': '#4CAF50',  # 次要色（绿色）
    'accent': '#FF9800',     # 强调色（橙色）
    'warning': '#FFC107',    # 警告色（黄色）
    'danger': '#F44336',     # 危险色（红色）
    'success': '#4CAF50',    # 成功色（绿色）
    'info': '#03A9F4',       # 信息色（浅蓝）
    'light': '#E3F2FD',      # 浅色背景
    'dark': '#1976D2'        # 深色
}
```

## 📊 具体图表说明

### 1. 算法性能比较图
- **数据来源**: 论文表4-2
- **算法**: 8种情感分析算法
- **指标**: 准确率、F1分数
- **最佳性能**: 混合模型（准确率0.91）

### 2. 回归分析结果图
- **数据来源**: 论文表5-9
- **变量**: 上游、中游、下游维度得分
- **模型**: R²=0.758, F=1,312.25
- **样本**: 212,000条评论

### 3. 情感分布图
- **正面评价**: 75.8%
- **中性评价**: 11.5%
- **负面评价**: 12.7%
- **总数**: 234,880条评论

### 4. 平台分布图
- **淘宝**: 40.1% (85,000条)
- **京东**: 32.1% (68,000条)
- **天猫**: 19.8% (42,000条)
- **其他**: 8.0% (17,000条)

## 🛠️ 维护说明

### 修改图表数据
1. 编辑 `generate_all_figures_chinese.py`
2. 找到对应的函数（如 `create_algorithm_comparison()`）
3. 修改数据数组
4. 重新运行脚本

### 添加新图表
1. 在 `generate_all_figures_chinese.py` 中创建新函数
2. 在 `generate_all_figures()` 中调用新函数
3. 遵循现有的命名和样式规范

### 字体问题解决
如果中文显示异常：
1. 检查系统是否安装中文字体
2. 修改 `plt.rcParams['font.sans-serif']` 配置
3. 使用 `matplotlib.font_manager.findfont()` 检查可用字体

## 📁 文件组织

```
utils/
├── generate_all_figures_chinese.py      # 主要绘图脚本
├── generate_regression_analysis_figure.py # 回归分析专用脚本
├── generate_figures_with_real_data.py    # 真实数据图表脚本
└── README_绘图代码说明.md               # 本说明文件

output/figures/
├── algorithm_comparison.png             # 算法比较图
├── regression_results.png               # 回归结果图
├── sentiment_distribution.png           # 情感分布图
└── ...                                  # 其他图表文件
```

## 🚀 快速使用指南

1. **生成所有图表**:
   ```bash
   python utils/generate_all_figures_chinese.py
   ```

2. **仅生成回归分析图**:
   ```bash
   python utils/generate_regression_analysis_figure.py
   ```

3. **查看生成的图表**:
   ```bash
   ls -la output/figures/*.png
   ```

4. **检查图表质量**:
   所有图表均为300 DPI高分辨率，适合直接插入论文使用。
