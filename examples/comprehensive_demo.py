#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
综合功能演示脚本
展示所有新增功能的使用方法
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from chinese_herbal_sentiment import (
    SentimentAnalysis,
    KeywordExtraction,
    SupplyChainRegression,
    PredictionService,
    TimeSeriesAnalyzer
)


def demo_regression_analysis():
    """演示回归分析功能"""
    print("=" * 60)
    print("1. 回归分析演示")
    print("=" * 60)

    # 创建回归分析器
    regressor = SupplyChainRegression(model_type='linear')

    # 生成模拟数据
    print("生成供应链质量评价模拟数据...")
    data = regressor.generate_supply_chain_data(n_samples=500)
    print(f"数据维度: {data.shape}")

    # 准备数据
    feature_columns = [col for col in data.columns
                      if col not in ['service_quality', 'enterprise_size', 'product_type', 'region']]
    categorical_columns = ['enterprise_size', 'product_type', 'region']

    X, y = regressor.prepare_data(
        data=data,
        target_column='service_quality',
        feature_columns=feature_columns + categorical_columns,
        categorical_columns=categorical_columns
    )

    # 训练模型
    print("训练回归模型...")
    results = regressor.train(X, y, test_size=0.2, random_state=42)

    print(f"训练集R²: {results['train_r2']:.4f}")
    print(f"测试集R²: {results['test_r2']:.4f}")
    print(f"测试集MSE: {results['test_mse']:.4f}")

    # 特征重要性
    importance_df = regressor.feature_importance()
    print("\n前5个重要特征:")
    for _, row in importance_df.head(5).iterrows():
        print(f"  - {row['feature']}: {row['coefficient']:.4f}")

    # 预测演示
    print("\n预测演示:")
    X_new = X[:3]
    predictions = regressor.predict(X_new)
    actual_values = y[:3]

    for i, (pred, actual) in enumerate(zip(predictions, actual_values)):
        print(f"  样本{i+1}: 预测={pred:.3f}, 实际={actual:.3f}")

    # 保存模型和结果
    os.makedirs('output', exist_ok=True)
    regressor.save_model('output/demo_regression_model.pkl')
    regressor.visualize_results('output/regression_analysis.png')
    regressor.generate_report('output/regression_report.md')

    print("回归分析完成! 结果已保存到output目录")
    return regressor


def demo_prediction_service():
    """演示预测服务功能"""
    print("\n" + "=" * 60)
    print("2. 预测服务演示")
    print("=" * 60)

    # 创建预测服务
    service = PredictionService()

    # 演示文本
    demo_texts = [
        "这个中药质量很好，效果不错，物流也很快",
        "包装破损严重，产品质量很差，不推荐购买",
        "服务态度一般，产品还可以，价格合理",
        "非常满意的购买体验，产品质量优秀，会再次购买"
    ]

    print("演示文本:")
    for i, text in enumerate(demo_texts, 1):
        print(f"  {i}. {text}")

    # 情感分析
    print("\n情感分析结果:")
    sentiment_results = service.predict_sentiment(demo_texts, methods=['dictionary'])

    for i, (text, pred, score) in enumerate(zip(
        demo_texts,
        sentiment_results['predictions']['dictionary']['labels'],
        sentiment_results['predictions']['dictionary']['scores']
    ), 1):
        sentiment_name = "正面" if pred == 1 else ("负面" if pred == -1 else "中性")
        print(f"  {i}. {sentiment_name} (评分: {score:.3f})")

    # 关键词提取
    print("\n关键词提取结果:")
    keyword_results = service.extract_keywords_batch(demo_texts, methods=['tfidf'])

    if 'tfidf' in keyword_results['keywords']:
        for i, keywords in enumerate(keyword_results['keywords']['tfidf'][:3], 1):
            top_keywords = [word for word, score in keywords[:5]]
            print(f"  文本{i}: {', '.join(top_keywords)}")

    # 综合分析
    print("\n综合分析:")
    comprehensive_results = service.analyze_comprehensive(
        texts=demo_texts[:2],
        include_sentiment=True,
        include_keywords=True
    )

    print(f"  分析了 {comprehensive_results['texts_count']} 条文本")
    print(f"  包含的分析类型: {list(comprehensive_results['results'].keys())}")

    # 模型信息
    model_info = service.get_model_info()
    print(f"\n可用的分析器: {model_info['available_analyzers']}")

    print("预测服务演示完成!")
    return service


def demo_time_series_analysis():
    """演示时间序列分析功能"""
    print("\n" + "=" * 60)
    print("3. 时间序列分析演示")
    print("=" * 60)

    # 创建时间序列分析器
    analyzer = TimeSeriesAnalyzer()

    # 生成示例数据
    print("生成时间序列数据...")
    sample_data = analyzer.generate_sample_data(
        start_date='2023-01-01',
        periods=180,  # 半年数据
        frequency='D'
    )

    print(f"数据维度: {sample_data.shape}")
    print(f"时间范围: {sample_data['date'].min()} 到 {sample_data['date'].max()}")

    # 加载数据
    analyzer.load_data(sample_data, 'date', 'sentiment_score')

    # 趋势分析
    print("\n趋势分析...")
    trend_results = analyzer.trend_analysis(method='linear')
    print(f"  趋势方向: {trend_results['trend_direction']}")
    print(f"  趋势强度: {trend_results['trend_strength']:.4f}")
    if 'slope' in trend_results:
        print(f"  变化率: {trend_results['slope']:.6f} 单位/天")

    # 季节性分析
    print("\n季节性分析...")
    seasonal_results = analyzer.seasonal_analysis()
    if 'seasonal_strength' in seasonal_results:
        print(f"  季节性强度: {seasonal_results['seasonal_strength']:.4f}")
    print(f"  检测周期: {seasonal_results.get('period', 'N/A')}")

    # 预测分析
    print("\n预测分析...")
    forecast_results = analyzer.forecast(periods=14, method='auto')
    analyzer.forecast_results = forecast_results

    print(f"  预测方法: {forecast_results['method']}")
    print(f"  预测期数: {forecast_results['periods']}")
    predictions = forecast_results['predictions']
    print(f"  预测均值: {np.mean(predictions):.4f}")
    print(f"  预测范围: {min(predictions):.4f} - {max(predictions):.4f}")

    # 异常值检测
    print("\n异常值检测...")
    anomalies = analyzer.detect_anomalies(method='iqr')
    print(f"  检测到 {anomalies['anomaly_count']} 个异常值")
    print(f"  异常值比例: {anomalies['anomaly_percentage']:.2f}%")

    if anomalies['anomaly_count'] > 0:
        print("  主要异常值:")
        for i, (date, value) in enumerate(zip(
            anomalies['anomaly_dates'][:3],
            anomalies['anomaly_values'][:3]
        )):
            print(f"    - {date}: {value:.4f}")

    # 可视化和报告
    try:
        analyzer.visualize_analysis(
            include_trend=True,
            include_seasonal=True,
            include_forecast=True,
            include_anomalies=True,
            save_path='output/time_series_analysis.png'
        )

        analyzer.generate_report('output/time_series_report.md')
        print("\n时间序列分析完成! 结果已保存到output目录")
    except Exception as e:
        print(f"\n可视化或报告生成失败: {e}")

    return analyzer


def demo_api_usage():
    """演示API使用（如果可用）"""
    print("\n" + "=" * 60)
    print("4. API服务演示")
    print("=" * 60)

    try:
        from chinese_herbal_sentiment.api import run_server
        print("API模块可用!")
        print("要启动API服务器，请运行:")
        print("  python -m chinese_herbal_sentiment.api.app")
        print("或者:")
        print("  from chinese_herbal_sentiment.api import run_server")
        print("  run_server()")
        print("\nAPI将在 http://localhost:8000 提供服务")
        print("API文档: http://localhost:8000/docs")

        # 展示API端点
        print("\n可用的API端点:")
        endpoints = [
            "GET  /health - 健康检查",
            "POST /api/v1/sentiment/analyze - 情感分析",
            "POST /api/v1/keywords/extract - 关键词提取",
            "POST /api/v1/quality/predict - 质量预测",
            "POST /api/v1/timeseries/analyze - 时间序列分析",
            "POST /api/v1/analyze/comprehensive - 综合分析",
            "GET  /api/v1/models/info - 模型信息",
            "GET  /api/v1/predictions/history - 预测历史"
        ]

        for endpoint in endpoints:
            print(f"  {endpoint}")

    except ImportError:
        print("API模块不可用 (需要安装 fastapi 和 uvicorn)")
        print("安装命令: pip install fastapi uvicorn")


def demo_integration_example():
    """演示集成使用示例"""
    print("\n" + "=" * 60)
    print("5. 集成使用示例")
    print("=" * 60)

    print("创建一个完整的分析流程...")

    # 1. 准备数据
    print("\n1. 准备评论数据...")
    reviews_data = {
        'date': pd.date_range('2023-01-01', periods=30, freq='D'),
        'review': [
            f"第{i}天的中药材评论，质量{'很好' if i % 3 == 0 else '一般' if i % 3 == 1 else '较差'}"
            for i in range(30)
        ],
        'rating': np.random.randint(3, 6, 30)  # 3-5分评分
    }
    df = pd.DataFrame(reviews_data)

    # 2. 情感分析
    print("\n2. 批量情感分析...")
    sentiment_analyzer = SentimentAnalysis()
    sentiment_scores = []

    for review in df['review']:
        score = sentiment_analyzer.dictionary_based_analysis(review)
        sentiment_scores.append(score)

    df['sentiment_score'] = sentiment_scores
    print(f"  平均情感得分: {np.mean(sentiment_scores):.3f}")

    # 3. 关键词提取
    print("\n3. 提取关键词...")
    keyword_extractor = KeywordExtraction()
    all_keywords = keyword_extractor.tfidf_extraction(df['review'].tolist(), top_k=10)

    print("  主要关键词:")
    if all_keywords and len(all_keywords) > 0:
        for item in all_keywords[:5]:
            if isinstance(item, tuple) and len(item) == 2:
                word, score = item
                print(f"    - {word}: {score:.3f}")
            else:
                print(f"    - {item}")

    # 4. 时间序列分析
    print("\n4. 时间序列趋势分析...")
    ts_analyzer = TimeSeriesAnalyzer()

    # 使用评分作为时间序列数据
    ts_data = df[['date', 'rating']].copy()
    ts_analyzer.load_data(ts_data, 'date', 'rating')

    trend_result = ts_analyzer.trend_analysis(method='linear')
    print(f"  评分趋势: {trend_result['trend_direction']}")
    print(f"  趋势强度: {trend_result['trend_strength']:.3f}")

    # 5. 质量预测
    print("\n5. 基于评论预测供应链质量...")

    # 创建特征（基于情感分析结果）
    avg_sentiment = np.mean(sentiment_scores)
    avg_rating = np.mean(df['rating'])

    # 模拟供应链特征
    features = {
        'material_quality': 5 + avg_sentiment * 2,  # 基于情感调整
        'technology': 7.0 + avg_rating - 4,  # 基于评分调整
        'delivery_speed': 7.5,
        'after_sales_service': 6.5 + avg_sentiment,
        'material_consistency': 7.2,
        'production_efficiency': 7.4,
        'quality_standard': 7.6,
        'product_consistency': 7.3,
        'processing_environment': 7.1,
        'packaging': 7.8,
        'order_accuracy': 8.0,
        'inventory_management': 7.4,
        'information_transparency': 6.9
    }

    # 使用预测服务
    prediction_service = PredictionService()

    # 尝试创建和训练回归模型用于预测
    try:
        regressor = SupplyChainRegression()
        sample_data = regressor.generate_supply_chain_data(200)

        feature_cols = [col for col in sample_data.columns
                       if col not in ['service_quality', 'enterprise_size', 'product_type', 'region']]

        X, y = regressor.prepare_data(sample_data, 'service_quality', feature_cols)
        regressor.train(X, y)

        # 预测质量评分
        feature_vector = np.array([[features[col] if col in features else 7.5
                                  for col in feature_cols]])
        predicted_quality = regressor.predict(feature_vector)[0]

        print(f"  预测供应链质量评分: {predicted_quality:.2f}/10")

    except Exception as e:
        print(f"  质量预测失败: {e}")

    # 6. 生成综合报告
    print("\n6. 生成综合分析报告...")

    report = f"""
    # 中药材评论综合分析报告

    ## 数据概况
    - 分析期间: {df['date'].min().strftime('%Y-%m-%d')} 至 {df['date'].max().strftime('%Y-%m-%d')}
    - 评论数量: {len(df)} 条
    - 平均评分: {avg_rating:.2f}/5
    - 平均情感得分: {avg_sentiment:.3f}

    ## 关键发现
    - 评分趋势: {trend_result['trend_direction']}
    - 趋势强度: {trend_result['trend_strength']:.3f}

        ## 主要关键词
    """
    
    if all_keywords and len(all_keywords) > 0:
        for item in all_keywords[:5]:
            if isinstance(item, tuple) and len(item) == 2:
                word, score = item
                report += f"    - {word}: {score:.3f}\n"
            else:
                report += f"    - {item}\n"

    # 保存报告
    os.makedirs('output', exist_ok=True)
    with open('output/comprehensive_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report)

    print("综合分析完成! 报告已保存到 output/comprehensive_analysis_report.md")


def main():
    """主演示函数"""
    print("🌿 中药材情感分析系统 - 综合功能演示")
    print("=" * 80)

    try:
        # 1. 回归分析演示
        regressor = demo_regression_analysis()

        # 2. 预测服务演示
        service = demo_prediction_service()

        # 3. 时间序列分析演示
        analyzer = demo_time_series_analysis()

        # 4. API使用演示
        demo_api_usage()

        # 5. 集成使用示例
        demo_integration_example()

        print("\n" + "=" * 80)
        print("🎉 所有演示完成!")
        print("=" * 80)
        print("\n📁 输出文件:")
        output_files = [
            "output/demo_regression_model.pkl - 回归模型",
            "output/regression_analysis.png - 回归分析图表",
            "output/regression_report.md - 回归分析报告",
            "output/time_series_analysis.png - 时间序列图表",
            "output/time_series_report.md - 时间序列报告",
            "output/comprehensive_analysis_report.md - 综合分析报告"
        ]

        for file_desc in output_files:
            print(f"  • {file_desc}")

        print("\n🚀 要启动API服务器，请运行:")
        print("  python examples/comprehensive_demo.py --api")
        print("或者:")
        print("  python -c \"from chinese_herbal_sentiment.api import run_server; run_server()\"")

    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--api":
        # 启动API服务器
        try:
            from chinese_herbal_sentiment.api import run_server
            print("🚀 启动API服务器...")
            run_server(host="127.0.0.1", port=8000, reload=True)
        except ImportError:
            print("❌ API功能不可用。请安装: pip install fastapi uvicorn")
    else:
        # 运行演示
        main()
