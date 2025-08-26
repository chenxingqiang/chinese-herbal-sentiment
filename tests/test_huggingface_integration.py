#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试Hugging Face数据集集成
简单测试脚本，验证数据集加载和基本功能
"""

import sys
import os

# 添加项目目录到Python路径
sys.path.insert(0, '.')

def test_basic_dataset_loading():
    """测试基础数据集加载功能"""
    print("🧪 测试1: 基础数据集加载")
    
    try:
        from chinese_herbal_sentiment.utils.dataset_loader import load_chinese_herbal_dataset
        
        # 加载数据集
        loader = load_chinese_herbal_dataset("xingqiang/chinese-herbal-medicine-sentiment")
        
        # 获取小样本数据
        comments, labels = loader.get_data_for_analysis(
            split='train',
            sample_size=10,
            balance_classes=False
        )
        
        print(f"✅ 成功加载 {len(comments)} 条评论")
        print(f"✅ 样本评论: {comments[0][:50]}...")
        print(f"✅ 样本标签: {labels[0]}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据集加载测试失败: {str(e)}")
        return False

def test_sentiment_analysis_integration():
    """测试情感分析集成"""
    print("\n🧪 测试2: 情感分析集成")
    
    try:
        from chinese_herbal_sentiment.core.sentiment_analysis import SentimentAnalysis
        
        analyzer = SentimentAnalysis()
        
        # 测试词典方法（不需要训练）
        test_comments = [
            "这个中药质量很好，效果不错",
            "质量一般，没什么特别的",
            "质量太差了，完全没效果"
        ]
        
        print("✅ 词典情感分析测试:")
        for comment in test_comments:
            score = analyzer.dictionary_based_analysis(comment)
            sentiment = "正面" if score > 0.2 else ("负面" if score < -0.2 else "中性")
            print(f"  评论: {comment}")
            print(f"  得分: {score:.3f} ({sentiment})")
        
        return True
        
    except Exception as e:
        print(f"❌ 情感分析集成测试失败: {str(e)}")
        return False

def test_imports():
    """测试导入功能"""
    print("\n🧪 测试3: 模块导入")
    
    try:
        from chinese_herbal_sentiment import (
            SentimentAnalysis, 
            DatasetLoader, 
            load_chinese_herbal_dataset
        )
        print("✅ 主要模块导入成功")
        
        # 测试可选的深度学习模块
        try:
            from chinese_herbal_sentiment import BERTSentimentAnalysis
            print("✅ BERT模块导入成功")
        except ImportError as e:
            print(f"⚠️  BERT模块导入失败 (可能缺少依赖): {e}")
        
        try:
            from chinese_herbal_sentiment import DeepLearningSentiment
            print("✅ 深度学习模块导入成功")
        except ImportError as e:
            print(f"⚠️  深度学习模块导入失败 (可能缺少依赖): {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模块导入测试失败: {str(e)}")
        return False

def main():
    """运行所有测试"""
    print("🚀 中药材情感分析系统 - Hugging Face集成测试")
    print("=" * 60)
    
    tests = [
        ("模块导入", test_imports),
        ("基础数据集加载", test_basic_dataset_loading),
        ("情感分析集成", test_sentiment_analysis_integration),
    ]
    
    success_count = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                success_count += 1
                print(f"✅ {test_name} 测试通过")
            else:
                print(f"❌ {test_name} 测试失败")
        except Exception as e:
            print(f"❌ {test_name} 测试出错: {str(e)}")
    
    print("\n" + "=" * 60)
    print(f"测试结果: {success_count}/{len(tests)} 个测试通过")
    
    if success_count == len(tests):
        print("🎉 所有测试通过！Hugging Face集成正常工作。")
    elif success_count > 0:
        print("⚠️  部分测试通过。可能需要安装额外依赖。")
        print("💡 提示: pip install datasets transformers torch tensorflow")
    else:
        print("❌ 所有测试失败。请检查安装和配置。")
    
    print("\n📖 使用说明:")
    print("1. 安装依赖: pip install datasets")
    print("2. 运行完整演示: python chinese_herbal_sentiment/examples/huggingface_demo.py")
    print("3. 查看文档: chinese_herbal_sentiment/README_HUGGINGFACE.md")

if __name__ == "__main__":
    main()
