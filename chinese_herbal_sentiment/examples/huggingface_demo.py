#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用Hugging Face数据集进行中药材情感分析的演示脚本
展示如何使用上传的xingqiang/chinese-herbal-medicine-sentiment数据集
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from chinese_herbal_sentiment.core.sentiment_analysis import SentimentAnalysis
from chinese_herbal_sentiment.core.bert_sentiment_analysis import BERTSentimentAnalysis
from chinese_herbal_sentiment.core.deep_learning_sentiment import DeepLearningSentiment
from chinese_herbal_sentiment.core.textcnn_sentiment_analysis import TextCNNSentimentAnalysis
from chinese_herbal_sentiment.utils.dataset_loader import load_chinese_herbal_dataset

def demo_dataset_loading():
    """演示数据集加载功能"""
    print("=" * 60)
    print("演示1: 数据集加载与基本信息")
    print("=" * 60)
    
    try:
        # 加载数据集
        loader = load_chinese_herbal_dataset("xingqiang/chinese-herbal-medicine-sentiment")
        
        # 获取统计信息
        stats = loader.get_statistics()
        print("\n数据集统计信息:")
        for split_name, split_stats in stats.items():
            print(f"\n{split_name} 分片:")
            print(f"  总样本数: {split_stats['total_samples']:,}")
            
            if 'sentiment_distribution' in split_stats:
                print(f"  情感分布:")
                for sentiment, info in split_stats['sentiment_distribution'].items():
                    print(f"    {sentiment}: {info['count']:,} 条 ({info['percentage']:.1f}%)")
        
        # 获取样本数据
        print(f"\n样本数据预览:")
        comments, labels = loader.get_data_for_analysis(split='train', sample_size=5)
        for i, (comment, label) in enumerate(zip(comments[:3], labels[:3])):
            sentiment_name = "正面" if label == 1 else ("负面" if label == -1 else "中性")
            print(f"  样本 {i+1} [{sentiment_name}]: {comment[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据集加载演示失败: {str(e)}")
        return False

def demo_traditional_sentiment_analysis():
    """演示传统情感分析方法（词典+机器学习）"""
    print("\n" + "=" * 60)
    print("演示2: 传统情感分析方法")
    print("=" * 60)
    
    try:
        analyzer = SentimentAnalysis()
        
        # 使用小样本进行快速演示
        results = analyzer.analyze_comments_from_huggingface(
            dataset_name="xingqiang/chinese-herbal-medicine-sentiment",
            sample_size=5000,  # 小样本，便于快速演示
            balance_classes=True
        )
        
        print("\n传统方法分析结果:")
        for method, metrics in results.items():
            if isinstance(metrics, dict) and 'accuracy' in metrics:
                print(f"  {method}:")
                print(f"    准确率: {metrics['accuracy']:.4f}")
                print(f"    F1值: {metrics['f1']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 传统情感分析演示失败: {str(e)}")
        return False

def demo_deep_learning_analysis():
    """演示深度学习情感分析方法"""
    print("\n" + "=" * 60)
    print("演示3: 深度学习情感分析方法 (LSTM)")
    print("=" * 60)
    
    try:
        analyzer = DeepLearningSentiment()
        
        # 使用小样本进行快速演示
        results = analyzer.analyze_comments_from_huggingface(
            dataset_name="xingqiang/chinese-herbal-medicine-sentiment",
            sample_size=3000  # 小样本，便于快速演示
        )
        
        print("\nLSTM分析结果:")
        print(f"  准确率: {results['accuracy']:.4f}")
        print(f"  精确率: {results['precision']:.4f}")
        print(f"  召回率: {results['recall']:.4f}")
        print(f"  F1值: {results['f1']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ LSTM情感分析演示失败: {str(e)}")
        return False

def demo_bert_analysis():
    """演示BERT情感分析方法"""
    print("\n" + "=" * 60)
    print("演示4: BERT情感分析方法")
    print("=" * 60)
    print("注意: BERT训练需要较长时间和较大内存，此演示使用小样本")
    
    try:
        analyzer = BERTSentimentAnalysis()
        
        # 使用很小的样本进行演示
        results = analyzer.analyze_comments_from_huggingface(
            dataset_name="xingqiang/chinese-herbal-medicine-sentiment",
            sample_size=1000,  # 很小的样本
            epochs=1,  # 只训练1轮
            batch_size=8  # 小批量大小
        )
        
        print("\nBERT分析结果:")
        print(f"  准确率: {results['accuracy']:.4f}")
        print(f"  精确率: {results['precision']:.4f}")
        print(f"  召回率: {results['recall']:.4f}")
        print(f"  F1值: {results['f1']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ BERT情感分析演示失败: {str(e)}")
        print("提示: BERT需要PyTorch和transformers库，如果未安装，请运行:")
        print("pip install torch transformers")
        return False

def demo_textcnn_analysis():
    """演示TextCNN情感分析方法"""
    print("\n" + "=" * 60)
    print("演示5: TextCNN情感分析方法")
    print("=" * 60)
    
    try:
        from chinese_herbal_sentiment.core.textcnn_sentiment_analysis import TextCNNSentimentAnalysis
        
        analyzer = TextCNNSentimentAnalysis()
        
        # 加载数据集
        loader = load_chinese_herbal_dataset("xingqiang/chinese-herbal-medicine-sentiment")
        comments, labels = loader.get_data_for_analysis(
            split='train',
            sample_size=2000,
            balance_classes=True
        )
        
        # 使用现有的分析方法
        results = analyzer.analyze_comments_with_data(comments, labels)
        
        print("\nTextCNN分析结果:")
        print(f"  准确率: {results['accuracy']:.4f}")
        print(f"  精确率: {results['precision']:.4f}")
        print(f"  召回率: {results['recall']:.4f}")
        print(f"  F1值: {results['f1']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ TextCNN情感分析演示失败: {str(e)}")
        print("提示: TextCNN需要PyTorch库，如果未安装，请运行:")
        print("pip install torch")
        return False

def main():
    """主演示函数"""
    print("🚀 中药材情感分析系统 - Hugging Face数据集演示")
    print("=" * 60)
    print("数据集: xingqiang/chinese-herbal-medicine-sentiment")
    print("包含: 234,880条中药材电商评论，涵盖正面、中性、负面情感")
    print("=" * 60)
    
    demos = [
        ("数据集加载与基本信息", demo_dataset_loading),
        ("传统情感分析方法", demo_traditional_sentiment_analysis),
        ("深度学习方法 (LSTM)", demo_deep_learning_analysis),
        ("BERT方法", demo_bert_analysis),
        ("TextCNN方法", demo_textcnn_analysis)
    ]
    
    success_count = 0
    
    for demo_name, demo_func in demos:
        print(f"\n🔍 开始演示: {demo_name}")
        
        try:
            if demo_func():
                success_count += 1
                print(f"✅ {demo_name} 演示成功")
            else:
                print(f"❌ {demo_name} 演示失败")
        except KeyboardInterrupt:
            print(f"\n⏹️  用户中断了 {demo_name} 演示")
            break
        except Exception as e:
            print(f"❌ {demo_name} 演示出错: {str(e)}")
    
    print("\n" + "=" * 60)
    print("🎯 演示总结")
    print("=" * 60)
    print(f"成功完成: {success_count}/{len(demos)} 个演示")
    
    if success_count == len(demos):
        print("🎉 所有演示都成功完成！您可以开始使用Hugging Face数据集进行中药材情感分析了。")
    else:
        print("⚠️  部分演示失败，可能需要安装相关依赖库。")
    
    print("\n📖 使用说明:")
    print("1. 确保已安装必要的依赖: pip install datasets transformers torch tensorflow")
    print("2. 在代码中导入: from chinese_herbal_sentiment.utils.dataset_loader import load_chinese_herbal_dataset")
    print("3. 加载数据集: loader = load_chinese_herbal_dataset('xingqiang/chinese-herbal-medicine-sentiment')")
    print("4. 获取数据: comments, labels = loader.get_data_for_analysis()")
    print("5. 使用各种分析方法进行情感分析")
    
    print("\n感谢使用中药材情感分析系统！")

if __name__ == "__main__":
    main()
