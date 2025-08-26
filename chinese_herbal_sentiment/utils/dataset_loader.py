#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据集加载器
支持从Hugging Face Hub加载中药材情感分析数据集
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

class DatasetLoader:
    """数据集加载器类"""
    
    def __init__(self, dataset_name: str = "xingqiang/chinese-herbal-medicine-sentiment"):
        """
        初始化数据集加载器
        
        Args:
            dataset_name: Hugging Face数据集名称
        """
        self.dataset_name = dataset_name
        self.dataset = None
        self.train_data = None
        self.validation_data = None
        
    def load_from_huggingface(self, cache_dir: Optional[str] = None) -> bool:
        """
        从Hugging Face Hub加载数据集
        
        Args:
            cache_dir: 缓存目录路径
            
        Returns:
            bool: 是否成功加载
        """
        try:
            print(f"正在从Hugging Face Hub加载数据集: {self.dataset_name}")
            
            # 加载数据集
            self.dataset = load_dataset(
                self.dataset_name, 
                cache_dir=cache_dir,
                trust_remote_code=True  # 如果需要的话
            )
            
            print(f"✓ 成功加载数据集")
            print(f"数据集包含以下splits: {list(self.dataset.keys())}")
            
            # 获取训练集和验证集
            if 'train' in self.dataset:
                self.train_data = self.dataset['train']
                print(f"训练集大小: {len(self.train_data)}")
            
            if 'validation' in self.dataset:
                self.validation_data = self.dataset['validation']
                print(f"验证集大小: {len(self.validation_data)}")
            
            # 打印数据集信息
            self._print_dataset_info()
            
            return True
            
        except Exception as e:
            print(f"❌ 加载数据集失败: {str(e)}")
            return False
    
    def _print_dataset_info(self):
        """打印数据集基本信息"""
        if self.train_data is None:
            return
            
        print("\n=== 数据集信息 ===")
        
        # 获取特征信息
        features = self.train_data.features
        print(f"数据特征: {list(features.keys())}")
        
        # 获取样本数据
        sample = self.train_data[0]
        print(f"样本数据结构: {sample.keys()}")
        
        # 统计情感分布
        if 'sentiment_label' in sample:
            labels = self.train_data['sentiment_label']
            unique_labels, counts = np.unique(labels, return_counts=True)
            print(f"情感标签分布:")
            for label, count in zip(unique_labels, counts):
                sentiment_name = self._get_sentiment_name(label)
                percentage = count / len(labels) * 100
                print(f"  {sentiment_name} ({label}): {count:,} ({percentage:.1f}%)")
    
    def _get_sentiment_name(self, label: int) -> str:
        """根据标签获取情感名称"""
        sentiment_map = {
            -1: "负面",
            0: "中性", 
            1: "正面"
        }
        return sentiment_map.get(label, f"未知({label})")
    
    def get_data_for_analysis(self, 
                            split: str = 'train',
                            sample_size: Optional[int] = None,
                            balance_classes: bool = False) -> Tuple[List[str], List[int]]:
        """
        获取用于分析的数据
        
        Args:
            split: 数据分片 ('train' 或 'validation')
            sample_size: 采样大小，None表示使用全部数据
            balance_classes: 是否平衡类别
            
        Returns:
            Tuple[List[str], List[int]]: (评论列表, 标签列表)
        """
        if self.dataset is None:
            raise ValueError("数据集未加载，请先调用 load_from_huggingface()")
        
        if split not in self.dataset:
            raise ValueError(f"数据集中不存在split: {split}")
        
        data = self.dataset[split]
        
        # 提取评论和标签
        comments = data['comment_text'] if 'comment_text' in data.features else data['text']
        labels = data['sentiment_label'] if 'sentiment_label' in data.features else data['label']
        
        # 转换为列表
        comments = list(comments)
        labels = list(labels)
        
        print(f"从 {split} 分片获取数据: {len(comments)} 条评论")
        
        # 类别平衡
        if balance_classes:
            comments, labels = self._balance_classes(comments, labels)
            print(f"类别平衡后数据: {len(comments)} 条评论")
        
        # 采样
        if sample_size and len(comments) > sample_size:
            indices = np.random.choice(len(comments), sample_size, replace=False)
            comments = [comments[i] for i in indices]
            labels = [labels[i] for i in indices]
            print(f"随机采样后数据: {len(comments)} 条评论")
        
        return comments, labels
    
    def _balance_classes(self, comments: List[str], labels: List[int]) -> Tuple[List[str], List[int]]:
        """平衡类别数据"""
        # 统计各类别数量
        unique_labels, counts = np.unique(labels, return_counts=True)
        min_count = min(counts)
        
        print(f"类别平衡: 每个类别保留 {min_count} 条数据")
        
        balanced_comments = []
        balanced_labels = []
        
        for label in unique_labels:
            # 获取该类别的所有索引
            label_indices = [i for i, l in enumerate(labels) if l == label]
            
            # 随机选择min_count个样本
            if len(label_indices) > min_count:
                selected_indices = np.random.choice(label_indices, min_count, replace=False)
            else:
                selected_indices = label_indices
            
            # 添加到平衡数据集
            for idx in selected_indices:
                balanced_comments.append(comments[idx])
                balanced_labels.append(labels[idx])
        
        return balanced_comments, balanced_labels
    
    def get_pandas_dataframe(self, split: str = 'train') -> pd.DataFrame:
        """
        获取pandas DataFrame格式的数据
        
        Args:
            split: 数据分片
            
        Returns:
            pd.DataFrame: 数据框
        """
        if self.dataset is None:
            raise ValueError("数据集未加载，请先调用 load_from_huggingface()")
        
        if split not in self.dataset:
            raise ValueError(f"数据集中不存在split: {split}")
        
        # 转换为pandas DataFrame
        df = self.dataset[split].to_pandas()
        
        print(f"转换 {split} 数据为DataFrame: {len(df)} 行 × {len(df.columns)} 列")
        
        return df
    
    def get_statistics(self) -> dict:
        """获取数据集统计信息"""
        if self.dataset is None:
            return {}
        
        stats = {}
        
        for split_name, split_data in self.dataset.items():
            split_stats = {
                'total_samples': len(split_data),
                'features': list(split_data.features.keys())
            }
            
            # 情感分布统计
            if 'sentiment_label' in split_data.features:
                labels = split_data['sentiment_label']
                unique_labels, counts = np.unique(labels, return_counts=True)
                
                sentiment_dist = {}
                for label, count in zip(unique_labels, counts):
                    sentiment_name = self._get_sentiment_name(label)
                    sentiment_dist[sentiment_name] = {
                        'count': int(count),
                        'percentage': float(count / len(labels) * 100)
                    }
                
                split_stats['sentiment_distribution'] = sentiment_dist
            
            stats[split_name] = split_stats
        
        return stats

def load_chinese_herbal_dataset(dataset_name: str = "xingqiang/chinese-herbal-medicine-sentiment",
                               cache_dir: Optional[str] = None) -> DatasetLoader:
    """
    便捷函数：加载中药材情感分析数据集
    
    Args:
        dataset_name: 数据集名称
        cache_dir: 缓存目录
        
    Returns:
        DatasetLoader: 数据集加载器实例
    """
    loader = DatasetLoader(dataset_name)
    
    if loader.load_from_huggingface(cache_dir):
        return loader
    else:
        raise RuntimeError(f"无法加载数据集: {dataset_name}")

def main():
    """示例用法"""
    print("=== 中药材情感分析数据集加载器示例 ===")
    
    try:
        # 加载数据集
        loader = load_chinese_herbal_dataset()
        
        # 获取统计信息
        stats = loader.get_statistics()
        print(f"\n数据集统计信息: {stats}")
        
        # 获取训练数据
        comments, labels = loader.get_data_for_analysis(
            split='train',
            sample_size=1000,  # 采样1000条数据进行演示
            balance_classes=True
        )
        
        print(f"\n获取的数据:")
        print(f"评论数量: {len(comments)}")
        print(f"标签数量: {len(labels)}")
        print(f"示例评论: {comments[0][:50]}...")
        print(f"示例标签: {labels[0]}")
        
        # 转换为DataFrame
        df = loader.get_pandas_dataframe('train')
        print(f"\nDataFrame信息:")
        print(df.info())
        print(f"\n前5行数据:")
        print(df.head())
        
    except Exception as e:
        print(f"❌ 示例运行失败: {str(e)}")

if __name__ == "__main__":
    main()
