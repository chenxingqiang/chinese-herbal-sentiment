#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于BERT的情感分析模型
实现论文中提到的BERT预训练模型微调用于情感分析
"""

import os
import glob
import pandas as pd
import numpy as np
import jieba
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
import warnings
warnings.filterwarnings('ignore')

# 导入新的数据集加载器
try:
    from ..utils.dataset_loader import load_chinese_herbal_dataset
except ImportError:
    from chinese_herbal_sentiment.utils.dataset_loader import load_chinese_herbal_dataset

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 用于显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用于正常显示负号

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

class CommentDataset(Dataset):
    """评论数据集类"""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # 使用BERT tokenizer处理文本
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label + 1, dtype=torch.long)  # 将[-1, 0, 1]转换为[0, 1, 2]
        }

class BERTSentimentAnalysis:
    def __init__(self):
        # 加载停用词
        self.stopwords = self.load_stopwords()
        # 初始化jieba分词
        jieba.initialize()
        # 设置模型参数
        self.model_name = 'bert-base-chinese'  # 使用预训练的中文BERT模型
        self.max_length = 128
        self.batch_size = 16
        self.epochs = 3
        self.learning_rate = 2e-5

        # 加载tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)

    def load_stopwords(self):
        """加载停用词表"""
        stopwords = set()
        try:
            # 尝试从文件加载停用词，如果文件不存在则使用默认停用词
            with open('stopwords.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    stopwords.add(line.strip())
        except:
            # 默认停用词
            default_stopwords = ['的', '了', '和', '是', '就', '都', '而', '及', '与', '着',
                               '或', '一个', '没有', '我们', '你们', '他们', '它们', '这个',
                               '那个', '这些', '那些', '这样', '那样', '之', '的话', '什么']
            stopwords = set(default_stopwords)
        return stopwords

    def preprocess(self, text):
        """文本预处理"""
        if not isinstance(text, str):
            return ""
        # 去除特殊字符和数字
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
        # 分词
        words = jieba.lcut(text)
        # 去除停用词
        words = [word for word in words if word not in self.stopwords and len(word.strip()) > 0]
        return ' '.join(words)

    def train_model(self, train_dataloader, val_dataloader, num_labels=3, epochs=None, learning_rate=None):
        """训练BERT模型"""
        # 使用实例变量中的参数，如果没有传入特定值
        epochs_to_use = epochs if epochs is not None else self.epochs
        learning_rate_to_use = learning_rate if learning_rate is not None else self.learning_rate

        # 加载预训练的BERT模型
        model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            output_attentions=False,
            output_hidden_states=False
        )

        # 将模型移至GPU（如果可用）
        model.to(device)

        # 设置优化器
        optimizer = AdamW(model.parameters(), lr=learning_rate_to_use, eps=1e-8)

        # 计算训练总步数
        total_steps = len(train_dataloader) * epochs_to_use

        # 创建学习率调度器
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        # 训练循环
        training_stats = []

        for epoch in range(epochs_to_use):
            print(f'\n======== Epoch {epoch + 1} / {epochs_to_use} ========')
            print('训练中...')

            # 重置总损失
            total_train_loss = 0
            model.train()

            for batch in train_dataloader:
                # 将数据移至GPU（如果可用）
                b_input_ids = batch['input_ids'].to(device)
                b_attention_mask = batch['attention_mask'].to(device)
                b_labels = batch['labels'].to(device)

                # 清除之前的梯度
                model.zero_grad()

                # 前向传播
                outputs = model(
                    b_input_ids,
                    attention_mask=b_attention_mask,
                    labels=b_labels
                )

                loss = outputs.loss
                total_train_loss += loss.item()

                # 反向传播
                loss.backward()

                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # 更新参数
                optimizer.step()

                # 更新学习率
                scheduler.step()

            # 计算平均损失
            avg_train_loss = total_train_loss / len(train_dataloader)
            print(f"平均训练损失: {avg_train_loss}")

            # 验证
            print("验证中...")
            model.eval()

            total_val_loss = 0
            all_preds = []
            all_labels = []

            for batch in val_dataloader:
                b_input_ids = batch['input_ids'].to(device)
                b_attention_mask = batch['attention_mask'].to(device)
                b_labels = batch['labels'].to(device)

                with torch.no_grad():
                    outputs = model(
                        b_input_ids,
                        attention_mask=b_attention_mask,
                        labels=b_labels
                    )

                loss = outputs.loss
                total_val_loss += loss.item()

                # 获取预测结果
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                labels = b_labels.cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels)

            # 计算平均验证损失
            avg_val_loss = total_val_loss / len(val_dataloader)
            print(f"平均验证损失: {avg_val_loss}")

            # 计算验证准确率
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)

            # 将预测结果转换回原始标签 [0, 1, 2] -> [-1, 0, 1]
            all_preds_original = all_preds - 1
            all_labels_original = all_labels - 1

            accuracy = accuracy_score(all_labels_original, all_preds_original)
            precision = precision_score(all_labels_original, all_preds_original, average='weighted')
            recall = recall_score(all_labels_original, all_preds_original, average='weighted')
            f1 = f1_score(all_labels_original, all_preds_original, average='weighted')

            print(f"验证准确率: {accuracy:.4f}")
            print(f"验证精确率: {precision:.4f}")
            print(f"验证召回率: {recall:.4f}")
            print(f"验证F1值: {f1:.4f}")

            # 保存训练统计数据
            training_stats.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_accuracy': accuracy,
                'val_precision': precision,
                'val_recall': recall,
                'val_f1': f1
            })

        print("\n训练完成!")

        # 保存模型
        model_save_path = 'bert_sentiment_model'
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        print(f"保存模型到 {model_save_path}")
        model.save_pretrained(model_save_path)
        self.tokenizer.save_pretrained(model_save_path)

        return model, training_stats

    def evaluate_model(self, model, test_dataloader):
        """评估模型性能"""
        print("评估模型...")
        model.eval()

        all_preds = []
        all_labels = []

        for batch in test_dataloader:
            b_input_ids = batch['input_ids'].to(device)
            b_attention_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)

            with torch.no_grad():
                outputs = model(
                    b_input_ids,
                    attention_mask=b_attention_mask
                )

            # 获取预测结果
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = b_labels.cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

        # 将预测结果转换回原始标签 [0, 1, 2] -> [-1, 0, 1]
        all_preds = np.array(all_preds) - 1
        all_labels = np.array(all_labels) - 1

        # 计算评估指标
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        report = classification_report(all_labels, all_preds)

        print(f"测试准确率: {accuracy:.4f}")
        print(f"测试精确率: {precision:.4f}")
        print(f"测试召回率: {recall:.4f}")
        print(f"测试F1值: {f1:.4f}")
        print("\n分类报告:")
        print(report)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'report': report
        }

    def visualize_training_stats(self, training_stats):
        """可视化训练统计数据"""
        # 提取数据
        epochs = [stat['epoch'] for stat in training_stats]
        train_loss = [stat['train_loss'] for stat in training_stats]
        val_loss = [stat['val_loss'] for stat in training_stats]
        val_accuracy = [stat['val_accuracy'] for stat in training_stats]

        # 创建图表
        plt.figure(figsize=(12, 8))

        # 绘制损失曲线
        plt.subplot(2, 1, 1)
        plt.plot(epochs, train_loss, 'b-o', label='训练损失')
        plt.plot(epochs, val_loss, 'r-o', label='验证损失')
        plt.title('训练和验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True)

        # 绘制准确率曲线
        plt.subplot(2, 1, 2)
        plt.plot(epochs, val_accuracy, 'g-o', label='验证准确率')
        plt.title('验证准确率')
        plt.xlabel('Epoch')
        plt.ylabel('准确率')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('bert_training_stats.png')
        plt.close()

    def analyze_comments(self, comments_files, sample_size=None, epochs=None, batch_size=None):
        """分析评论数据"""
        all_comments = []
        all_labels = []

        # 读取所有评论文件
        for file_path in comments_files:
            try:
                df = pd.read_excel(file_path)
                # 提取评论内容
                comments = df['评论内容'].tolist()

                # 根据文件名确定情感标签
                label = 1 if '好评' in file_path else (-1 if '差评' in file_path else 0)
                labels = [label] * len(comments)

                all_comments.extend(comments)
                all_labels.extend(labels)

                print(f"已读取 {file_path}，包含 {len(comments)} 条评论")
            except Exception as e:
                print(f"读取 {file_path} 失败: {e}")

        print(f"共读取 {len(all_comments)} 条评论")

        # 如果指定了样本大小，随机采样
        if sample_size and len(all_comments) > sample_size:
            indices = np.random.choice(len(all_comments), sample_size, replace=False)
            all_comments = [all_comments[i] for i in indices]
            all_labels = [all_labels[i] for i in indices]
            print(f"随机采样 {sample_size} 条评论进行分析")

        # 预处理文本
        preprocessed_comments = [self.preprocess(comment) for comment in all_comments]

        # 划分训练集、验证集和测试集
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            preprocessed_comments, all_labels, test_size=0.2, random_state=42
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.2, random_state=42
        )

        print(f"训练集大小: {len(X_train)}")
        print(f"验证集大小: {len(X_val)}")
        print(f"测试集大小: {len(X_test)}")

        # 使用传入的参数或默认参数
        batch_size_to_use = batch_size if batch_size is not None else self.batch_size

        # 创建数据集
        train_dataset = CommentDataset(X_train, y_train, self.tokenizer, self.max_length)
        val_dataset = CommentDataset(X_val, y_val, self.tokenizer, self.max_length)
        test_dataset = CommentDataset(X_test, y_test, self.tokenizer, self.max_length)

        # 创建数据加载器
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size_to_use,
            shuffle=True
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size_to_use,
            shuffle=False
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size_to_use,
            shuffle=False
        )

        # 训练模型
        print("\n开始训练BERT模型...")
        model, training_stats = self.train_model(train_dataloader, val_dataloader, num_labels=3, epochs=epochs)

        # 可视化训练统计数据
        self.visualize_training_stats(training_stats)

        # 评估模型
        results = self.evaluate_model(model, test_dataloader)

        return results

    def analyze_comments_with_data(self, comments, labels, epochs=None, batch_size=None):
        """分析直接传入的评论数据"""
        print(f"使用直接传入的 {len(comments)} 条评论进行BERT分析")

        # 预处理文本
        preprocessed_comments = [self.preprocess(comment) for comment in comments]

        # 划分训练集、验证集和测试集
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            preprocessed_comments, labels, test_size=0.2, random_state=42
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.2, random_state=42
        )

        print(f"训练集大小: {len(X_train)}")
        print(f"验证集大小: {len(X_val)}")
        print(f"测试集大小: {len(X_test)}")

        # 使用传入的参数或默认参数
        batch_size_to_use = batch_size if batch_size is not None else self.batch_size

        # 创建数据集
        train_dataset = CommentDataset(X_train, y_train, self.tokenizer, self.max_length)
        val_dataset = CommentDataset(X_val, y_val, self.tokenizer, self.max_length)
        test_dataset = CommentDataset(X_test, y_test, self.tokenizer, self.max_length)

        # 创建数据加载器
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size_to_use,
            shuffle=True
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size_to_use,
            shuffle=False
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size_to_use,
            shuffle=False
        )

        # 训练模型
        print("\n开始训练BERT模型...")
        model, training_stats = self.train_model(train_dataloader, val_dataloader, num_labels=3, epochs=epochs)

        # 可视化训练统计数据
        self.visualize_training_stats(training_stats)

        # 评估模型
        results = self.evaluate_model(model, test_dataloader)

        return results
    
    def analyze_comments_from_huggingface(self, 
                                         dataset_name="xingqiang/chinese-herbal-medicine-sentiment",
                                         sample_size=None,
                                         epochs=None,
                                         batch_size=None,
                                         cache_dir=None):
        """从Hugging Face数据集分析评论数据"""
        print(f"从Hugging Face数据集加载BERT训练数据: {dataset_name}")
        
        try:
            # 加载数据集
            loader = load_chinese_herbal_dataset(dataset_name, cache_dir)
            
            # 获取训练数据
            comments, labels = loader.get_data_for_analysis(
                split='train',
                sample_size=sample_size,
                balance_classes=True
            )
            
            print(f"成功加载 {len(comments)} 条评论数据用于BERT训练")
            
            # 打印数据分布信息
            unique_labels, counts = np.unique(labels, return_counts=True)
            print("数据分布:")
            for label, count in zip(unique_labels, counts):
                sentiment_name = "正面" if label == 1 else ("负面" if label == -1 else "中性")
                percentage = count / len(labels) * 100
                print(f"  {sentiment_name}: {count} 条 ({percentage:.1f}%)")
            
            # 使用现有的分析方法
            return self.analyze_comments_with_data(comments, labels, epochs, batch_size)
            
        except Exception as e:
            print(f"❌ 从Hugging Face加载BERT数据失败: {str(e)}")
            print("回退到本地文件分析...")
            # 回退到原有的文件分析方法
            return self.analyze_comments([], sample_size, epochs, batch_size)

def main():
    # 创建BERT情感分析对象
    bert_analyzer = BERTSentimentAnalysis()
    
    print("=== BERT情感分析系统 ===")
    print("选择数据源:")
    print("1. 从Hugging Face数据集加载 (推荐)")
    print("2. 从本地Excel文件加载")
    
    choice = input("请选择 (1/2): ").strip()
    
    if choice == "1":
        print("\n使用Hugging Face数据集进行BERT训练...")
        try:
            # 从Hugging Face分析评论
            results = bert_analyzer.analyze_comments_from_huggingface(
                dataset_name="xingqiang/chinese-herbal-medicine-sentiment",
                sample_size=10000,  # 可根据GPU内存调整
                epochs=3,  # BERT训练轮次
                batch_size=16
            )
            print("\n✓ BERT训练完成！结果已保存到 bert_training_stats.png")
            print(f"✓ 使用了Hugging Face数据集: xingqiang/chinese-herbal-medicine-sentiment")
        except Exception as e:
            print(f"❌ Hugging Face数据集加载失败: {str(e)}")
            print("回退到本地文件分析...")
            choice = "2"  # 回退到本地文件
    
    if choice == "2":
        print("\n使用本地Excel文件进行BERT训练...")
        # 获取评论文件列表
        comments_files = []
        for category in ['好评', '中评', '差评']:
            files = glob.glob(f"comments/*{category}*.xls") + glob.glob(f"comments/*{category}*.xlsx")
            comments_files.extend(files)

        if not comments_files:
            print("❌ 未找到本地数据文件，请确保comments目录下有Excel文件")
            print("建议使用Hugging Face数据集 (选项1)")
            return

        # 分析评论（使用部分数据进行训练，避免内存不足）
        sample_size = 10000  # 可以根据可用内存调整
        results = bert_analyzer.analyze_comments(comments_files, sample_size)
        print("\n✓ BERT训练完成！结果已保存到 bert_training_stats.png")
    
    # 打印结果摘要
    if 'results' in locals() and results:
        print("\n=== BERT训练结果摘要 ===")
        print(f"准确率: {results['accuracy']:.4f}")
        print(f"精确率: {results['precision']:.4f}")
        print(f"召回率: {results['recall']:.4f}")
        print(f"F1值: {results['f1']:.4f}")
    
    print("\n感谢使用BERT情感分析系统！")

if __name__ == "__main__":
    main()