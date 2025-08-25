#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于TextCNN的情感分析模型
实现论文中提到的TextCNN情感分析算法
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 用于显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用于正常显示负号

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

class CommentDataset(Dataset):
    """评论数据集类"""
    def __init__(self, texts, labels, vocab, max_length=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # 文本转换为索引序列
        tokens = text.split()
        indices = [self.vocab.get(token, 0) for token in tokens]
        
        # 填充或截断到固定长度
        if len(indices) < self.max_length:
            indices += [0] * (self.max_length - len(indices))
        else:
            indices = indices[:self.max_length]
            
        return {
            'text': torch.tensor(indices, dtype=torch.long),
            'label': torch.tensor(label + 1, dtype=torch.long)  # 将[-1, 0, 1]转换为[0, 1, 2]
        }

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, 
                      out_channels=n_filters, 
                      kernel_size=(fs, embedding_dim)) 
            for fs in filter_sizes
        ])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # text = [batch size, sent len]
        embedded = self.embedding(text)
        # embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.unsqueeze(1)
        # embedded = [batch size, 1, sent len, emb dim]
        
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled_n = [batch size, n_filters]
        
        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = [batch size, n_filters * len(filter_sizes)]
            
        return self.fc(cat)

class TextCNNSentimentAnalysis:
    def __init__(self):
        # 加载停用词
        self.stopwords = self.load_stopwords()
        # 初始化jieba分词
        jieba.initialize()
        # 设置模型参数
        self.max_length = 100
        self.batch_size = 64
        self.epochs = 10
        self.embedding_dim = 100
        self.n_filters = 100
        self.filter_sizes = [3, 4, 5]
        self.dropout = 0.5
        self.vocab_size = 10000  # 词汇表大小
        
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
        text = re.sub(r'[^\u4e00-\u9fa5]', ' ', text)
        # 分词
        words = jieba.lcut(text)
        # 去除停用词
        words = [word for word in words if word not in self.stopwords and len(word.strip()) > 0]
        return ' '.join(words)
    
    def build_vocab(self, texts, max_size=10000):
        """构建词汇表"""
        word_counts = {}
        for text in texts:
            for word in text.split():
                if word in word_counts:
                    word_counts[word] += 1
                else:
                    word_counts[word] = 1
        
        # 按频率排序
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # 构建词汇表（前max_size个词）
        vocab = {'<pad>': 0, '<unk>': 1}
        for i, (word, _) in enumerate(sorted_words[:max_size-2]):
            vocab[word] = i + 2
            
        return vocab
    
    def train_model(self, train_dataloader, val_dataloader, vocab_size, pad_idx=0):
        """训练TextCNN模型"""
        # 初始化模型
        model = TextCNN(
            vocab_size=vocab_size,
            embedding_dim=self.embedding_dim,
            n_filters=self.n_filters,
            filter_sizes=self.filter_sizes,
            output_dim=3,  # 3个类别
            dropout=self.dropout,
            pad_idx=pad_idx
        )
        
        # 将模型移至GPU（如果可用）
        model.to(device)
        
        # 设置优化器和损失函数
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        # 训练循环
        best_valid_loss = float('inf')
        training_stats = []
        
        for epoch in range(self.epochs):
            print(f'\n======== Epoch {epoch + 1} / {self.epochs} ========')
            
            # 训练模式
            model.train()
            epoch_loss = 0
            epoch_acc = 0
            
            for batch in train_dataloader:
                # 获取数据
                text = batch['text'].to(device)
                labels = batch['label'].to(device)
                
                # 前向传播
                optimizer.zero_grad()
                predictions = model(text)
                
                # 计算损失
                loss = criterion(predictions, labels)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                # 计算准确率
                preds = torch.argmax(predictions, dim=1)
                acc = (preds == labels).float().mean()
                
                epoch_loss += loss.item()
                epoch_acc += acc.item()
            
            # 计算平均损失和准确率
            train_loss = epoch_loss / len(train_dataloader)
            train_acc = epoch_acc / len(train_dataloader)
            
            # 验证模式
            model.eval()
            valid_loss = 0
            valid_acc = 0
            
            with torch.no_grad():
                for batch in val_dataloader:
                    # 获取数据
                    text = batch['text'].to(device)
                    labels = batch['label'].to(device)
                    
                    # 前向传播
                    predictions = model(text)
                    
                    # 计算损失
                    loss = criterion(predictions, labels)
                    
                    # 计算准确率
                    preds = torch.argmax(predictions, dim=1)
                    acc = (preds == labels).float().mean()
                    
                    valid_loss += loss.item()
                    valid_acc += acc.item()
            
            # 计算平均验证损失和准确率
            valid_loss = valid_loss / len(val_dataloader)
            valid_acc = valid_acc / len(val_dataloader)
            
            # 保存最佳模型
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), 'textcnn_model.pt')
            
            # 记录训练统计数据
            training_stats.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': valid_loss,
                'val_acc': valid_acc
            })
            
            print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%')
            print(f'Val. Loss: {valid_loss:.4f} | Val. Acc: {valid_acc*100:.2f}%')
        
        return model, training_stats
    
    def evaluate_model(self, model, test_dataloader):
        """评估模型性能"""
        model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_dataloader:
                text = batch['text'].to(device)
                labels = batch['label'].cpu().numpy()
                
                predictions = model(text)
                preds = torch.argmax(predictions, dim=1).cpu().numpy()
                
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
        train_acc = [stat['train_acc'] for stat in training_stats]
        val_acc = [stat['val_acc'] for stat in training_stats]
        
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
        plt.plot(epochs, train_acc, 'b-o', label='训练准确率')
        plt.plot(epochs, val_acc, 'r-o', label='验证准确率')
        plt.title('训练和验证准确率')
        plt.xlabel('Epoch')
        plt.ylabel('准确率')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('textcnn_training_stats.png')
        plt.close()
    
    def analyze_comments(self, comments_files, sample_size=None):
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
        
        return self.analyze_comments_with_data(all_comments, all_labels)
    
    def analyze_comments_with_data(self, comments, labels):
        """分析直接传入的评论数据"""
        print(f"使用直接传入的 {len(comments)} 条评论进行TextCNN分析")
        
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
        
        # 构建词汇表
        vocab = self.build_vocab(X_train, max_size=self.vocab_size)
        print(f"词汇表大小: {len(vocab)}")
        
        # 创建数据集
        train_dataset = CommentDataset(X_train, y_train, vocab, self.max_length)
        val_dataset = CommentDataset(X_val, y_val, vocab, self.max_length)
        test_dataset = CommentDataset(X_test, y_test, vocab, self.max_length)
        
        # 创建数据加载器
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size)
        
        # 训练模型
        print("\n开始训练TextCNN模型...")
        model, training_stats = self.train_model(train_dataloader, val_dataloader, len(vocab))
        
        # 可视化训练统计数据
        self.visualize_training_stats(training_stats)
        
        # 评估模型
        print("\n评估TextCNN模型...")
        results = self.evaluate_model(model, test_dataloader)
        
        return results

def main():
    # 创建TextCNN情感分析对象
    textcnn_analyzer = TextCNNSentimentAnalysis()
    
    # 获取评论文件列表
    comments_files = []
    for category in ['好评', '中评', '差评']:
        files = glob.glob(f"comments/*{category}*.xls") + glob.glob(f"comments/*{category}*.xlsx")
        comments_files.extend(files[:2])  # 每类取前两个文件进行测试
    
    # 分析评论
    results = textcnn_analyzer.analyze_comments(comments_files)
    
    print("\n分析完成！结果已保存到 textcnn_training_stats.png")

if __name__ == "__main__":
    main() 