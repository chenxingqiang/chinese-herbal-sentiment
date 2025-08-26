#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于深度学习的情感分析模型（LSTM）
实现论文中提到的深度学习情感分析算法
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
# Optional TensorFlow imports with error handling
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
    TF_AVAILABLE = True
except ImportError:
    print("Warning: TensorFlow not available. Deep learning features will be disabled.")
    TF_AVAILABLE = False
    tf = None
    Tokenizer = None
    pad_sequences = None
    Sequential = None
    Embedding = LSTM = Dense = Dropout = Bidirectional = None

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

class DeepLearningSentiment:
    def __init__(self):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for DeepLearningSentiment. Please install: pip install tensorflow")
        # 加载停用词
        self.stopwords = self.load_stopwords()
        # 初始化jieba分词
        jieba.initialize()
        # 设置模型参数
        self.max_words = 10000  # 词汇表大小
        self.max_sequence_length = 100  # 序列最大长度
        self.embedding_dim = 100  # 词嵌入维度
        self.epochs = 10  # 训练轮次
        self.batch_size = 32  # 批量大小
        
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
    
    def build_lstm_model(self, vocab_size):
        """构建LSTM模型"""
        model = Sequential()
        # 词嵌入层
        model.add(Embedding(vocab_size, self.embedding_dim, input_length=self.max_sequence_length))
        # 双向LSTM层
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(Dropout(0.2))
        # 第二个LSTM层
        model.add(LSTM(64))
        model.add(Dropout(0.2))
        # 全连接层
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        # 输出层，3个类别（好评、中评、差评）
        model.add(Dense(3, activation='softmax'))
        
        # 编译模型
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=None, batch_size=None):
        """训练模型"""
        # 将标签转换为适合分类的格式
        # 将 [-1, 0, 1] 转换为 [0, 1, 2]
        y_train_converted = np.array([y + 1 for y in y_train])
        y_val_converted = np.array([y + 1 for y in y_val])
        
        # 构建模型
        model = self.build_lstm_model(self.max_words + 1)
        
        # 使用实例变量中的参数，如果没有传入特定值
        epochs_to_use = epochs if epochs is not None else self.epochs
        batch_size_to_use = batch_size if batch_size is not None else self.batch_size
        
        # 训练模型
        history = model.fit(
            X_train, y_train_converted,
            epochs=epochs_to_use,
            batch_size=batch_size_to_use,
            validation_data=(X_val, y_val_converted),
            verbose=1
        )
        
        return model, history
    
    def evaluate_model(self, model, X_test, y_test):
        """评估模型性能"""
        # 将标签转换为适合分类的格式
        y_test_converted = np.array([y + 1 for y in y_test])
        
        # 预测
        y_pred_probs = model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # 将预测结果转换回原始标签 [0, 1, 2] -> [-1, 0, 1]
        y_pred_original = np.array([y - 1 for y in y_pred])
        
        # 计算评估指标
        accuracy = accuracy_score(y_test, y_pred_original)
        precision = precision_score(y_test, y_pred_original, average='weighted')
        recall = recall_score(y_test, y_pred_original, average='weighted')
        f1 = f1_score(y_test, y_pred_original, average='weighted')
        report = classification_report(y_test, y_pred_original)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'report': report
        }
    
    def visualize_training_history(self, history):
        """可视化训练历史"""
        # 绘制训练和验证准确率
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('模型准确率')
        plt.ylabel('准确率')
        plt.xlabel('训练轮次')
        plt.legend(['训练集', '验证集'], loc='lower right')
        
        # 绘制训练和验证损失
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('模型损失')
        plt.ylabel('损失')
        plt.xlabel('训练轮次')
        plt.legend(['训练集', '验证集'], loc='upper right')
        
        plt.tight_layout()
        plt.savefig('lstm_training_history.png')
        plt.close()
    
    def analyze_comments(self, comments_files, max_comments=80000):
        """分析评论数据"""
        # 按类别分组文件
        good_files = [f for f in comments_files if '好评' in f]
        neutral_files = [f for f in comments_files if '中评' in f]
        bad_files = [f for f in comments_files if '差评' in f]
        
        # 计算每类应读取的评论数量（尽量均衡）
        comments_per_category = max_comments // 3
        remaining = max_comments % 3
        
        all_comments = []
        all_labels = []
        label_counts = {'好评': 0, '中评': 0, '差评': 0}
        
        # 读取好评
        target_good = comments_per_category + (1 if remaining > 0 else 0)
        for file_path in good_files:
            if label_counts['好评'] >= target_good:
                break
            try:
                df = pd.read_excel(file_path)
                comments = df['评论内容'].tolist()
                needed = min(len(comments), target_good - label_counts['好评'])
                
                all_comments.extend(comments[:needed])
                all_labels.extend([1] * needed)
                label_counts['好评'] += needed
                
                print(f"已读取 {file_path}，取 {needed}/{len(comments)} 条好评，当前好评总数: {label_counts['好评']}")
            except Exception as e:
                print(f"读取 {file_path} 失败: {e}")
        
        # 读取中评
        target_neutral = comments_per_category + (1 if remaining > 1 else 0)
        for file_path in neutral_files:
            if label_counts['中评'] >= target_neutral:
                break
            try:
                df = pd.read_excel(file_path)
                comments = df['评论内容'].tolist()
                needed = min(len(comments), target_neutral - label_counts['中评'])
                
                all_comments.extend(comments[:needed])
                all_labels.extend([0] * needed)
                label_counts['中评'] += needed
                
                print(f"已读取 {file_path}，取 {needed}/{len(comments)} 条中评，当前中评总数: {label_counts['中评']}")
            except Exception as e:
                print(f"读取 {file_path} 失败: {e}")
        
        # 读取差评
        target_bad = comments_per_category
        for file_path in bad_files:
            if label_counts['差评'] >= target_bad:
                break
            try:
                df = pd.read_excel(file_path)
                comments = df['评论内容'].tolist()
                needed = min(len(comments), target_bad - label_counts['差评'])
                
                all_comments.extend(comments[:needed])
                all_labels.extend([-1] * needed)
                label_counts['差评'] += needed
                
                print(f"已读取 {file_path}，取 {needed}/{len(comments)} 条差评，当前差评总数: {label_counts['差评']}")
            except Exception as e:
                print(f"读取 {file_path} 失败: {e}")
        
        print(f"共读取 {len(all_comments)} 条评论")
        print(f"评论分布: 好评 {label_counts['好评']} 条, 中评 {label_counts['中评']} 条, 差评 {label_counts['差评']} 条")
        
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
        
        # 创建词汇表
        tokenizer = Tokenizer(num_words=self.max_words)
        tokenizer.fit_on_texts(X_train)
        
        # 将文本转换为序列
        X_train_seq = tokenizer.texts_to_sequences(X_train)
        X_val_seq = tokenizer.texts_to_sequences(X_val)
        X_test_seq = tokenizer.texts_to_sequences(X_test)
        
        # 填充序列
        X_train_pad = pad_sequences(X_train_seq, maxlen=self.max_sequence_length)
        X_val_pad = pad_sequences(X_val_seq, maxlen=self.max_sequence_length)
        X_test_pad = pad_sequences(X_test_seq, maxlen=self.max_sequence_length)
        
        print("\n训练LSTM模型...")
        model, history = self.train_model(X_train_pad, y_train, X_val_pad, y_val, epochs=10)
        
        # 评估模型
        results = self.evaluate_model(model, X_test_pad, y_test)
        
        print(f"LSTM准确率: {results['accuracy']:.4f}")
        print(f"LSTM精确率: {results['precision']:.4f}")
        print(f"LSTM召回率: {results['recall']:.4f}")
        print(f"LSTM F1值: {results['f1']:.4f}")
        
        # 可视化训练历史
        self.visualize_training_history(history)
        
        # 保存模型
        model.save('lstm_sentiment_model.h5')
        print("模型已保存到 lstm_sentiment_model.h5")
        
        return results

    def analyze_comments_with_data(self, comments, labels):
        """分析直接传入的评论数据"""
        print(f"使用直接传入的 {len(comments)} 条评论进行LSTM分析")
        
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
        
        # 创建词汇表
        tokenizer = Tokenizer(num_words=self.max_words)
        tokenizer.fit_on_texts(X_train)
        
        # 将文本转换为序列
        X_train_seq = tokenizer.texts_to_sequences(X_train)
        X_val_seq = tokenizer.texts_to_sequences(X_val)
        X_test_seq = tokenizer.texts_to_sequences(X_test)
        
        # 填充序列
        X_train_pad = pad_sequences(X_train_seq, maxlen=self.max_sequence_length)
        X_val_pad = pad_sequences(X_val_seq, maxlen=self.max_sequence_length)
        X_test_pad = pad_sequences(X_test_seq, maxlen=self.max_sequence_length)
        
        print("\n训练LSTM模型...")
        model, history = self.train_model(X_train_pad, y_train, X_val_pad, y_val, epochs=10)
        
        # 评估模型
        results = self.evaluate_model(model, X_test_pad, y_test)
        
        print(f"LSTM准确率: {results['accuracy']:.4f}")
        print(f"LSTM精确率: {results['precision']:.4f}")
        print(f"LSTM召回率: {results['recall']:.4f}")
        print(f"LSTM F1值: {results['f1']:.4f}")
        
        # 可视化训练历史
        self.visualize_training_history(history)
        
        return results
    
    def analyze_comments_from_huggingface(self, 
                                         dataset_name="xingqiang/chinese-herbal-medicine-sentiment",
                                         sample_size=None,
                                         cache_dir=None):
        """从Hugging Face数据集分析评论数据"""
        print(f"从Hugging Face数据集加载LSTM训练数据: {dataset_name}")
        
        try:
            # 加载数据集
            loader = load_chinese_herbal_dataset(dataset_name, cache_dir)
            
            # 获取训练数据
            comments, labels = loader.get_data_for_analysis(
                split='train',
                sample_size=sample_size,
                balance_classes=True
            )
            
            print(f"成功加载 {len(comments)} 条评论数据用于LSTM训练")
            
            # 打印数据分布信息
            unique_labels, counts = np.unique(labels, return_counts=True)
            print("数据分布:")
            for label, count in zip(unique_labels, counts):
                sentiment_name = "正面" if label == 1 else ("负面" if label == -1 else "中性")
                percentage = count / len(labels) * 100
                print(f"  {sentiment_name}: {count} 条 ({percentage:.1f}%)")
            
            # 使用现有的分析方法
            return self.analyze_comments_with_data(comments, labels)
            
        except Exception as e:
            print(f"❌ 从Hugging Face加载LSTM数据失败: {str(e)}")
            print("回退到本地文件分析...")
            # 回退到原有的文件分析方法
            return self.analyze_comments([], max_comments=sample_size or 80000)

def main():
    # 创建深度学习情感分析对象
    dl_analyzer = DeepLearningSentiment()
    
    # 获取所有评论文件列表
    comments_files = []
    for category in ['好评', '中评', '差评']:
        files = glob.glob(f"comments/*{category}*.xls") + glob.glob(f"comments/*{category}*.xlsx")
        # 读取所有文件，不再限制数量
        comments_files.extend(files)
    
    print(f"找到 {len(comments_files)} 个评论文件")
    
    # 分析评论，目标读取80000条评论
    results = dl_analyzer.analyze_comments(comments_files, max_comments=80000)
    
    print("\n分析完成！结果已保存到 lstm_training_history.png")

if __name__ == "__main__":
    main() 