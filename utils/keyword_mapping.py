#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import jieba
import re
import json
import matplotlib.pyplot as plt
from collections import defaultdict
from gensim import corpora, models, similarities
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class KeywordMapping:
    def __init__(self, comments_dir='comments'):
        """
        初始化关键词映射类
        
        参数:
        comments_dir: 评论数据目录
        """
        self.comments_dir = comments_dir
        self.stop_words = set()
        self.indicator_system = self._create_indicator_system()
        self.indicator_keywords = self._create_indicator_keywords()
        self.load_stop_words()
    
    def load_stop_words(self):
        """
        加载停用词
        """
        stop_words = [
            '的', '了', '和', '是', '就', '都', '而', '及', '与', '或', '一个',
            '那个', '这个', '一', '在', '中', '有', '对', '上', '下', '但', '也',
            '我', '你', '他', '她', '它', '们', '这', '那', '啊', '吧', '呢', '吗'
        ]
        self.stop_words = set(stop_words)
    
    def _create_indicator_system(self):
        """
        创建评价指标体系
        
        返回:
        指标体系字典
        """
        indicator_system = {
            "upstream": {
                "material_quality": "原料质量评分",
                "material_consistency": "原料规格一致性",
                "material_traceability": "原材料可追溯性评分",
                "material_price": "原料价格合理性",
                "supply_stability": "供应稳定性"
            },
            "midstream": {
                "production_efficiency": "生产效率评分",
                "technology": "工艺技术评价",
                "quality_standard": "质检标准符合度",
                "product_consistency": "产品一致性",
                "processing_environment": "加工环境评价"
            },
            "downstream": {
                "inventory_management": "库存管理评分",
                "order_accuracy": "订单准确性",
                "delivery_speed": "交货速度",
                "packaging": "包装评分",
                "after_sales_service": "售后服务质量",
                "information_transparency": "信息透明度"
            }
        }
        return indicator_system
    
    def _create_indicator_keywords(self):
        """
        为每个指标创建关键词集合
        
        返回:
        指标-关键词字典
        """
        indicator_keywords = {
            # 上游（原料）维度
            "material_quality": [
                "新鲜", "品质", "原料", "质量", "好", "纯正", "真品", "优质", 
                "天然", "农残", "重金属", "纯度", "霉变", "变质", "陈旧"
            ],
            "material_consistency": [
                "规格", "一致", "大小", "均匀", "整齐", "统一", "标准", 
                "差异", "不一", "参差不齐", "尺寸"
            ],
            "material_traceability": [
                "产地", "来源", "追溯", "溯源", "产地证明", "种植环境", 
                "种植基地", "采收", "采摘时间", "生产日期"
            ],
            "material_price": [
                "价格", "实惠", "便宜", "贵", "合理", "物有所值", "性价比", 
                "划算", "昂贵", "优惠"
            ],
            "supply_stability": [
                "供应", "稳定", "缺货", "断货", "库存", "有货", "现货", 
                "常年供应", "季节性", "货源"
            ],
            
            # 中游（加工）维度
            "production_efficiency": [
                "生产效率", "加工速度", "交货周期", "等待时间", "生产周期", 
                "加工时间", "效率", "快", "慢"
            ],
            "technology": [
                "工艺", "技术", "传统", "现代", "精细", "粗糙", "精湛", 
                "专业", "先进", "落后", "加工方式"
            ],
            "quality_standard": [
                "质量标准", "检验", "符合标准", "检测报告", "质检", "认证", 
                "合格证", "不合格", "检测", "质量问题"
            ],
            "product_consistency": [
                "一致性", "批次差异", "稳定", "每次", "波动", "差别", 
                "相同", "变化", "持续", "稳定性"
            ],
            "processing_environment": [
                "加工环境", "卫生", "洁净", "污染", "干净", "脏", "安全", 
                "健康", "环境", "加工条件", "设施"
            ],
            
            # 下游（销售与物流）维度
            "inventory_management": [
                "库存", "管理", "现货", "缺货", "积压", "备货", "存货", 
                "储备", "库存量", "库存不足"
            ],
            "order_accuracy": [
                "订单", "准确", "错误", "发错", "正确", "对应", "不符", 
                "匹配", "数量不对", "品种错误"
            ],
            "delivery_speed": [
                "发货", "配送", "物流", "速度", "快递", "及时", "延迟", 
                "慢", "等待", "收货", "到货时间"
            ],
            "packaging": [
                "包装", "精美", "结实", "破损", "保护", "盒子", "袋子", 
                "纸箱", "防潮", "密封", "完好"
            ],
            "after_sales_service": [
                "售后", "服务", "退换", "退款", "客服", "解决问题", "态度", 
                "响应", "回复", "投诉", "解决方案"
            ],
            "information_transparency": [
                "信息", "透明", "说明", "介绍", "详细", "隐瞒", "标注", 
                "标签", "成分", "含量", "说明书", "配料表"
            ]
        }
        return indicator_keywords
    
    def preprocess_text(self, text):
        """
        文本预处理：去除特殊字符，分词等
        
        参数:
        text: 输入的文本
        
        返回:
        分词列表
        """
        # 去除特殊字符
        text = re.sub(r'[^\w\s]', '', text)
        
        # 分词
        words = jieba.lcut(text)
        
        # 去除停用词
        words = [word for word in words if word not in self.stop_words and len(word.strip()) > 0]
        
        return words
    
    def load_excel_data(self):
        """
        加载Excel评论数据
        
        返回:
        评论列表
        """
        all_comments = []
        
        # 遍历评论目录下的所有文件
        for filename in os.listdir(self.comments_dir):
            if not (filename.endswith('.xls') or filename.endswith('.xlsx')):
                continue
            
            filepath = os.path.join(self.comments_dir, filename)
            try:
                # 读取Excel文件
                df = pd.read_excel(filepath)
                
                # 找到包含评论的列（可能需要根据实际数据调整）
                comment_column = None
                for col in df.columns:
                    if isinstance(col, str) and ('评论' in col or '内容' in col or 'content' in col.lower()):
                        comment_column = col
                        break
                
                if comment_column is None and len(df.columns) > 0:
                    # 如果找不到，默认使用第一列
                    comment_column = df.columns[0]
                
                if comment_column is not None:
                    # 提取评论内容
                    comments = df[comment_column].dropna().astype(str).tolist()
                    all_comments.extend(comments)
                
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")
                
        return all_comments
    
    def keyword_extraction(self, comments, method='tfidf', top_n=100):
        """
        从评论中提取关键词
        
        参数:
        comments: 评论列表
        method: 提取方法，'tfidf'或'count'
        top_n: 返回前N个关键词
        
        返回:
        关键词及其权重的列表
        """
        # 文本预处理
        preprocessed_comments = []
        for comment in comments:
            words = self.preprocess_text(comment)
            if words:
                preprocessed_comments.append(' '.join(words))
        
        if method == 'tfidf':
            # 使用TF-IDF提取关键词
            vectorizer = TfidfVectorizer(max_features=1000)
            X = vectorizer.fit_transform(preprocessed_comments)
            
            # 获取特征名称（即词语）
            feature_names = vectorizer.get_feature_names_out()
            
            # 计算每个词的TF-IDF值的平均值
            tfidf_mean = X.mean(axis=0).A1
            
            # 构建词-权重对
            keywords = [(feature_names[i], tfidf_mean[i]) for i in range(len(feature_names))]
            
            # 按权重排序并返回前N个
            return sorted(keywords, key=lambda x: x[1], reverse=True)[:top_n]
            
        elif method == 'count':
            # 使用词频提取关键词
            all_words = []
            for comment in comments:
                all_words.extend(self.preprocess_text(comment))
            
            # 计算词频
            word_counts = {}
            for word in all_words:
                if word in word_counts:
                    word_counts[word] += 1
                else:
                    word_counts[word] = 1
            
            # 构建词-权重对
            keywords = [(word, count) for word, count in word_counts.items()]
            
            # 按权重排序并返回前N个
            return sorted(keywords, key=lambda x: x[1], reverse=True)[:top_n]
        
        else:
            raise ValueError("method must be either 'tfidf' or 'count'")
    
    def keyword_mapping_rule_based(self, keywords):
        """
        基于规则的关键词映射
        
        参数:
        keywords: 关键词及其权重的列表
        
        返回:
        指标-关键词映射结果
        """
        mapping_result = {
            "upstream": defaultdict(list),
            "midstream": defaultdict(list),
            "downstream": defaultdict(list)
        }
        
        # 遍历每个关键词
        for keyword, weight in keywords:
            mapped = False
            
            # 遍历每个维度和指标
            for dimension, indicators in self.indicator_system.items():
                for indicator_code, indicator_name in indicators.items():
                    # 检查关键词是否在指标的关键词列表中
                    if keyword in self.indicator_keywords[indicator_code]:
                        mapping_result[dimension][indicator_code].append((keyword, weight))
                        mapped = True
                        break
                
                if mapped:
                    break
        
        # 转换默认字典为普通字典
        for dimension in mapping_result:
            mapping_result[dimension] = dict(mapping_result[dimension])
        
        return mapping_result
    
    def keyword_mapping_semantic(self, keywords):
        """
        基于语义的关键词映射
        
        参数:
        keywords: 关键词及其权重的列表
        
        返回:
        指标-关键词映射结果
        """
        mapping_result = {
            "upstream": defaultdict(list),
            "midstream": defaultdict(list),
            "downstream": defaultdict(list)
        }
        
        # 构建指标描述
        indicator_texts = {}
        for dimension, indicators in self.indicator_system.items():
            for indicator_code, indicator_name in indicators.items():
                indicator_keywords = self.indicator_keywords[indicator_code]
                indicator_texts[indicator_code] = indicator_name + " " + " ".join(indicator_keywords)
        
        # 对每个关键词计算与各指标的相似度
        keywords_list = [kw for kw, _ in keywords]
        for i, (keyword, weight) in enumerate(keywords):
            best_score = 0
            best_dimension = None
            best_indicator = None
            
            for dimension, indicators in self.indicator_system.items():
                for indicator_code, indicator_name in indicators.items():
                    # 计算关键词与指标关键词的相似度
                    indicator_keywords = self.indicator_keywords[indicator_code]
                    
                    # 计算相似度（简化版，实际应使用词向量）
                    # 这里使用关键词是否在指标关键词中，或指标关键词是否在关键词中
                    for ind_kw in indicator_keywords:
                        if ind_kw in keyword or keyword in ind_kw:
                            score = len(set(ind_kw).intersection(set(keyword))) / max(len(ind_kw), len(keyword))
                            if score > best_score:
                                best_score = score
                                best_dimension = dimension
                                best_indicator = indicator_code
            
            # 如果相似度超过阈值，则映射到该指标
            if best_score > 0.3:  # 阈值可调整
                mapping_result[best_dimension][best_indicator].append((keyword, weight))
        
        # 转换默认字典为普通字典
        for dimension in mapping_result:
            mapping_result[dimension] = dict(mapping_result[dimension])
        
        return mapping_result
    
    def calculate_indicator_scores(self, mapping_result):
        """
        计算各指标的得分
        
        参数:
        mapping_result: 指标-关键词映射结果
        
        返回:
        各指标的得分
        """
        scores = {}
        
        # 遍历每个维度
        for dimension, indicators in mapping_result.items():
            dimension_scores = {}
            
            # 遍历每个指标
            for indicator_code, keywords in indicators.items():
                if keywords:  # 如果有映射到该指标的关键词
                    # 计算加权平均得分
                    total_weight = sum(weight for _, weight in keywords)
                    indicator_scores = {}
                    
                    # 遍历每个关键词
                    for keyword, weight in keywords:
                        # 这里可以加入情感分析，或者使用预先计算的情感得分
                        # 这里简单地使用权重作为得分
                        indicator_scores[keyword] = weight
                    
                    # 计算指标得分（加权平均）
                    if total_weight > 0:
                        dimension_scores[indicator_code] = sum(indicator_scores.values()) / total_weight
                    else:
                        dimension_scores[indicator_code] = 0
                else:
                    dimension_scores[indicator_code] = 0
            
            scores[dimension] = dimension_scores
        
        return scores
    
    def visualize_indicator_scores(self, scores):
        """
        可视化各指标得分
        
        参数:
        scores: 各指标的得分
        """
        # 创建图表
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        dimensions = ['upstream', 'midstream', 'downstream']
        dimension_names = ['上游（原料）', '中游（加工）', '下游（销售与物流）']
        
        # 遍历每个维度
        for i, dimension in enumerate(dimensions):
            if dimension in scores:
                indicators = scores[dimension]
                
                # 获取指标名称和得分
                indicator_codes = list(indicators.keys())
                indicator_names = [self.indicator_system[dimension][code] for code in indicator_codes]
                scores_list = [indicators[code] for code in indicator_codes]
                
                # 创建条形图
                axes[i].barh(indicator_names, scores_list)
                axes[i].set_title(dimension_names[i])
                axes[i].set_xlabel('得分')
        
        # 保存图表
        plt.tight_layout()
        plt.savefig('indicator_scores.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_mapping_results(self, mapping_result, filename='mapping_results.json'):
        """
        保存映射结果到JSON文件
        
        参数:
        mapping_result: 指标-关键词映射结果
        filename: 保存的文件名
        """
        # 转换结果为可序列化的格式
        serializable_result = {}
        for dimension, indicators in mapping_result.items():
            serializable_result[dimension] = {}
            for indicator_code, keywords in indicators.items():
                serializable_result[dimension][indicator_code] = [
                    {"keyword": kw, "weight": float(wt)} for kw, wt in keywords
                ]
        
        # 保存到文件
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, ensure_ascii=False, indent=2)
    
    def run_mapping(self):
        """
        运行完整的关键词映射流程
        
        返回:
        mapping_result: 指标-关键词映射结果
        scores: 各指标的得分
        """
        print("开始加载评论数据...")
        comments = self.load_excel_data()
        print(f"加载完成，共有 {len(comments)} 条评论")
        
        print("开始提取关键词...")
        keywords = self.keyword_extraction(comments, method='tfidf', top_n=200)
        print(f"提取完成，共有 {len(keywords)} 个关键词")
        
        print("开始规则映射...")
        rule_mapping_result = self.keyword_mapping_rule_based(keywords)
        
        print("开始语义映射...")
        semantic_mapping_result = self.keyword_mapping_semantic(keywords)
        
        # 合并两种映射结果
        mapping_result = rule_mapping_result.copy()
        for dimension, indicators in semantic_mapping_result.items():
            for indicator_code, keywords in indicators.items():
                if indicator_code not in mapping_result[dimension]:
                    mapping_result[dimension][indicator_code] = keywords
                else:
                    # 合并关键词，避免重复
                    existing_keywords = {kw for kw, _ in mapping_result[dimension][indicator_code]}
                    for kw, wt in keywords:
                        if kw not in existing_keywords:
                            mapping_result[dimension][indicator_code].append((kw, wt))
        
        print("计算指标得分...")
        scores = self.calculate_indicator_scores(mapping_result)
        
        print("可视化指标得分...")
        self.visualize_indicator_scores(scores)
        
        print("保存映射结果...")
        self.save_mapping_results(mapping_result)
        
        print("分析完成！")
        
        return mapping_result, scores


if __name__ == "__main__":
    mapper = KeywordMapping()
    mapping_result, scores = mapper.run_mapping() 