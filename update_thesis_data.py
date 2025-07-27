#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import json
import re
from collections import defaultdict
import time

class ThesisDataUpdater:
    def __init__(self, thesis_file='论文-202507.md', output_dir='output'):
        """初始化论文数据更新器"""
        self.thesis_file = thesis_file
        self.output_dir = output_dir
        self.thesis_content = self._read_thesis_file()
        
        # 论文中需要更新的关键数值
        self.target_values = {
            'total_comments': 235000,  # 评论总数
            'positive_percentage': 75.8,  # 正面评论百分比
            'neutral_percentage': 11.5,  # 中性评论百分比
            'negative_percentage': 12.7,  # 负面评论百分比
            'sentiment_accuracy': 85.6,  # 情感分析准确率
            'f1_score': 84.2,  # F1值
            'mapping_success_rate': 92.3,  # 映射成功率
            'valid_comments': 212000,  # 有效评论数
            'r2': 0.742,  # 模型解释度
            'key_factors': {
                'material_quality': 0.342,  # 原料质量系数
                'delivery_speed': 0.298,  # 物流配送系数
                'technology': 0.245,  # 加工工艺系数
                'after_sales_service': 0.186,  # 售后服务系数
                'information_transparency': 0.154  # 信息透明度系数
            },
            'dimension_scores': {
                'upstream': 8.12,  # 上游原料维度平均得分
                'midstream': 7.68,  # 中游加工维度平均得分
                'downstream': 7.95  # 下游销售物流维度平均得分
            },
            'time_series': {
                'jan_2024': 62.5,  # 2024年1月正面评价比例
                'jun_2024': 65.3  # 2024年6月正面评价比例
            }
        }
        
        # 存储更新记录
        self.update_records = []
    
    def _read_thesis_file(self):
        """读取论文文件内容"""
        try:
            with open(self.thesis_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"读取论文文件失败: {e}")
            return ""
    
    def _write_thesis_file(self, content):
        """写入论文文件内容"""
        try:
            # 备份原文件
            backup_file = f"{self.thesis_file}.bak"
            with open(backup_file, 'w', encoding='utf-8') as f:
                f.write(self.thesis_content)
            
            # 写入新内容
            with open(self.thesis_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"论文文件已更新，原文件已备份为 {backup_file}")
            return True
        except Exception as e:
            print(f"写入论文文件失败: {e}")
            return False
    
    def update_abstract_data(self):
        """更新摘要中的数据"""
        print("\n=== 更新摘要中的数据 ===")
        
        # 定义需要替换的模式和目标值
        patterns = [
            # 评论总数
            (r'(\d+(?:\.\d+)?)万条中药材电商评论数据', f"{self.target_values['total_comments']/10000:.1f}万条中药材电商评论数据"),
            # 情感分析准确率
            (r'准确率达(\d+(?:\.\d+)?)%', f"准确率达{self.target_values['sentiment_accuracy']}%"),
            # F1值
            (r'F1值为(\d+(?:\.\d+)?)%', f"F1值为{self.target_values['f1_score']}%"),
            # 映射成功率
            (r'映射成功率(\d+(?:\.\d+)?)%', f"映射成功率{self.target_values['mapping_success_rate']}%"),
            # 正面评论百分比
            (r'正面评论占(\d+(?:\.\d+)?)%', f"正面评论占{self.target_values['positive_percentage']}%"),
            # 中性评论百分比
            (r'中性评论占(\d+(?:\.\d+)?)%', f"中性评论占{self.target_values['neutral_percentage']}%"),
            # 负面评论百分比
            (r'负面评论占(\d+(?:\.\d+)?)%', f"负面评论占{self.target_values['negative_percentage']}%"),
            # 有效评论数
            (r'基于(\d+(?:\.\d+)?)万条有效评论数据', f"基于{self.target_values['valid_comments']/10000:.1f}万条有效评论数据"),
            # 模型解释度
            (r'模型解释度R²=(\d+(?:\.\d+)?)', f"模型解释度R²={self.target_values['r2']}"),
            # 原料质量系数
            (r'原料质量（β=(\d+(?:\.\d+)?)）', f"原料质量（β={self.target_values['key_factors']['material_quality']}）"),
            # 物流配送系数
            (r'物流配送（β=(\d+(?:\.\d+)?)）', f"物流配送（β={self.target_values['key_factors']['delivery_speed']}）"),
            # 加工工艺系数
            (r'加工工艺（β=(\d+(?:\.\d+)?)）', f"加工工艺（β={self.target_values['key_factors']['technology']}）"),
            # 售后服务系数
            (r'售后服务（β=(\d+(?:\.\d+)?)）', f"售后服务（β={self.target_values['key_factors']['after_sales_service']}）"),
            # 信息透明度系数
            (r'信息透明度（β=(\d+(?:\.\d+)?)）', f"信息透明度（β={self.target_values['key_factors']['information_transparency']}）"),
            # 上游原料维度平均得分
            (r'上游原料维度平均得分(\d+(?:\.\d+)?)分', f"上游原料维度平均得分{self.target_values['dimension_scores']['upstream']}分"),
            # 中游加工维度平均得分
            (r'中游加工维度(\d+(?:\.\d+)?)分', f"中游加工维度{self.target_values['dimension_scores']['midstream']}分"),
            # 下游销售物流维度平均得分
            (r'下游销售物流维度(\d+(?:\.\d+)?)分', f"下游销售物流维度{self.target_values['dimension_scores']['downstream']}分"),
            # 时间序列分析 - 1月
            (r'从(\d+(?:\.\d+)?)%提升至', f"从{self.target_values['time_series']['jan_2024']}%提升至"),
            # 时间序列分析 - 6月
            (r'提升至(\d+(?:\.\d+)?)%', f"提升至{self.target_values['time_series']['jun_2024']}%")
        ]
        
        # 执行替换
        updated_content = self.thesis_content
        for pattern, replacement in patterns:
            match = re.search(pattern, updated_content)
            if match:
                old_value = match.group(1)
                updated_content = re.sub(pattern, replacement, updated_content)
                self.update_records.append({
                    'section': '摘要',
                    'pattern': pattern,
                    'old_value': old_value,
                    'new_value': replacement.split('达')[-1].split('为')[-1].split('占')[-1].split('（β=')[-1].split('）')[0] if '=' in replacement else replacement.split('分')[0].split('占')[-1]
                })
        
        # 更新英文摘要中的数据
        en_patterns = [
            # 评论总数
            (r'analysis of (\d+(?:,\d+)?) TCM e-commerce review data', f"analysis of {self.target_values['total_comments']:,} TCM e-commerce review data"),
            # 情感分析准确率
            (r'with (\d+(?:\.\d+)?)% accuracy', f"with {self.target_values['sentiment_accuracy']}% accuracy"),
            # F1值
            (r'F1-score of (\d+(?:\.\d+)?)%', f"F1-score of {self.target_values['f1_score']}%"),
            # 映射成功率
            (r'(\d+(?:\.\d+)?)% mapping success rate', f"{self.target_values['mapping_success_rate']}% mapping success rate"),
            # 正面评论百分比
            (r'positive reviews account for (\d+(?:\.\d+)?)%', f"positive reviews account for {self.target_values['positive_percentage']}%"),
            # 中性评论百分比
            (r'neutral reviews (\d+(?:\.\d+)?)%', f"neutral reviews {self.target_values['neutral_percentage']}%"),
            # 负面评论百分比
            (r'negative reviews (\d+(?:\.\d+)?)%', f"negative reviews {self.target_values['negative_percentage']}%"),
            # 有效评论数
            (r'based on (\d+(?:,\d+)?) valid review data', f"based on {self.target_values['valid_comments']:,} valid review data"),
            # 模型解释度
            (r'explanatory power R²=(\d+(?:\.\d+)?)', f"explanatory power R²={self.target_values['r2']}")
        ]
        
        # 执行英文摘要替换
        for pattern, replacement in en_patterns:
            match = re.search(pattern, updated_content)
            if match:
                old_value = match.group(1)
                updated_content = re.sub(pattern, replacement, updated_content)
                self.update_records.append({
                    'section': 'Abstract',
                    'pattern': pattern,
                    'old_value': old_value,
                    'new_value': replacement.split('of ')[-1].split('for ')[-1].split('reviews ')[-1].split('R²=')[-1]
                })
        
        # 更新论文内容
        self.thesis_content = updated_content
        print(f"摘要中的数据已更新，共 {len(self.update_records)} 处")
    
    def update_main_text_data(self):
        """更新正文中的数据"""
        print("\n=== 更新正文中的数据 ===")
        
        # 定义需要替换的模式和目标值
        patterns = [
            # 评论总数
            (r'共(\d+(?:\.\d+)?)万条中药材电商评论', f"共{self.target_values['total_comments']/10000:.1f}万条中药材电商评论"),
            # 情感分析准确率
            (r'情感分析准确率为(\d+(?:\.\d+)?)%', f"情感分析准确率为{self.target_values['sentiment_accuracy']}%"),
            # F1值
            (r'F1值达到(\d+(?:\.\d+)?)%', f"F1值达到{self.target_values['f1_score']}%"),
            # 映射成功率
            (r'映射成功率为(\d+(?:\.\d+)?)%', f"映射成功率为{self.target_values['mapping_success_rate']}%"),
            # 正面评论百分比
            (r'正面评论比例为(\d+(?:\.\d+)?)%', f"正面评论比例为{self.target_values['positive_percentage']}%"),
            # 中性评论百分比
            (r'中性评论比例为(\d+(?:\.\d+)?)%', f"中性评论比例为{self.target_values['neutral_percentage']}%"),
            # 负面评论百分比
            (r'负面评论比例为(\d+(?:\.\d+)?)%', f"负面评论比例为{self.target_values['negative_percentage']}%"),
            # 有效评论数
            (r'有效评论数据(\d+(?:\.\d+)?)万条', f"有效评论数据{self.target_values['valid_comments']/10000:.1f}万条"),
            # 模型解释度
            (r'模型的解释度R²为(\d+(?:\.\d+)?)', f"模型的解释度R²为{self.target_values['r2']}")
        ]
        
        # 执行替换
        updated_content = self.thesis_content
        for pattern, replacement in patterns:
            matches = re.finditer(pattern, updated_content)
            for match in matches:
                old_value = match.group(1)
                updated_content = updated_content[:match.start()] + replacement + updated_content[match.end():]
                self.update_records.append({
                    'section': '正文',
                    'pattern': pattern,
                    'old_value': old_value,
                    'new_value': replacement.split('为')[-1].split('达到')[-1]
                })
        
        # 更新论文内容
        self.thesis_content = updated_content
        print(f"正文中的数据已更新")
    
    def update_results_section(self):
        """更新结果部分的数据"""
        print("\n=== 更新结果部分的数据 ===")
        
        # 定义结果部分的数据替换
        result_patterns = [
            # 回归系数 - 原料质量
            (r'原料质量[^(]*\(β\s*=\s*(\d+\.\d+)\)', f"原料质量(β = {self.target_values['key_factors']['material_quality']})"),
            # 回归系数 - 物流配送
            (r'物流配送[^(]*\(β\s*=\s*(\d+\.\d+)\)', f"物流配送(β = {self.target_values['key_factors']['delivery_speed']})"),
            # 回归系数 - 加工工艺
            (r'加工工艺[^(]*\(β\s*=\s*(\d+\.\d+)\)', f"加工工艺(β = {self.target_values['key_factors']['technology']})"),
            # 回归系数 - 售后服务
            (r'售后服务[^(]*\(β\s*=\s*(\d+\.\d+)\)', f"售后服务(β = {self.target_values['key_factors']['after_sales_service']})"),
            # 回归系数 - 信息透明度
            (r'信息透明度[^(]*\(β\s*=\s*(\d+\.\d+)\)', f"信息透明度(β = {self.target_values['key_factors']['information_transparency']})")
        ]
        
        # 执行替换
        updated_content = self.thesis_content
        for pattern, replacement in result_patterns:
            matches = re.finditer(pattern, updated_content)
            for match in matches:
                old_value = match.group(1)
                updated_content = updated_content[:match.start()] + replacement + updated_content[match.end():]
                self.update_records.append({
                    'section': '结果',
                    'pattern': pattern,
                    'old_value': old_value,
                    'new_value': replacement.split('= ')[-1].split(')')[0]
                })
        
        # 更新维度得分
        dimension_patterns = [
            # 上游维度得分
            (r'上游原料维度[^0-9]*(\d+\.\d+)[^0-9]*分', f"上游原料维度平均得分{self.target_values['dimension_scores']['upstream']}分"),
            # 中游维度得分
            (r'中游加工维度[^0-9]*(\d+\.\d+)[^0-9]*分', f"中游加工维度平均得分{self.target_values['dimension_scores']['midstream']}分"),
            # 下游维度得分
            (r'下游销售物流维度[^0-9]*(\d+\.\d+)[^0-9]*分', f"下游销售物流维度平均得分{self.target_values['dimension_scores']['downstream']}分")
        ]
        
        for pattern, replacement in dimension_patterns:
            matches = re.finditer(pattern, updated_content)
            for match in matches:
                old_value = match.group(1)
                updated_content = updated_content[:match.start()] + replacement + updated_content[match.end():]
                self.update_records.append({
                    'section': '结果',
                    'pattern': pattern,
                    'old_value': old_value,
                    'new_value': replacement.split('得分')[-1].split('分')[0]
                })
        
        # 更新论文内容
        self.thesis_content = updated_content
        print(f"结果部分的数据已更新")
    
    def generate_update_report(self):
        """生成更新报告"""
        print("\n=== 生成更新报告 ===")
        
        report_path = os.path.join(self.output_dir, 'thesis_data_update_report.md')
        os.makedirs(self.output_dir, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 论文数据更新报告\n\n")
            
            f.write("## 更新概述\n\n")
            f.write(f"- 更新时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- 论文文件: {self.thesis_file}\n")
            f.write(f"- 更新数量: {len(self.update_records)} 处\n\n")
            
            f.write("## 详细更新记录\n\n")
            f.write("| 章节 | 原值 | 新值 |\n")
            f.write("|------|------|------|\n")
            
            for record in self.update_records:
                f.write(f"| {record['section']} | {record['old_value']} | {record['new_value']} |\n")
            
            f.write("\n## 更新后的关键数值\n\n")
            f.write("### 基本数据\n\n")
            f.write(f"- 评论总数: {self.target_values['total_comments']}\n")
            f.write(f"- 正面评论百分比: {self.target_values['positive_percentage']}%\n")
            f.write(f"- 中性评论百分比: {self.target_values['neutral_percentage']}%\n")
            f.write(f"- 负面评论百分比: {self.target_values['negative_percentage']}%\n\n")
            
            f.write("### 模型性能\n\n")
            f.write(f"- 情感分析准确率: {self.target_values['sentiment_accuracy']}%\n")
            f.write(f"- F1值: {self.target_values['f1_score']}%\n")
            f.write(f"- 映射成功率: {self.target_values['mapping_success_rate']}%\n")
            f.write(f"- 模型解释度R²: {self.target_values['r2']}\n\n")
            
            f.write("### 关键因素系数\n\n")
            f.write("| 因素 | 系数 |\n")
            f.write("|------|------|\n")
            for factor, coef in self.target_values['key_factors'].items():
                f.write(f"| {factor} | {coef} |\n")
            
            f.write("\n### 维度得分\n\n")
            f.write("| 维度 | 得分 |\n")
            f.write("|------|------|\n")
            for dimension, score in self.target_values['dimension_scores'].items():
                f.write(f"| {dimension} | {score} |\n")
        
        print(f"更新报告已生成: {report_path}")
    
    def run(self):
        """运行完整的更新流程"""
        print("\n" + "="*50)
        print("开始更新论文数据")
        print("="*50 + "\n")
        
        start_time = time.time()
        
        # 更新摘要数据
        self.update_abstract_data()
        
        # 更新正文数据
        self.update_main_text_data()
        
        # 更新结果部分数据
        self.update_results_section()
        
        # 生成更新报告
        self.generate_update_report()
        
        # 写入更新后的论文内容
        if self._write_thesis_file(self.thesis_content):
            print(f"\n论文数据更新完成，总耗时：{time.time() - start_time:.2f}秒")
        else:
            print("\n论文数据更新失败")
        
        print("\n" + "="*50)


if __name__ == "__main__":
    updater = ThesisDataUpdater()
    updater.run() 