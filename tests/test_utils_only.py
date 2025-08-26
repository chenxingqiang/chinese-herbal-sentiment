#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简化测试脚本：只测试 utils 模块的导入
"""

def test_utils_imports():
    """测试 utils 模块导入"""
    print("🧪 测试 utils 模块导入...")
    
    try:
        # 测试数据集加载器
        from chinese_herbal_sentiment.utils.dataset_loader import DatasetLoader, load_chinese_herbal_dataset
        print("✅ DatasetLoader 导入成功")
        
        # 测试关键词映射
        from chinese_herbal_sentiment.utils.keyword_mapping import KeywordMapping
        print("✅ KeywordMapping 导入成功")
        
        # 测试配色方案
        from chinese_herbal_sentiment.utils.unified_color_scheme import (
            setup_unified_style, get_custom_font, get_color_palette,
            UNIFIED_COLORS, COLOR_SCHEMES
        )
        print("✅ 配色方案导入成功")
        
        # 测试数据分析
        from chinese_herbal_sentiment.utils.data_analysis import (
            load_excel_files, preprocess_text, extract_keywords, analyze_sentiment_distribution
        )
        print("✅ 数据分析函数导入成功")
        
        # 测试学术搜索
        from chinese_herbal_sentiment.utils.scholar_search import (
            search_google_scholar, generate_tcm_search_queries
        )
        print("✅ 学术搜索函数导入成功")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_utils_module():
    """测试完整的 utils 模块"""
    print("\n🔧 测试完整 utils 模块...")
    
    try:
        import chinese_herbal_sentiment.utils as utils
        print("✅ utils 模块导入成功")
        
        # 检查主要属性
        attrs_to_check = [
            'DatasetLoader', 'load_chinese_herbal_dataset', 'KeywordMapping',
            'UNIFIED_COLORS', 'setup_unified_style', 'load_excel_files',
            'search_google_scholar'
        ]
        
        missing_attrs = []
        for attr in attrs_to_check:
            if hasattr(utils, attr):
                print(f"✅ {attr} 可用")
            else:
                missing_attrs.append(attr)
                print(f"❌ {attr} 不可用")
        
        if missing_attrs:
            print(f"⚠️  缺少属性: {missing_attrs}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 模块导入失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("Utils 模块导入测试")
    print("=" * 60)
    
    results = []
    
    # 运行测试
    results.append(test_utils_imports())
    results.append(test_utils_module())
    
    # 输出结果
    print("\n" + "=" * 60)
    print("测试结果")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 所有测试通过！({passed}/{total})")
        print("✅ Utils 模块整理成功")
    else:
        print(f"⚠️  部分测试失败：({passed}/{total})")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
