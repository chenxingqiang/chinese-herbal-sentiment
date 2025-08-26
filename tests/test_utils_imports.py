#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试脚本：验证 utils 模块的导入是否正常工作
"""

def test_core_imports():
    """测试核心类的导入"""
    print("🧪 测试核心类导入...")
    
    try:
        from chinese_herbal_sentiment.utils import DatasetLoader, load_chinese_herbal_dataset
        print("✅ DatasetLoader 导入成功")
        
        from chinese_herbal_sentiment.utils import KeywordMapping
        print("✅ KeywordMapping 导入成功")
        
        return True
    except ImportError as e:
        print(f"❌ 核心类导入失败: {e}")
        return False

def test_color_scheme_imports():
    """测试配色方案的导入"""
    print("\n🎨 测试配色方案导入...")
    
    try:
        from chinese_herbal_sentiment.utils import (
            setup_unified_style, get_custom_font, get_color_palette,
            UNIFIED_COLORS, COLOR_SCHEMES
        )
        print("✅ 配色方案模块导入成功")
        
        # 测试配色方案是否可用
        colors = get_color_palette('algorithm_performance', 3)
        print(f"✅ 获取算法性能配色: {colors[:3]}")
        
        return True
    except ImportError as e:
        print(f"❌ 配色方案导入失败: {e}")
        return False

def test_data_analysis_imports():
    """测试数据分析函数的导入"""
    print("\n📊 测试数据分析函数导入...")
    
    try:
        from chinese_herbal_sentiment.utils import (
            load_excel_files, preprocess_text, extract_keywords
        )
        print("✅ 数据分析函数导入成功")
        
        return True
    except ImportError as e:
        print(f"❌ 数据分析函数导入失败: {e}")
        return False

def test_figure_generation_imports():
    """测试图表生成函数的导入"""
    print("\n📈 测试图表生成函数导入...")
    
    try:
        from chinese_herbal_sentiment.utils import (
            generate_chinese_figures, generate_unified_figures,
            generate_all_training_figures, generate_lstm_architecture
        )
        print("✅ 图表生成函数导入成功")
        
        return True
    except ImportError as e:
        print(f"❌ 图表生成函数导入失败: {e}")
        return False

def test_academic_search_imports():
    """测试学术搜索函数的导入"""
    print("\n🔍 测试学术搜索函数导入...")
    
    try:
        from chinese_herbal_sentiment.utils import (
            search_google_scholar, generate_tcm_search_queries
        )
        print("✅ 学术搜索函数导入成功")
        
        return True
    except ImportError as e:
        print(f"❌ 学术搜索函数导入失败: {e}")
        return False

def test_all_imports():
    """测试完整的 __all__ 导入"""
    print("\n🔧 测试完整 __all__ 导入...")
    
    try:
        import chinese_herbal_sentiment.utils as utils
        print("✅ 使用模块导入成功")
        
        # 验证几个关键函数是否可用
        print(f"✅ DatasetLoader 类可用: {hasattr(utils, 'DatasetLoader')}")
        print(f"✅ load_chinese_herbal_dataset 函数可用: {hasattr(utils, 'load_chinese_herbal_dataset')}")
        print(f"✅ UNIFIED_COLORS 配色可用: {hasattr(utils, 'UNIFIED_COLORS')}")
        
        return True
    except Exception as e:
        print(f"❌ 完整导入失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("中药材情感分析系统 - Utils 模块导入测试")
    print("=" * 60)
    
    results = []
    
    # 运行所有测试
    results.append(test_core_imports())
    results.append(test_color_scheme_imports())
    results.append(test_data_analysis_imports())
    results.append(test_figure_generation_imports())
    results.append(test_academic_search_imports())
    results.append(test_all_imports())
    
    # 输出测试结果
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 所有测试通过！({passed}/{total})")
        print("✅ Utils 模块重新整理成功")
    else:
        print(f"⚠️  部分测试失败：({passed}/{total})")
        print("❌ 需要检查并修复失败的导入")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
