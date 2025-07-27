#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re

# 读取论文文件
with open('论文-202507.md', 'r', encoding='utf-8') as f:
    content = f.read()

# 读取新的参考文献
with open('参考文献.txt', 'r', encoding='utf-8') as f:
    new_references = f.read()

# 使用正则表达式查找参考文献部分
pattern = r'## 参考文献\s*\n\n(\[\d+\].*?)\n\n'
if '## 参考文献' in content:
    # 分割文件，找到参考文献部分
    parts = content.split('## 参考文献', 1)
    if len(parts) == 2:
        # 进一步分割，找到参考文献部分的结束位置
        before_refs = parts[0]
        refs_and_after = parts[1]
        
        # 查找参考文献部分的结束位置（下一个章节开始）
        next_section_match = re.search(r'\n\n##\s', refs_and_after)
        if next_section_match:
            after_refs = refs_and_after[next_section_match.start():]
            # 组合新内容
            new_content = before_refs + new_references + after_refs
        else:
            # 如果参考文献是最后一部分，直接添加
            new_content = before_refs + new_references
        
        # 写入新文件
        with open('论文-202507-fixed.md', 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("参考文献已更新，新文件保存为 '论文-202507-fixed.md'")
    else:
        print("无法找到参考文献部分的结束位置")
else:
    print("未找到参考文献部分") 