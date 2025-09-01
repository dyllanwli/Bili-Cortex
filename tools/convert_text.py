#!/usr/bin/env python3
"""
文本繁简转换工具
用于转录结果的后处理
"""

import sys
import argparse
from pathlib import Path

# 简单的繁简对照表 (部分常用字符)
TRADITIONAL_TO_SIMPLIFIED = {
    '這': '这', '個': '个', '來': '来', '過': '过', '時': '时',
    '問': '问', '題': '题', '還': '还', '會': '会', '後': '后',
    '現': '现', '開': '开', '關': '关', '長': '长', '說': '说',
    '見': '见', '聽': '听', '買': '买', '賣': '卖', '錢': '钱',
    '業': '业', '專': '专', '學': '学', '習': '习', '練': '练',
    '經': '经', '營': '营', '場': '场', '機': '机', '會': '会',
    '種': '种', '類': '类', '別': '别', '計': '计', '劃': '划',
    '進': '进', '選': '选', '擇': '择', '決': '决', '定': '定',
    '確': '确', '實': '实', '際': '际', '況': '况', '標': '标',
    '準': '准', '則': '则', '規': '规', '務': '务', '議': '议',
    '論': '论', '講': '讲', '話': '话', '語': '语', '詞': '词'
}

SIMPLIFIED_TO_TRADITIONAL = {v: k for k, v in TRADITIONAL_TO_SIMPLIFIED.items()}


def convert_traditional_to_simplified(text: str) -> str:
    """繁体转简体"""
    result = text
    for trad, simp in TRADITIONAL_TO_SIMPLIFIED.items():
        result = result.replace(trad, simp)
    return result


def convert_simplified_to_traditional(text: str) -> str:
    """简体转繁体"""
    result = text
    for simp, trad in SIMPLIFIED_TO_TRADITIONAL.items():
        result = result.replace(simp, trad)
    return result


def convert_transcript_file(input_file: Path, output_file: Path, mode: str):
    """转换转录文件"""
    try:
        content = input_file.read_text(encoding='utf-8')
        
        if mode == 'to_simplified':
            converted_content = convert_traditional_to_simplified(content)
            print(f"繁体 → 简体: {input_file} → {output_file}")
        elif mode == 'to_traditional':
            converted_content = convert_simplified_to_traditional(content)
            print(f"简体 → 繁体: {input_file} → {output_file}")
        else:
            raise ValueError("Invalid mode")
        
        output_file.write_text(converted_content, encoding='utf-8')
        print("转换完成！")
        
    except Exception as e:
        print(f"转换失败: {e}")
        return 1
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="转录文本繁简转换工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 繁体转简体
  python tools/convert_text.py --to-simplified transcript.md output.md
  
  # 简体转繁体
  python tools/convert_text.py --to-traditional transcript.md output.md
  
  # 批量转换目录下所有.md文件
  python tools/convert_text.py --to-simplified data/transcripts/ converted/
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--to-simplified', action='store_true', help='转换为简体中文')
    group.add_argument('--to-traditional', action='store_true', help='转换为繁体中文')
    
    parser.add_argument('input', help='输入文件或目录')
    parser.add_argument('output', help='输出文件或目录')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    mode = 'to_simplified' if args.to_simplified else 'to_traditional'
    
    if input_path.is_file():
        # 单文件转换
        if output_path.is_dir():
            output_path = output_path / input_path.name
        
        return convert_transcript_file(input_path, output_path, mode)
    
    elif input_path.is_dir():
        # 批量转换
        output_path.mkdir(parents=True, exist_ok=True)
        
        success_count = 0
        total_count = 0
        
        for md_file in input_path.glob('*.md'):
            total_count += 1
            output_file = output_path / md_file.name
            
            if convert_transcript_file(md_file, output_file, mode) == 0:
                success_count += 1
        
        print(f"\n批量转换完成: {success_count}/{total_count} 个文件")
        return 0 if success_count == total_count else 1
    
    else:
        print(f"输入路径不存在: {input_path}")
        return 1


if __name__ == "__main__":
    sys.exit(main())