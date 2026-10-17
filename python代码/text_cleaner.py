"""
PDF文本清洗（v2）：保守策略，只去除明确无用内容
"""
import re


def clean_full_text(raw_text):
    """
    对从PDF提取的原始文本进行清洗。
    保守策略：只去除页眉页脚、合并断行、压缩空行。
    不再激进删除表格和数字行（交给各模块自行处理）。
    """
    text = raw_text

    # 1. 去除[PAGE_N]标记
    text = re.sub(r'\[PAGE_\d+\]', '', text)

    # 2. 去除页眉行（"XXX股份有限公司2023年年度报告全文"）
    text = re.sub(r'.{6,40}(?:股份有限公司|有限公司).{0,30}年年度报告全文.{0,20}\n?', '', text)

    # 3. 去除独立的页码行
    text = re.sub(r'\n\s*\d{1,4}\s*\n', '\n', text)

    # 4. 合并被硬换行打断的中文句子
    lines = text.split('\n')
    merged = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if i + 1 < len(lines):
            next_line = lines[i + 1]
            # 当前行以中文字符结尾（无句号/分号/冒号等），下一行以中文字符开头
            if (line and next_line and
                re.search(r'[一-鿿　-〿＀-￯]$', line) and
                not re.search(r'[。；：」』）\)\.\?:!]$', line) and
                re.search(r'^[一-鿿　-〿＀-￯]', next_line)):
                line = line + next_line
                i += 1
        merged.append(line)
        i += 1
    text = '\n'.join(merged)

    # 5. 压缩多余空行（4个以上→3个）
    text = re.sub(r'\n{4,}', '\n\n\n', text)

    return text


if __name__ == "__main__":
    from pdf_reader import load_or_extract
    from mapper import extract_company_list, build_pdf_map

    companies = extract_company_list()
    mapping = build_pdf_map(companies)

    for comp in mapping[:1]:
        raw = load_or_extract(comp['pdf_path'])
        print(f'原始文本: {len(raw):,} 字符')

        cleaned = clean_full_text(raw)
        reduction = (1 - len(cleaned)/len(raw)) * 100 if raw else 0
        print(f'清洗后: {len(cleaned):,} 字符 (-{reduction:.0f}%)')
