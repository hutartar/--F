"""
年报章节边界定位（v3.1：密度聚类跳过目录 + 行首偏好 + 位置校验）
"""
import re


def _get_all_section_positions(full_text, patterns):
    positions = []
    for pattern, name in patterns:
        for m in re.finditer(pattern, full_text):
            if m.start() > len(full_text) * 0.005:
                positions.append((m.start(), name))
    positions.sort(key=lambda x: x[0])
    return positions


def _find_toc_end(positions, full_text_len):
    """通过密度聚类找到目录区结束位置"""
    if len(positions) < 3:
        return int(full_text_len * 0.02)

    gaps = [positions[i + 1][0] - positions[i][0] for i in range(len(positions) - 1)]

    toc_end = int(full_text_len * 0.02)
    for i, gap in enumerate(gaps):
        if gap > 2500:
            toc_end = positions[i][0] + 500
            break

    return min(toc_end, int(full_text_len * 0.06))


def _is_real_section_header(full_text, match):
    """判断匹配是否是真正的章节标题（非交叉引用）"""
    before = full_text[max(0, match.start() - 40):match.start()]
    if re.search(r'(?:详见|参见|见本报告.{0,15}第|[在位于].{0,10}第)', before):
        return False
    # 行首优先（真正的章节标题通常在行首）
    if match.start() > 0 and full_text[match.start() - 1] == '\n':
        return True
    # 不在行首但也不像交叉引用，可能是PDF提取格式问题
    if not re.search(r'详见|参见', before):
        return True
    return False


def find_section_boundaries(full_text):
    """定位所有节起始位置，返回 {section_name: char_position}"""
    from config import SECTION_PATTERNS

    all_positions = _get_all_section_positions(full_text, SECTION_PATTERNS)
    toc_end = _find_toc_end(all_positions, len(full_text))

    boundaries = {}
    for pattern, name in SECTION_PATTERNS:
        matches = list(re.finditer(pattern, full_text))
        # 过滤目录区匹配
        real_matches = [m for m in matches if m.start() >= toc_end and _is_real_section_header(full_text, m)]

        if not real_matches:
            # 回退：不过滤目录，取最后一个非引用匹配
            real_matches = [m for m in matches if _is_real_section_header(full_text, m)]

        if real_matches:
            boundaries[name] = real_matches[-1].start()

    # 校验第十节位置
    # 财务报告通常在文档后半部分，但某些短报告可能在30%位置
    if "第十节" in boundaries:
        sec10_pos = boundaries["第十节"]
        if sec10_pos < len(full_text) * 0.08:
            # 只在非常靠前（<8%，明显在TOC区域）时才删除
            del boundaries["第十节"]
            for m in re.finditer(r'第十节\s*(?:财务报告|监事会报告)', full_text):
                if m.start() >= len(full_text) * 0.2 and _is_real_section_header(full_text, m):
                    boundaries["第十节"] = m.start()
                    break

    # 如果第十节仍然没找到，在文档后20%搜索
    if "第十节" not in boundaries:
        for m in re.finditer(r'第十节\s*(?:财务报告|监事会报告)', full_text):
            if m.start() >= len(full_text) * 0.2 and _is_real_section_header(full_text, m):
                boundaries["第十节"] = m.start()
                break
        if "第十节" not in boundaries:
            for pat in [r'\n一[、，]\s*审计报告', r'\n审计报告']:
                for m in re.finditer(pat, full_text):
                    if m.start() >= len(full_text) * 0.3:
                        boundaries["第十节"] = m.start()
                        break
                if "第十节" in boundaries:
                    break

    # 校验章节顺序：第四节必须在第三节之后
    if "第三节" in boundaries and "第四节" in boundaries:
        if boundaries["第四节"] < boundaries["第三节"]:
            del boundaries["第四节"]
            for m in re.finditer(r'第四节\s*公司治理', full_text):
                if m.start() > boundaries["第三节"] and _is_real_section_header(full_text, m):
                    boundaries["第四节"] = m.start()
                    break

    return boundaries


def find_subsection_position(full_text, start_pos, end_pos, patterns):
    """在指定范围内定位子节位置"""
    search_text = full_text[start_pos:end_pos] if end_pos else full_text[start_pos:]
    best_pos = None
    for pattern in patterns:
        matches = list(re.finditer(pattern, search_text))
        if matches:
            abs_pos = start_pos + matches[-1].start()
            best_pos = abs_pos
    return best_pos


def _check_xref(abs_pos, text_before, match_text):
    """检查匹配位置是否是交叉引用"""
    before = text_before + match_text[:20]
    if re.search(r'(?:详见|参见|见本报告.{0,15}第|见.{0,10}附注|之说明)', before):
        return True
    return False


def locate_target_subsections(full_text):
    """定位5个模块的目标子节位置"""
    boundaries = find_section_boundaries(full_text)
    if not boundaries:
        return {}

    targets = {}
    sec3_start = boundaries.get("第三节")
    sec4_start = boundaries.get("第四节")
    sec10_start = boundaries.get("第十节")

    # Module 1: 第三节 → 十一、公司未来发展的展望
    if sec3_start is not None:
        mda_end = sec4_start
        pos = find_subsection_position(
            full_text, sec3_start, mda_end,
            [r'十一[、，]\s*公司未来.*发展.*展望', r'公司未来.*发展.*展望', r'十一[、，].*展望']
        )
        if pos is not None:
            targets["module1"] = pos
        else:
            mda_text = full_text[sec3_start:sec4_start] if sec4_start else full_text[sec3_start:]
            for pat in [r'未来.*展望', r'发展.*展望', r'行业格局.*趋势', r'面临.*风险']:
                m = re.search(pat, mda_text)
                if m:
                    targets["module1"] = sec3_start + m.start()
                    break

    # Module 2: 第三节 → 衍生品投资情况
    if sec3_start is not None:
        mda_end = sec4_start
        pos = find_subsection_position(
            full_text, sec3_start, mda_end,
            [r'衍生品投资情况', r'套期保值']
        )
        if pos is not None:
            targets["module2"] = pos
        else:
            mda_text = full_text[sec3_start:sec4_start] if sec4_start else full_text[sec3_start:]
            for pat in [r'衍生品', r'套期']:
                m = re.search(pat, mda_text)
                if m:
                    targets["module2"] = sec3_start + m.start()
                    break

    # Module 3, 4, 5: 第十节（财务报告）+ 前置区域
    if sec10_start is not None:
        # 搜索起点：优先从第九节开始，覆盖整个财务报表区域
        sec9_start = boundaries.get("第九节")
        fin_search_start = sec9_start if (sec9_start is not None and sec9_start < sec10_start) else sec10_start

        fin_text = full_text[fin_search_start:]

        # Module 3: 外币货币性项目 - 取最后一个非引用匹配
        for pattern in [r'\d+[、，]\s*外币货币性项目', r'外币货币性项目']:
            matches = list(re.finditer(pattern, fin_text))
            if matches:
                for m in reversed(matches):
                    abs_pos = fin_search_start + m.start()
                    before = full_text[max(0, abs_pos - 60):abs_pos]
                    if not _check_xref(abs_pos, before, m.group()):
                        targets["module3"] = abs_pos
                        break
                if "module3" not in targets:
                    targets["module3"] = fin_search_start + matches[0].start()
                break

        # Module 4: 记账本位币 - 取第一个实质匹配（不含模板文字）
        for pattern in [r'(?:4[、，)]|（4）|四[、，])\s*记账本位币', r'记账本位币']:
            matches = list(re.finditer(pattern, fin_text))
            if matches:
                for m in matches:
                    ctx_after = fin_text[m.start():m.start() + 200]
                    if re.search(r'发生变化的还应披露原因', ctx_after[:80]):
                        continue
                    targets["module4"] = fin_search_start + m.start()
                    break
                if "module4" in targets:
                    break

        # Module 5: 与金融工具相关的风险 - 取最后一个非引用匹配
        for pattern in [r'与金融工具相关的风险', r'金融工具.*?风险']:
            matches = list(re.finditer(pattern, fin_text))
            if matches:
                for m in reversed(matches):
                    abs_pos = fin_search_start + m.start()
                    ctx_after = fin_text[m.start():m.start() + 180]
                    # 跳过交叉引用（详见/参见 + 之说明）
                    if re.search(r'(?:详见|参见|见.{0,10}附注).{0,50}与金融工具相关的风险', ctx_after):
                        continue
                    # 跳过"XX、与金融工具相关的风险(一)X"XXX"之说明"这类交叉引用
                    if re.search(r'与金融工具相关的风险.{0,20}之说明', ctx_after):
                        if not re.search(r'(?:1、金融工具|（一）金融工具|一、金融工具)', ctx_after[:100]):
                            continue
                    before = full_text[max(0, abs_pos - 60):abs_pos]
                    if not _check_xref(abs_pos, before, m.group()):
                        targets["module5"] = abs_pos
                        break
                if "module5" not in targets and matches:
                    targets["module5"] = fin_search_start + matches[0].start()
                break

    return targets


if __name__ == "__main__":
    from pdf_reader import load_or_extract
    from mapper import extract_company_list, build_pdf_map

    companies = extract_company_list()
    mapping = build_pdf_map(companies)

    test_companies = ["劲仔食品", "阳光电源", "大洋生物", "大禹节水", "舒泰神"]
    for comp in mapping:
        if comp["name"] in test_companies:
            print(f"\n{'='*60}")
            print(f"测试: {comp['code']} {comp['name']}")
            text = load_or_extract(comp["pdf_path"])
            boundaries = find_section_boundaries(text)
            print(f"找到 {len(boundaries)} 个章节边界")
            for name in sorted(boundaries.keys()):
                print(f"  {name}: 位置 {boundaries[name]:,}")
            targets = locate_target_subsections(text)
            print(f"找到 {len(targets)} 个目标子节: {list(targets.keys())}")
