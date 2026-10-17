"""
5个模块的专用文本提取函数（v3：彻底重写，基于指南精确匹配）
核心改进：
- M1：在十一、公司未来发展的展望中找所有汇率/外汇段落，包含父标题和序号
- M2：多轮检索+跟随交叉引用+结构化提取子部分
- M3：精准提取币种名称
- M4：正确处理三种情况（不适用→搜附近/无披露留空/适用→提取）
- M5：完整提取汇率风险段落（含父标题、其他价格风险）
"""
import re
from config import MODULES

# ============ 通用工具函数 ============

# 子节标题正则 — 匹配各种编号格式的行首标题
HEADING_PATTERN = re.compile(
    r'(?:^|\n)\s*'
    r'(?:'
    r'(?:（[一二三四五六七八九十\d]+）)'      # （一）（1）（三）
    r'|'
    r'(?:[一二三四五六七八九十]+[、，])'       # 一、 二，
    r'|'
    r'(?:\d{1,2}[、，)）])'                   # 1、 2） 3，
    r'|'
    r'(?:[（(]\d{1,2}[)）])'                  # (1) (2)
    r')'
    r'\s*\S+',
    re.MULTILINE
)

SKIP_PATTERNS = [
    r'四[、，]\s*汇率变动对现金及现金等价物的影响',
    r'外币业务和外币报表折算',
    r'外币财务报表折算',
    r'汇率变动对现金及现金等价物的影响',
]

CURRENCY_NAMES = [
    '印度卢比', '新加坡币', '越南盾', '澳元', '英镑', '加元',
    '卢比', '福林', '先令', '韩元', '里拉', '美元', '欧元',
    '港币', '日元', '瑞士法郎', '瑞典克朗', '澳门币', '新西兰元',
    '新台币', '澳门元', '新加坡元', '菲律宾比索', '马来西亚林吉特',
    '印尼盾', '泰铢', '俄罗斯卢布', '巴西雷亚尔', '墨西哥比索',
    '南非兰特', '迪拉姆', '坚戈',
]
CURRENCY_RE = re.compile('|'.join(sorted(CURRENCY_NAMES, key=len, reverse=True)))


def _should_skip(text):
    for pat in SKIP_PATTERNS:
        if re.search(pat, text):
            return True
    return False


def _clean_text(text):
    """通用文本清洁"""
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        s = line.strip()
        if not s:
            cleaned.append('')
            continue
        # 跳过纯数字行
        if re.match(r'^[\d\s,，.￥$€%\-+\(\)（）]+$', s):
            continue
        # 跳过页眉页脚
        if re.search(r'(?:股份有限公司|年度报告全文|年年度报告)', s) and len(s) < 60:
            continue
        # 跳过明显表格标头
        if re.match(r'^(?:项\s*目|单位[：:]|其中[：:])\s', s) and re.search(r'[\d,，.]{4,}', s):
            continue
        cleaned.append(s)
    # 合并多余空行
    result = []
    blank_count = 0
    for line in cleaned:
        if line == '':
            blank_count += 1
            if blank_count <= 2:
                result.append(line)
        else:
            blank_count = 0
            result.append(line)
    return '\n'.join(result).strip()


def _find_headings_in_range(text):
    """找出文本范围内所有子标题的位置和内容 [(pos, text), ...]"""
    headings = []
    for m in HEADING_PATTERN.finditer(text):
        h = m.group().strip()
        if len(h) >= 3 and len(h) <= 80:
            headings.append((m.start(), h))
    return headings


def _extract_paragraphs_with_context(text, keywords, window_before=100):
    """
    从文本中提取包含关键词的段落，并尝试包含其父标题。
    返回按原文顺序的段落列表。
    """
    paragraphs = re.split(r'\n\s*\n', text)
    result_paras = []
    last_heading = ""

    for i, para in enumerate(paragraphs):
        para_s = para.strip()
        if not para_s:
            continue

        # 检测是否为标题行
        is_heading = bool(HEADING_PATTERN.match(para_s))

        if is_heading and len(para_s) < 80:
            last_heading = para_s
            continue

        # 检查是否包含目标关键词
        if any(kw in para_s for kw in keywords):
            if _should_skip(para_s):
                continue
            # 如果上文有标题且不在当前段落中，添加标题前缀
            if last_heading and last_heading not in para_s:
                result_paras.append(last_heading + '\n' + para_s)
            else:
                result_paras.append(para_s)

    return result_paras


# ============ 模块1：管理层讨论与分析 → 公司未来发展的展望 ============

def extract_module1(full_text, targets, module_cfg):
    """
    在十一、公司未来发展的展望部分提取所有汇率/外汇相关段落。
    包含父标题和序号。两轮检索：汇率→外汇。
    """
    if "module1" not in targets:
        return ""

    pos = targets["module1"]
    window = module_cfg.get("context_window", 2500)
    ctx = full_text[pos:min(len(full_text), pos + window)]

    # 找到子标题，确定展望部分的实际边界
    headings = _find_headings_in_range(ctx)
    outlook_start = 0  # 在ctx中的起始位置

    # 找到"公司未来发展的展望"或"十一、"开头的标题
    for hpos, htext in headings:
        if re.search(r'(?:未来.*展望|发展.*展望|公司.*展望)', htext):
            outlook_start = hpos
            break

    # 如果没有子标题结构，使用整个ctx
    work_text = ctx[outlook_start:]

    # 第一轮：汇率
    paras_1 = _extract_paragraphs_with_context(work_text, ["汇率"])
    # 第二轮：外汇（去重）
    paras_2 = _extract_paragraphs_with_context(work_text, ["外汇"])

    # 合并去重，保持顺序
    seen = set()
    all_paras = []
    for p in paras_1 + paras_2:
        key = p[:80]
        if key not in seen:
            seen.add(key)
            all_paras.append(p)

    if not all_paras:
        # 回退：在展望部分直接搜索汇率/外汇段落
        for m in re.finditer(r'(?:汇率|外汇)', work_text):
            start = max(0, m.start() - 200)
            end = min(len(work_text), m.end() + 500)
            # 扩展到段落边界
            para_start = work_text.rfind('\n\n', start, m.start())
            para_end = work_text.find('\n\n', m.end(), end)
            if para_start < 0:
                para_start = max(0, work_text.rfind('\n', 0, m.start()))
            if para_end < 0:
                para_end = min(len(work_text), m.end() + 500)
            para = work_text[max(0, para_start):para_end].strip()
            if para and not _should_skip(para):
                key = para[:80]
                if key not in seen:
                    seen.add(key)
                    all_paras.append(para)
            if len(all_paras) >= 10:
                break

    result = '\n\n'.join(all_paras)
    return _clean_text(result)[:2000]


# ============ 模块2：衍生品投资情况 ============

def extract_module2(full_text, targets, module_cfg):
    """
    衍生品投资情况提取（最复杂模块）。
    步骤：
    1. 找到衍生品投资情况节，判断适用/不适用
    2. 如适用：提取投资类型 + 四类说明文字
    3. 多轮检索：汇率→外汇→远期→期权→掉期→互换
    4. 跟随交叉引用至财务报告
    """
    if "module2" not in targets:
        return ""

    pos = targets["module2"]
    window = module_cfg.get("context_window", 5000)
    ctx = full_text[max(0, pos - 200):min(len(full_text), pos + window)]

    # 检查是否存在衍生品投资
    # 情况1：不适用
    if re.search(r'(?:公司|本集团).*?报告期.*?不.*?存在.*?衍生品投资', ctx):
        if '以投机为目的' not in ctx[max(0, ctx.find('不存在衍生品投资')-200):]:
            return "公司报告期不存在衍生品投资。"

    # 情况2：不适用标记
    if re.search(r'衍生品投资情况\s*\n\s*[□☐]\s*适用\s*[☑☒✓]\s*不适用', ctx):
        return "公司报告期不存在衍生品投资。"

    # 情况3：适用 → 提取详细信息
    result_parts = []

    # 3a. 检查套期保值适用性
    hedging_match = re.search(
        r'(?:1\s*[）)]\s*)?报告期内以套期保值为目的的衍生品投资\s*\n?\s*[☑☒✓]\s*适用',
        ctx
    )
    if hedging_match:
        result_parts.append("1）报告期内以套期保值为目的的衍生品投资：适用")

    # 3b. 提取衍生品投资类型（从表格或描述中）
    type_patterns = [
        r'衍生品投资类型[：:]\s*([^\n]{5,100})',
        r'衍生品投资类型\s*\n\s*([^\n]{3,100})',
    ]
    for pat in type_patterns:
        m = re.search(pat, ctx)
        if m:
            inv_type = m.group(1).strip()
            if re.search(r'(?:远期|外汇|汇率|期权|掉期|互换|套期)', inv_type):
                result_parts.append(f"衍生品投资类型：{inv_type}")
            break

    # 3c. 多轮检索提取段落
    search_passes = module_cfg.get("search_passes", [
        ["汇率", "外汇"],
        ["远期", "期权"],
        ["掉期", "互换"],
    ])

    para_splits = re.split(r'\n\s*\n', ctx)
    matched_paras = set()

    for pass_kws in search_passes:
        for para in para_splits:
            para_s = para.strip()
            if not para_s or len(para_s) < 10:
                continue
            if _should_skip(para_s):
                continue
            # 跳过表格数据
            digit_ratio = len(re.findall(r'[\d\s,，.￥$€%]', para_s)) / max(1, len(para_s))
            if digit_ratio > 0.55 and len(para_s) > 30:
                continue
            # 跳过商品期货
            if re.search(r'(?:铜|铝|热卷|聚氯乙烯|不锈钢|螺纹钢|铁矿石|焦炭|PTA)\s*(?:的)?\s*(?:期货|期权)', para_s):
                continue
            # 跳过股票期权
            if re.search(r'股票期权|股权激励.*期权|限制性股票', para_s):
                continue
            # 跳过多列表格碎片
            space_frags = len(re.findall(r'\S{2,5}\s{4,}\S{2,5}', para_s))
            if space_frags >= 3:
                continue

            for kw in pass_kws:
                if kw in para_s:
                    matched_paras.add(para_s)
                    break

    # 按原文顺序排列
    ordered_paras = sorted(matched_paras, key=lambda p: ctx.find(p))

    # 3d. 提取四类结构化子部分
    sub_parts = [
        (r'报告期实际损益情况.*?说明', '报告期实际损益情况的说明'),
        (r'套期保值效果.*?说明', '套期保值效果的说明'),
        (r'报告期衍生品持仓.*?风险分析.*?控制措施.*?说明', '报告期衍生品持仓的风险分析及控制措施说明'),
        (r'独立董事.*?(?:意见|专项意见)', '独立董事对公司衍生品投资及风险控制情况的专项意见'),
    ]

    for part_pat, part_label in sub_parts:
        for para in ctx.split('\n\n'):
            if re.search(part_pat, para):
                cleaned = _clean_text(para)
                if cleaned and len(cleaned) > 15:
                    result_parts.append(f"【{part_label}】\n{cleaned}")
                break

    # 合并结果
    if result_parts:
        # 前两行（适用性+类型）+ 详细段落
        header_parts = result_parts[:2]
        body_paras = ordered_paras[:6]  # 最多6段
        # 确保结构化子部分在正文中
        detail_parts = result_parts[2:]

        final = header_parts + body_paras + detail_parts
        # 去重
        seen = set()
        unique = []
        for p in final:
            key = p[:80]
            if key not in seen:
                seen.add(key)
                unique.append(p)
        result = '\n\n'.join(unique)
        return _clean_text(result)[:3000]

    # 3e. 回退：仅在全文搜索衍生品相关内容
    if not ordered_paras:
        fallback = _full_text_fallback(full_text, pos)
        if fallback:
            return _clean_text(fallback)[:2000]
        return ""

    result = '\n\n'.join(ordered_paras)
    return _clean_text(result)[:3000]


def _full_text_fallback(full_text, pos):
    """全文回退搜索外汇衍生品相关内容"""
    # 在财务报告部分搜索
    fin_start = max(0, full_text.find('第十节', max(0, pos - 1000)))
    if fin_start < 0:
        fin_start = int(len(full_text) * 0.5)
    search_text = full_text[fin_start:]

    keywords = ["衍生金融", "套期保值", "远期结售汇", "外汇合约", "外汇远期",
                "远期外汇", "期权合约", "货币互换", "掉期", "远期结汇"]
    found = []
    seen = set()

    for kw in keywords:
        for m in re.finditer(kw, search_text):
            start = max(0, m.start() - 300)
            end = min(len(search_text), m.end() + 500)
            para = search_text[start:end].strip()
            if len(para) < 20 or para in seen:
                continue
            if not re.search(r'(?:汇率|外汇|美元|欧元|港币|远期)', para):
                continue
            digit_ratio = len(re.findall(r'[\d\s,，.￥$€%]', para)) / max(1, len(para))
            if digit_ratio > 0.5 and len(para) > 50:
                continue
            seen.add(para)
            found.append(para)
            if len(found) >= 8:
                break
        if len(found) >= 8:
            break

    return '\n\n'.join(found)


# ============ 模块3：外币货币性项目 ============

def extract_module3(full_text, targets, module_cfg):
    """
    提取外币货币性项目表格中的唯一币种名称。
    遵循指南：只抄录币种名称，不重复记录。
    """
    if "module3" not in targets:
        # 回退：使用已检测的章节边界来定位财务报表区域
        from section_locator import find_section_boundaries
        boundaries = find_section_boundaries(full_text)
        sec10_start = boundaries.get("第十节")
        if sec10_start:
            fin_start = sec10_start
            # 也搜索第十节之前的区域（结构异常的PDF）
            for prepend in [0, 10000, 20000, 30000]:
                search_start = max(0, fin_start - prepend)
                search_area = full_text[search_start:min(len(full_text), fin_start + 50000)]
                for pat in [r'\d+[、，]\s*外币货币性项目', r'外币货币性项目']:
                    m = re.search(pat, search_area)
                    if m:
                        pos = search_start + m.start()
                        targets["module3"] = pos
                        break
                if "module3" in targets:
                    break
            # 如果实在找不到，检查是否整个报告都没有外币相关内容
            if "module3" not in targets:
                fin_full = full_text[fin_start:]
                has_foreign_currency = bool(
                    re.search(r'(?:美元|欧元|港币|日元|韩元|英镑|澳元|越南盾)', fin_full[:100000]) or
                    re.search(r'外币货币性项目', fin_full[:100000])
                )
                if not has_foreign_currency:
                    return "无"
                return ""
        else:
            return ""

    pos = targets["module3"]
    window = module_cfg.get("context_window", 1500)
    ctx = full_text[max(0, pos - 50):min(len(full_text), pos + window)]

    if _should_skip(ctx[:300]):
        return ""

    # 提取所有币种名称（去重，按首次出现顺序）
    currency_seen = set()
    currency_order = []

    for m in CURRENCY_RE.finditer(ctx):
        c = m.group()
        if c not in currency_seen:
            currency_seen.add(c)
            currency_order.append(c)

    if not currency_order:
        # 扩大搜索范围
        extended = full_text[pos:min(len(full_text), pos + window * 2)]
        for m in CURRENCY_RE.finditer(extended):
            c = m.group()
            if c not in currency_seen:
                currency_seen.add(c)
                currency_order.append(c)

    if not currency_order:
        # 检查是否明确标注"无"（如"外币货币性项目：无"）
        if re.search(r'(?:外币货币性项目|境外经营)\S{0,10}(?:：|:)\s*无', ctx):
            return "无"
        # 如果窗口内没有任何币种且标题附近标注了不适用，返回"无"
        if re.search(r'外币货币性项目.*?(?:□|☐).*?适用.*?(?:☑|☒|✓).*?不适用', ctx, re.DOTALL):
            return "无"

    return '、'.join(currency_order) if currency_order else ""


# ============ 模块4：记账本位币/境外经营实体 ============

def extract_module4(full_text, targets, module_cfg):
    """
    记账本位币 / 境外经营实体说明。
    三种情况：
    1. 境外经营实体说明选"不适用" → 在外币货币性项目附近搜子公司记账本位币
    2. 无任何披露 → 留空
    3. 选"适用" → 提取境外经营实体完整说明
    4. 回退：提取"4、记账本位币"段落内容
    """
    if "module4" not in targets:
        return ""

    pos = targets["module4"]
    window = module_cfg.get("context_window", 2000)
    # 从目标位置向后搜索，精确提取段落
    search_area = full_text[max(0, pos - 100):min(len(full_text), pos + window)]

    # Step 1: 先搜索境外经营实体说明
    entity_matches = list(re.finditer(r'境外经营实体', search_area))
    if entity_matches:
        entity_start = entity_matches[0].start()
        entity_ctx = search_area[entity_start:entity_start + 1000]

        # 过滤掉模板/指南文字
        if re.search(r'包括对于重要的境外经营实体', entity_ctx[:100]):
            actual_start = entity_ctx.find('不适用')
            if actual_start < 0:
                actual_start = entity_ctx.find('适用')
            if actual_start > 0:
                entity_ctx = entity_ctx[actual_start:]

        # Case 1: 不适用
        if re.search(r'不适用', entity_ctx[:200]):
            if "module3" in targets:
                m3_pos = targets["module3"]
                extended = full_text[m3_pos:min(len(full_text), m3_pos + 3000)]
                matches = list(re.finditer(
                    r'(?:记账本位币|确定\s*\S{0,10}\s*(?:币|元|先令|美元|欧元)\s*为.*?本位币)[^。]{10,300}[。]',
                    extended
                ))
                if matches:
                    results = []
                    for m in matches[-4:]:
                        text_s = m.group().strip()
                        if not re.search(r'包括对于重要的|发生变化的还应', text_s):
                            results.append(text_s)
                    if results:
                        return '；\n'.join(results)[:800]
            return ""

        # Case 3: 适用 → 提取实质内容
        cleaned_lines = []
        for line in entity_ctx.split('\n'):
            s = line.strip()
            if not s:
                continue
            if re.search(r'包括对于重要的境外经营实体|应披露其境外主要经营地|发生变化的还应披露原因', s):
                continue
            if re.search(r'股份有限公司|年度报告全文', s) and len(s) < 60:
                continue
            cleaned_lines.append(s)
        if cleaned_lines:
            return '\n'.join(cleaned_lines)[:800]

    # Step 2: 精确提取"4、记账本位币"段落（从该标题到下一个编号标题）
    for heading_pat in [r'4[、，)]\s*记账本位币', r'（4）\s*记账本位币', r'四[、，]\s*记账本位币', r'(?<=\n)记账本位币']:
        m = re.search(heading_pat, search_area)
        if m:
            # 向前找到行首（不超过200字符，避免跨越段落块）
            para_start = m.start()
            prev_nl = search_area.rfind('\n', max(0, m.start() - 200), m.start())
            if prev_nl >= 0 and m.start() - prev_nl < 200:
                para_start = prev_nl + 1

            # 向后找到段落结束（下一个编号标题）
            after = search_area[m.end():]
            next_break = None
            for stop_pat in [
                r'\n\s*\d+[、，)]',
                r'\n\s*(?:[五六七八九十]+)[、，]',
                r'\n\s*（[五六七八九十\d]+）',
            ]:
                sm = re.search(stop_pat, after)
                if sm:
                    candidate = m.end() + sm.start()
                    if next_break is None or candidate < next_break:
                        next_break = candidate

            # 也检查段落空白
            para_end = search_area.find('\n\n', m.end())
            if para_end >= 0 and (next_break is None or para_end < next_break):
                next_break = para_end

            if next_break:
                para = search_area[para_start:next_break].strip()
            else:
                para = search_area[para_start:m.end() + 300].strip()

            # 过滤模板文字
            if re.search(r'包括对于重要的境外经营实体|应披露其境外主要经营地|发生变化的还应披露原因', para):
                continue
            if len(para) > 12:
                cleaned = _clean_text(para)
                if len(cleaned) > 10:
                    return cleaned[:800]

    # Step 3: 回退——在搜索区域中找记账本位币段落
    for para in search_area.split('\n\n'):
        if '记账本位币' in para and len(para.strip()) > 30:
            if re.search(r'包括对于重要的境外经营实体', para):
                continue
            cleaned = _clean_text(para.strip())
            if len(cleaned) > 20:
                return cleaned[:600]

    return ""


# ============ 模块5：与金融工具相关的风险 → 汇率风险 ============

def extract_module5(full_text, targets, module_cfg):
    """
    提取"与金融工具相关的风险"中汇率风险相关段落。
    策略：
    1. 定位"与金融工具相关的风险"大节
    2. 找到"汇率风险"或"外汇风险"子节
    3. 向前包含父标题（如"（1）市场风险"）
    4. 向后提取直到遇到利率/信用/流动性风险或下一大节
    5. 包含"其他价格风险"段落
    """
    if "module5" not in targets:
        return ""

    pos = targets["module5"]
    window = module_cfg.get("context_window", 6000)
    ctx = full_text[pos:min(len(full_text), pos + window)]

    # Step 1: 找到"汇率风险"或"外汇风险"子节起始位置
    fx_start = None
    for pat in [
        r'\n\s*\d+[、，)）]\s*汇率风险',
        r'\n\s*\d+[、，)）]\s*外汇风险',
        r'\n\s*\d+[、，)）]\s*外汇汇率风险',
        r'\n\s*（\d+）\s*汇率风险',
        r'\n\s*（\d+）\s*外汇风险',
    ]:
        m = re.search(pat, ctx)
        if m:
            if not re.search(r'利率|信用|流动', m.group()):
                fx_start = m.start()
                break

    if fx_start is None:
        m = re.search(r'(?:汇率风险|外汇风险)', ctx)
        if m:
            fx_start = m.start()

    if fx_start is None:
        return ""

    # Step 2: 确定提取起点 — 向上找父标题
    extract_start = fx_start
    lookback = ctx[max(0, fx_start - 500):fx_start]

    # 找"市场风险"父标题
    for parent_pat in [
        r'\n\s*（[一二三四五六七八九\d]+）\s*市场风险',
        r'\n\s*\d+[、，)）]\s*市场风险',
        r'\n\s*（[一二三四五六七八九\d]+）\s*外汇风险',
        r'\n\s*\d+[、，)）]\s*外汇风险',
    ]:
        parent_m = re.search(parent_pat, lookback)
        if parent_m:
            extract_start = fx_start - (len(lookback) - parent_m.start())
            break

    # Step 3: 确定提取终点
    result_text = ctx[extract_start:]

    # 停止条件：遇到利率/信用/流动性风险子节
    stop_at = len(result_text)
    stop_patterns = [
        r'\n[ \t]*\d+[、，)）.\-]*\s*利率风险',
        r'\n[ \t]*\d+[、，)）.\-]*\s*信用风险',
        r'\n[ \t]*\d+[、，)）.\-]*\s*流动性风险',
        r'\n[ \t]*（\d+）[ \t]*利率风险',
        r'\n[ \t]*（\d+）[ \t]*信用风险',
        r'\n[ \t]*（\d+）[ \t]*流动性风险',
        r'\n[ \t]*(?:十三|十四|十五)[、，]',  # 下一大附注
        r'\n[ \t]*2[、，][ \t]*套期',         # 套期会计部分
        r'\n[ \t]*[―\-]+[ \t]*(?:利率风险|信用风险)',  # 纯分隔符形式
    ]

    for stop_pat in stop_patterns:
        stop_m = re.search(stop_pat, result_text[60:])
        if stop_m:
            candidate = 60 + stop_m.start()
            matched_text = stop_m.group()
            if not re.search(r'(?:汇率|外汇)', matched_text):
                if candidate < stop_at:
                    stop_at = candidate

    result = result_text[:stop_at].strip()

    # Step 4: 额外提取"其他价格风险"段落（如有）
    remaining = result_text[stop_at:]
    other_price = re.search(
        r'其他价格风险.{50,600}?(?=\n\s*\d+[、，)）])',
        remaining
    )
    if other_price:
        other_text = other_price.group().strip()
        if not _should_skip(other_text):
            result = result + '\n\n' + other_text

    # Step 5: 清洁
    result = _clean_module5(result)
    return result[:3000]


def _clean_module5(text):
    """清洁模块5文本：去表格残片、页眉页码、表格数据行"""
    lines = text.split('\n')
    cleaned = []
    in_table = False
    for line in lines:
        s = line.strip()
        if not s:
            if not in_table:
                cleaned.append('')
            continue
        # 检测表格起始（"项目 期末 期初"等标头模式）
        if re.match(r'^(?:项\s*目|币\s*种)\s+(?:期末|期初|外币|人民币|账面)', s):
            in_table = True
            continue
        # 表格数据行特征：短文本+大空格+数字（典型的3列表格数据）
        if re.match(r'^[\w一-鿿\-–]{1,20}\s{3,}\d[\d,.]*', s):
            continue
        # 表格合计行
        if re.match(r'^(?:合计|小计)\s+\d', s):
            continue
        # 货币资金表格行（如"货币资金–蒙图 35.83 17.87"）
        if re.match(r'^(?:货币资金|应收账款|应付账款|其他应收|其他应付)[–\-]\S+\s+\d', s):
            continue
        if in_table:
            # 检测是否从表格中恢复（遇到带编号的标题）
            if re.match(r'^(?:[一二三四五六七八九十]+[、，]|\d+[、，)）]|（\d+）)', s):
                in_table = False
            else:
                continue
        # 跳过页眉页脚
        if re.search(r'(?:股份有限公司|年度报告|年年度报告)', s) and len(s) < 80:
            continue
        # 跳过纯数字/符号行
        if re.match(r'^[\d\s,，.￥$€%\-+\(\)（）]+$', s):
            continue
        # 跳过数据表格片段（数字密度>60%且长度>30）
        digit_ratio = len(re.findall(r'[\d\s,，.￥$€%\-+\s]', s)) / max(1, len(s))
        if digit_ratio > 0.6 and len(s) > 30:
            continue
        cleaned.append(s)

    # 合并多余空行
    result = []
    blank_count = 0
    for line in cleaned:
        if line == '':
            blank_count += 1
            if blank_count <= 2:
                result.append(line)
        else:
            blank_count = 0
            result.append(line)
    return '\n'.join(result).strip()


# ============ 制度信息提取 ============

def extract_policy_info(full_text, targets):
    """从年报中提取衍生品/外汇风险管理相关制度名称"""
    from config import POLICY_BLACKLIST_PHRASES
    policies = set()

    # 搜索范围：衍生品投资区域 + 全文
    search_areas = []
    if "module2" in targets:
        m2_pos = targets["module2"]
        search_areas.append(full_text[max(0, m2_pos - 200):min(len(full_text), m2_pos + 5000)])

    search_areas.append(full_text)

    # 风险管理制度关键词（收紧：必须与外汇/衍生品直接相关）
    RISK_POLICY_KW = [
        '衍生品', '套期保值', '外汇', '远期结售汇', '金融衍生',
        '期货套期', '远期外汇', '外汇套期', '汇率风险',
    ]

    for area in search_areas:
        for pat in [
            r'《[^》]{4,35}(?:管理办法|管理制度|风险控制|内部控制制度|实施细则|业务制度|管理细则|管理规定|交易制度|风控制度)[^》]*》',
        ]:
            for m in re.finditer(pat, area):
                p = m.group()
                # 长度检查
                if len(p) < 8 or len(p) > 38:
                    continue
                # 必须包含外汇/衍生品相关关键词
                if not any(kw in p for kw in RISK_POLICY_KW):
                    continue
                # 黑名单过滤：CSRC通用文档、审计报告、治理文件等
                if any(bad in p for bad in POLICY_BLACKLIST_PHRASES):
                    continue
                # 排除会计准则、法律法规
                if re.search(r'会计准则|审计准则|税法|公司法|证券法|金融工具确认|套期会计|金融工具列报|企业会计准则', p):
                    continue
                if re.search(r'第[一二三\d]+号', p):
                    continue
                # 排除股票/股权激励
                if re.search(r'股票|股权激励|限制性股票|员工持股|激励计划', p):
                    continue
                # 排除税务、财政政策
                if re.search(r'财政部|税务总局|所得税|增值税|优惠', p):
                    continue
                # 排除议案、决议
                if re.search(r'(?:议案|决议|会议|年度报告)', p):
                    continue
                # 排除空格过多的碎片
                if len(re.findall(r'\s{2,}', p)) >= 3:
                    continue
                # 汉字比例>65%
                chinese = len(re.findall(r'[一-鿿]', p))
                if chinese / len(p) < 0.60:
                    continue
                policies.add(p)

    return '; '.join(sorted(policies, key=len)[:5])


# ============ 主提取入口 ============

def extract_all_modules(full_text, targets):
    """执行所有5个模块的提取 + 制度信息"""
    results = {}

    results["module1"] = extract_module1(full_text, targets, MODULES["module1"])
    results["module2"] = extract_module2(full_text, targets, MODULES["module2"])
    results["module3"] = extract_module3(full_text, targets, MODULES["module3"])
    results["module4"] = extract_module4(full_text, targets, MODULES["module4"])
    results["module5"] = extract_module5(full_text, targets, MODULES["module5"])
    results["policy_info"] = extract_policy_info(full_text, targets)

    return results


if __name__ == "__main__":
    from pdf_reader import load_or_extract
    from text_cleaner import clean_full_text
    from section_locator import locate_target_subsections
    from mapper import extract_company_list, build_pdf_map

    companies = extract_company_list()
    mapping = build_pdf_map(companies)

    for test_name in ["劲仔食品", "阳光电源", "大洋生物", "方大集团"]:
        for comp in mapping:
            if comp["name"] == test_name:
                print(f"\n{'='*60}")
                print(f"{comp['code']} {comp['name']}")
                print(f"{'='*60}")

                raw = load_or_extract(comp["pdf_path"])
                text = clean_full_text(raw)
                targets = locate_target_subsections(text)
                print(f"Targets found: {list(targets.keys())}")
                results = extract_all_modules(text, targets)

                for mod_name in ["module1", "module2", "module3", "module4", "module5", "policy_info"]:
                    result = results.get(mod_name, "")
                    status = f"{len(result)} chars" if result else "EMPTY"
                    print(f"\n--- {mod_name} ({status}) ---")
                    if result:
                        print(result[:500].replace('\n', '\n    '))
                break
