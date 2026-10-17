"""
企业代码 → PDF文件映射（v2：正确处理代码格式）
"""
import os
import re
import zipfile
from config import PDF_DIR, COMPANY_LIST_DOCX, REPORT_YEAR


def extract_company_list(docx_path=COMPANY_LIST_DOCX):
    """从docx文件中提取50家企业代码和简称"""
    with zipfile.ZipFile(docx_path) as z:
        xml = z.read("word/document.xml").decode("utf-8")

    cleaned = re.sub(r"<[^>]+>", "\n", xml)
    lines = [l.strip() for l in cleaned.split("\n") if l.strip()]

    companies = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # 跳过表头
        if line in ("代码", "上市公司简称"):
            i += 1
            continue

        # 跳过"00"前缀行（它后面跟的是4位短代码+公司名）
        if line == "00":
            i += 1
            if i + 1 < len(lines):
                short_code = lines[i]  # e.g. "3000"
                name = lines[i + 1]   # e.g. "劲仔食品"
                if re.search(r"[一-鿿]", name):
                    companies.append({
                        "code": short_code,
                        "name": name,
                        "display_code": short_code,
                    })
                    i += 2
                    continue
            i += 1
            continue

        # 6位代码
        if re.match(r"^\d{6}$", line):
            if i + 1 < len(lines) and re.search(r"[一-鿿]", lines[i + 1]):
                name = lines[i + 1]
                companies.append({
                    "code": line,
                    "name": name,
                    "display_code": line,
                })
                i += 2
                continue

        i += 1

    return companies


def build_pdf_map(companies, pdf_dir=PDF_DIR):
    """建立企业代码到PDF文件路径的映射"""
    all_files = os.listdir(pdf_dir)

    mapping = []
    for comp in companies:
        code = comp["code"]
        name = comp["name"]
        matched_path = None

        # 策略1: 精确匹配公司名
        for f in all_files:
            if f.endswith(".pdf") and name in f:
                matched_path = f
                break

        # 策略2: 用代码匹配
        if not matched_path:
            for f in all_files:
                if f.endswith(".pdf") and code in f:
                    matched_path = f
                    break

        # 策略3: 模糊匹配
        if not matched_path:
            for f in all_files:
                if f.endswith(".pdf"):
                    # 去掉数字前缀和年份后缀
                    clean = re.sub(r"^\d+", "", f)
                    clean = re.sub(r"[：:]\s*\d{4}\s*年年度报告\.pdf$", "", clean)
                    clean = clean.replace("：2023年年度报告.pdf", "").replace(".pdf", "")
                    if name[:2] in clean or clean[:2] in name:
                        matched_path = f
                        break

        if matched_path:
            mapping.append({
                "code": code,
                "name": name,
                "pdf_path": os.path.join(pdf_dir, matched_path),
            })
        else:
            print(f"  [WARN] 未找到匹配PDF: {code} {name}")

    return mapping


if __name__ == "__main__":
    companies = extract_company_list()
    print(f"提取到 {len(companies)} 家企业")
    for c in companies[:5]:
        print(f"  {c['code']} {c['name']}")
    print("  ...")
    for c in companies[-3:]:
        print(f"  {c['code']} {c['name']}")

    mapping = build_pdf_map(companies)
    print(f"\n映射成功 {len(mapping)}/{len(companies)} 个PDF")
    for m in mapping[:5]:
        print(f"  {m['code']} {m['name']} → {os.path.basename(m['pdf_path'])}")
