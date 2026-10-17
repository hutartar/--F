"""
PDF文本全量提取与缓存
"""
import os
import pdfplumber
from config import CACHE_DIR, PDF_DIR


def ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def extract_full_text(pdf_path):
    """提取PDF全部文本，返回字符串"""
    with pdfplumber.open(pdf_path) as pdf:
        texts = []
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                texts.append(t)
        return "\n".join(texts)


def extract_with_page_markers(pdf_path):
    """提取PDF文本，每页以[PAGE_N]标记"""
    with pdfplumber.open(pdf_path) as pdf:
        texts = []
        for i, page in enumerate(pdf.pages):
            t = page.extract_text()
            if t:
                texts.append(f"[PAGE_{i + 1}]\n{t}")
        return "\n".join(texts)


def get_cache_path(pdf_path):
    """获取缓存txt路径"""
    basename = os.path.splitext(os.path.basename(pdf_path))[0]
    safe_name = basename.replace("：", "_").replace(":", "_")
    return os.path.join(CACHE_DIR, f"{safe_name}.txt")


def load_or_extract(pdf_path, use_cache=True):
    """加载缓存文本，若不存在则从PDF提取并缓存"""
    ensure_cache_dir()
    cache_path = get_cache_path(pdf_path)

    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()

    text = extract_with_page_markers(pdf_path)
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def batch_extract(pdf_paths, use_cache=True):
    """批量提取，返回 {pdf_path: full_text} 字典"""
    results = {}
    for i, path in enumerate(pdf_paths):
        name = os.path.basename(path)[:30]
        try:
            text = load_or_extract(path, use_cache=use_cache)
            results[path] = text
            if (i + 1) % 10 == 0:
                print(f"  已处理 {i + 1}/{len(pdf_paths)} ...")
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            results[path] = ""
    return results


if __name__ == "__main__":
    # 测试：提取所有PDF
    from mapper import extract_company_list, build_pdf_map

    companies = extract_company_list()
    mapping = build_pdf_map(companies)
    pdf_paths = [m["pdf_path"] for m in mapping]

    print(f"开始提取 {len(pdf_paths)} 份PDF文本...")
    results = batch_extract(pdf_paths, use_cache=True)

    success = sum(1 for v in results.values() if v)
    print(f"\n完成: {success}/{len(results)} 成功")
    if success < len(results):
        for path, text in results.items():
            if not text:
                print(f"  失败: {os.path.basename(path)}")
