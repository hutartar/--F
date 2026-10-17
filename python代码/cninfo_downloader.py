"""
巨潮资讯网年报PDF下载器（v1）：搜索→下载→缓存 一体化

数据源：巨潮资讯网 (www.cninfo.com.cn) — 中国证监会指定信息披露网站
接口特点：公开免费，无需登录即可搜索和下载年报PDF

API端点：
- 搜索: http://www.cninfo.com.cn/new/fulltextSearch/full
- 下载: http://static.cninfo.com.cn/{adjunctUrl}
"""
import os
import re
import json
import time
import hashlib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import requests


# ── 配置 ──
CNINFO_SEARCH_URL = "http://www.cninfo.com.cn/new/fulltextSearch/full"
CNINFO_DOWNLOAD_PREFIX = "http://static.cninfo.com.cn/"
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9",
}


@dataclass
class DownloadResult:
    company_code: str
    company_name: str
    year: str
    pdf_path: str          # 本地PDF路径
    pdf_url: str           # 下载URL
    file_size: int         # 文件大小（字节）
    downloaded: bool
    from_cache: bool
    error: str = ""


class CninfoDownloader:
    """
    巨潮资讯网年报下载器

    用法:
        downloader = CninfoDownloader(cache_dir="./pdf_cache")
        result = downloader.download("000001", "平安银行", "2023")
        print(result.pdf_path)  # ./pdf_cache/000001_平安银行_2023.pdf
    """

    def __init__(
        self,
        cache_dir: str = None,
        request_interval: float = 1.0,
        max_retries: int = 3,
    ):
        """
        Args:
            cache_dir: PDF缓存目录（None则使用config中的PDF_DIR）
            request_interval: 请求间隔（秒），避免被反爬
            max_retries: 下载失败最大重试次数
        """
        if cache_dir is None:
            from config import PDF_DIR
            cache_dir = PDF_DIR
        self.cache_dir = cache_dir
        self.request_interval = request_interval
        self.max_retries = max_retries
        os.makedirs(cache_dir, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update(REQUEST_HEADERS)
        self._last_request_time = 0

    def _rate_limit(self):
        """请求频率控制"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.request_interval:
            time.sleep(self.request_interval - elapsed)
        self._last_request_time = time.time()

    def _pdf_cache_path(self, code: str, name: str, year: str) -> str:
        """PDF缓存文件路径"""
        safe_name = re.sub(r'[\\/:*?"<>|]', '', name)
        filename = f"{code}_{safe_name}_{year}年年度报告.pdf"
        return os.path.join(self.cache_dir, filename)

    def search_annual_report(
        self, company_name: str, year: str
    ) -> Optional[dict]:
        """
        搜索公司年报

        Args:
            company_name: 公司简称（如"平安银行"）
            year: 年份（如"2023"）

        Returns:
            年报公告信息dict，包含title, adjunctUrl等；未找到返回None
        """
        self._rate_limit()

        search_key = f"{company_name}{year}年年度报告"
        params = {
            "searchkey": search_key,
            "sdate": "",
            "edate": "",
            "isfulltext": "false",
            "sortName": "pubdate",
            "sortType": "desc",
            "pageNum": 1,
            "type": "",
        }

        try:
            resp = self.session.get(
                CNINFO_SEARCH_URL,
                params=params,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  [WARN] 搜索失败 {company_name}: {e}")
            return None

        announcements = data.get("announcements", [])
        if not announcements:
            print(f"  [WARN] 未找到 {company_name} {year}年年报")
            return None

        # 筛选：必须是年度报告全文，排除摘要/补充/英文版/ESG等
        candidates = []  # 收集所有合格候选，按优先级排序

        for ann in announcements:
            title = ann.get("announcementTitle", "")
            # 清理HTML标签
            title_clean = re.sub(r"<[^>]+>", "", title)

            # 必须包含"年度报告"
            if "年度报告" not in title_clean:
                continue
            # 必须包含年份
            if year not in title_clean:
                continue

            # 黑名单过滤（各种变体都要排除）
            bad_keywords = [
                "摘要", "概要", "摘要版", "简明", "精简",
                "补充", "更正", "修订", "更新", "修正",
                "英文", "ESG", "环境", "社会责任",
                "半年", "季度", "一季度", "三季度", "四季度",
                "已取消", "取消", "作废",
                "审计报告", "内控", "内部控制", "鉴证",
                "独立董事", "股东大会", "董事会", "监事会",
                "利润分配", "分红", "股权激励", "限制性股票",
            ]
            if any(bad in title_clean for bad in bad_keywords):
                continue

            # 计算优先级：标题越接近"XXXX年年度报告"优先级越高
            priority = 0
            # 精确匹配年份+年度报告结尾，优先级最高
            if re.search(f"{year}年年度报告$", title_clean):
                priority = 100
            # 标题短且干净的，优先级次之
            if len(title_clean) < 25:
                priority += 10

            candidates.append({
                "title": title_clean,
                "adjunctUrl": ann.get("adjunctUrl", ""),
                "announcementId": ann.get("announcementId", ""),
                "publishDate": ann.get("publishDate", ""),
                "secCode": ann.get("secCode", ""),
                "secName": ann.get("secName", ""),
                "priority": priority,
            })

        # 按优先级排序，取最高优先级的
        if candidates:
            candidates.sort(key=lambda x: x["priority"], reverse=True)
            best = candidates[0]
            best.pop("priority")
            return best

        return None

    def download_pdf(
        self, ann_info: dict, code: str, name: str, year: str
    ) -> DownloadResult:
        """
        下载年报PDF到本地缓存

        Args:
            ann_info: search_annual_report返回的公告信息
            code: 股票代码
            name: 公司简称
            year: 年份

        Returns:
            DownloadResult
        """
        cache_path = self._pdf_cache_path(code, name, year)

        # 检查缓存
        if os.path.exists(cache_path) and os.path.getsize(cache_path) > 10000:
            return DownloadResult(
                company_code=code,
                company_name=name,
                year=year,
                pdf_path=cache_path,
                pdf_url="",
                file_size=os.path.getsize(cache_path),
                downloaded=False,
                from_cache=True,
            )

        # 构建下载URL
        adjunct_url = ann_info.get("adjunctUrl", "")
        if not adjunct_url:
            return DownloadResult(
                company_code=code, company_name=name, year=year,
                pdf_path="", pdf_url="", file_size=0,
                downloaded=False, from_cache=False,
                error="无PDF下载链接",
            )

        pdf_url = CNINFO_DOWNLOAD_PREFIX + adjunct_url

        # 下载（含重试）
        for attempt in range(self.max_retries):
            self._rate_limit()
            try:
                resp = self.session.get(pdf_url, timeout=60)
                resp.raise_for_status()

                # 检查是否是有效的PDF
                content = resp.content
                if len(content) < 500000:  # 正常年报>500KB，低于此值可能是摘要版
                    if attempt < self.max_retries - 1:
                        time.sleep(2)
                        continue
                    return DownloadResult(
                        company_code=code, company_name=name, year=year,
                        pdf_path="", pdf_url=pdf_url, file_size=len(content),
                        downloaded=False, from_cache=False,
                        error=f"文件过小({len(content)//1024}KB)，可能是摘要版而非完整年报",
                    )

                # 保存到缓存
                with open(cache_path, 'wb') as f:
                    f.write(content)

                return DownloadResult(
                    company_code=code,
                    company_name=name,
                    year=year,
                    pdf_path=cache_path,
                    pdf_url=pdf_url,
                    file_size=len(content),
                    downloaded=True,
                    from_cache=False,
                )

            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2)
                else:
                    return DownloadResult(
                        company_code=code, company_name=name, year=year,
                        pdf_path="", pdf_url=pdf_url, file_size=0,
                        downloaded=False, from_cache=False,
                        error=str(e),
                    )

        return DownloadResult(
            company_code=code, company_name=name, year=year,
            pdf_path="", pdf_url="", file_size=0,
            downloaded=False, from_cache=False,
            error="超过最大重试次数",
        )

    def download(
        self, code: str, name: str, year: str
    ) -> DownloadResult:
        """
        下载指定公司的年报（搜索+下载一体化）

        返回None时表示缓存中已存在
        """
        # 先检查缓存
        cache_path = self._pdf_cache_path(code, name, year)
        if os.path.exists(cache_path) and os.path.getsize(cache_path) > 10000:
            return DownloadResult(
                company_code=code, company_name=name, year=year,
                pdf_path=cache_path, pdf_url="",
                file_size=os.path.getsize(cache_path),
                downloaded=False, from_cache=True,
            )

        # 搜索
        ann_info = self.search_annual_report(name, year)
        if ann_info is None:
            return DownloadResult(
                company_code=code, company_name=name, year=year,
                pdf_path="", pdf_url="", file_size=0,
                downloaded=False, from_cache=False,
                error=f"未找到 {name} {year}年年报",
            )

        # 下载
        return self.download_pdf(ann_info, code, name, year)

    def batch_download(
        self, companies: List[dict], year: str
    ) -> List[DownloadResult]:
        """
        批量下载年报

        Args:
            companies: [{"code": "000001", "name": "平安银行"}, ...]
            year: 年份

        Returns:
            下载结果列表
        """
        results = []
        total = len(companies)

        for i, comp in enumerate(companies):
            code = comp.get("code", comp.get("display_code", ""))
            name = comp.get("name", "")

            print(f"  [{i+1}/{total}] {code} {name} ... ", end="", flush=True)

            result = self.download(code, name, year)

            if result.from_cache:
                print(f"缓存 (已存在)")
            elif result.downloaded:
                size_mb = result.file_size / 1024 / 1024
                print(f"下载完成 ({size_mb:.1f}MB)")
            else:
                print(f"失败: {result.error}")

            results.append(result)

        # 统计
        success = sum(1 for r in results if r.pdf_path)
        cached = sum(1 for r in results if r.from_cache)
        downloaded = sum(1 for r in results if r.downloaded)
        failed = total - success

        print(f"\n  批量下载完成: 成功{success}/失败{failed} "
              f"(缓存{cached}, 新下载{downloaded})")

        return results


if __name__ == "__main__":
    print("=== 巨潮资讯网年报下载器 测试 ===\n")

    downloader = CninfoDownloader(request_interval=1.0)

    # 测试搜索
    print("[1] 搜索测试: 劲仔食品 2023年年报")
    ann = downloader.search_annual_report("劲仔食品", "2023")
    if ann:
        print(f"  标题: {ann['title']}")
        print(f"  代码: {ann.get('secCode', 'N/A')}")
        print(f"  URL: {ann['adjunctUrl'][:80]}...")
    else:
        print("  未找到")

    # 测试下载
    print("\n[2] 下载测试: 劲仔食品 2023")
    result = downloader.download("003000", "劲仔食品", "2023")
    print(f"  路径: {result.pdf_path}")
    print(f"  大小: {result.file_size / 1024:.1f} KB")
    print(f"  下载: {result.downloaded}")
    print(f"  缓存: {result.from_cache}")
    if result.error:
        print(f"  错误: {result.error}")

    # 测试批量
    print("\n[3] 批量下载测试（3家公司）")
    test_companies = [
        {"code": "003000", "name": "劲仔食品"},
        {"code": "002745", "name": "木林森"},
        {"code": "300274", "name": "阳光电源"},
    ]
    results = downloader.batch_download(test_companies, "2023")
