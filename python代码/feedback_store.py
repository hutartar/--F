"""
自学习反馈存储系统（v1）：JSONL持久化 + 模式分析 + 阈值建议
"""
import os
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Optional, Any
from collections import Counter


@dataclass
class CorrectionRecord:
    """一次修正事件"""
    timestamp: str
    company_code: str
    company_name: str
    module_name: str
    original_text: str
    corrected_text: str
    quality_before: float
    quality_after: Optional[float]
    correction_type: str   # BOILERPLATE_REMOVAL / RELEVANCE_FILTER / TABLE_CLEANUP / TRUNCATION_FIX / OTHER
    pattern_triggered: Optional[str]
    original_length: int
    corrected_length: int

    @property
    def length_reduction(self) -> float:
        if self.original_length == 0:
            return 0
        return (self.original_length - self.corrected_length) / self.original_length


@dataclass
class ThresholdSuggestion:
    """阈值调整建议"""
    config_key: str
    current_value: float
    suggested_value: float
    confidence: float
    evidence_count: int
    description: str


class FeedbackStore:
    """JSONL格式的反馈存储与分析"""

    def __init__(self, store_path: str):
        self.store_path = store_path
        self._ensure_store()

    def _ensure_store(self):
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        if not os.path.exists(self.store_path):
            with open(self.store_path, 'w', encoding='utf-8') as f:
                pass  # 创建空文件

    def record_correction(self, record: CorrectionRecord):
        """追加一条修正记录"""
        with open(self.store_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + '\n')

    def record_batch(self, results: List[Dict]):
        """从精炼后的结果批量记录修正"""
        for rec in results:
            refined_modules = rec.get("_refined_modules", [])
            if not refined_modules:
                continue

            quality = rec.get("_quality", {})
            for mod in refined_modules:
                mod_quality = quality.get(mod, {})
                flags = mod_quality.get("flags", [])

                # 推断修正类型
                correction_type = "OTHER"
                if "HIGH_BOILERPLATE" in flags or "ACCOUNTING_POLICY_MIXED" in flags:
                    correction_type = "BOILERPLATE_REMOVAL"
                elif "LOW_RELEVANCE" in flags:
                    correction_type = "RELEVANCE_FILTER"
                elif "TABLE_RESIDUE" in flags:
                    correction_type = "TABLE_CLEANUP"
                elif "TRUNCATED" in flags:
                    correction_type = "TRUNCATION_FIX"

                cr = CorrectionRecord(
                    timestamp=datetime.now().isoformat(),
                    company_code=str(rec.get("code", "")),
                    company_name=str(rec.get("name", "")),
                    module_name=mod,
                    original_text="",  # 精炼后已覆盖，通过比较获取
                    corrected_text=rec.get(mod, ""),
                    quality_before=mod_quality.get("overall_score", 0),
                    quality_after=None,  # 精炼后需要重新评分
                    correction_type=correction_type,
                    pattern_triggered=", ".join(flags) if flags else None,
                    original_length=0,
                    corrected_length=len(rec.get(mod, "")),
                )
                self.record_correction(cr)

    def get_all_corrections(self) -> List[CorrectionRecord]:
        """加载所有修正记录"""
        records = []
        if not os.path.exists(self.store_path):
            return records
        with open(self.store_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        d = json.loads(line)
                        records.append(CorrectionRecord(**d))
                    except Exception:
                        continue
        return records

    def get_pattern_statistics(self) -> Dict[str, Dict]:
        """按修正类型和模块统计"""
        records = self.get_all_corrections()
        if not records:
            return {}

        stats = {}
        for r in records:
            key = f"{r.module_name}:{r.correction_type}"
            if key not in stats:
                stats[key] = {
                    "count": 0,
                    "total_quality_before": 0,
                    "avg_length_reduction": 0,
                }
            stats[key]["count"] += 1
            stats[key]["total_quality_before"] += r.quality_before
            stats[key]["avg_length_reduction"] += r.length_reduction

        for key in stats:
            s = stats[key]
            s["avg_quality_before"] = round(s["total_quality_before"] / s["count"], 1)
            s["avg_length_reduction"] = round(s["avg_length_reduction"] / s["count"], 2)

        return stats

    def suggest_threshold_adjustments(self) -> List[ThresholdSuggestion]:
        """基于修正记录分析，建议阈值调整"""
        records = self.get_all_corrections()
        if len(records) < 3:
            return []

        suggestions = []

        # 分析表格残片修正
        table_fixes = [r for r in records if r.correction_type == "TABLE_CLEANUP"]
        if len(table_fixes) >= 3:
            avg_orig_len = sum(r.original_length for r in table_fixes if r.original_length > 0) / max(1, len([r for r in table_fixes if r.original_length > 0]))
            # 如果大量表格残片修正，建议降低数字密度阈值
            suggestions.append(ThresholdSuggestion(
                config_key="QUALITY_THRESHOLDS.table_residue_digit_ratio",
                current_value=0.45,
                suggested_value=0.38,
                confidence=min(0.9, len(table_fixes) / 10),
                evidence_count=len(table_fixes),
                description=f"发现{len(table_fixes)}次表格残片修正，建议降低数字密度阈值以更早检测表格行",
            ))

        # 分析锅炉板修正
        bp_fixes = [r for r in records if r.correction_type == "BOILERPLATE_REMOVAL"]
        if len(bp_fixes) >= 3:
            suggestions.append(ThresholdSuggestion(
                config_key="QUALITY_THRESHOLDS.boilerplate_min_score",
                current_value=0.6,
                suggested_value=0.5,
                confidence=min(0.9, len(bp_fixes) / 10),
                evidence_count=len(bp_fixes),
                description=f"发现{len(bp_fixes)}次锅炉板修正，建议降低锅炉板检测阈值以提高敏感度",
            ))

        # 分析截断修正
        trunc_fixes = [r for r in records if r.correction_type == "TRUNCATION_FIX"]
        if len(trunc_fixes) >= 2:
            suggestions.append(ThresholdSuggestion(
                config_key="QUALITY_THRESHOLDS.truncation_detection",
                current_value=0.75,
                suggested_value=0.85,
                confidence=min(0.8, len(trunc_fixes) / 8),
                evidence_count=len(trunc_fixes),
                description=f"发现{len(trunc_fixes)}次截断修正，建议提高截断检测敏感度",
            ))

        return suggestions

    def get_stats_summary(self) -> str:
        """生成可读的统计摘要"""
        stats = self.get_pattern_statistics()
        records = self.get_all_corrections()

        if not records:
            return "无修正记录"

        by_module = Counter(r.module_name for r in records)
        by_type = Counter(r.correction_type for r in records)

        lines = [f"共 {len(records)} 条修正记录"]
        lines.append(f"按模块: {dict(by_module)}")
        lines.append(f"按类型: {dict(by_type)}")

        suggestions = self.suggest_threshold_adjustments()
        if suggestions:
            lines.append(f"\n阈值调整建议 ({len(suggestions)}条):")
            for s in suggestions:
                lines.append(f"  [{s.config_key}] {s.current_value} → {s.suggested_value} "
                           f"(置信度: {s.confidence:.0%}, 证据: {s.evidence_count}条)")

        return '\n'.join(lines)


if __name__ == "__main__":
    from config import FEEDBACK_STORE_PATH
    store = FeedbackStore(FEEDBACK_STORE_PATH)

    # 测试记录
    test_rec = CorrectionRecord(
        timestamp=datetime.now().isoformat(),
        company_code="3000",
        company_name="劲仔食品",
        module_name="module4",
        original_text="1、遵循企业会计准则的声明...2、会计期间...3、营业周期...4、记账本位币本公司采用人民币为记账本位币。",
        corrected_text="本公司采用人民币为记账本位币。",
        quality_before=35.0,
        quality_after=90.0,
        correction_type="BOILERPLATE_REMOVAL",
        pattern_triggered="HIGH_BOILERPLATE",
        original_length=120,
        corrected_length=15,
    )
    store.record_correction(test_rec)
    print(store.get_stats_summary())
