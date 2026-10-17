"""
自适应配置管理（v1）：基于反馈数据动态调整提取参数
"""
from typing import Dict, Any, List, Optional
from config import QUALITY_THRESHOLDS, AUTO_APPLY_THRESHOLD_CONFIDENCE


class AdaptiveThresholdManager:
    """运行时阈值管理，基于反馈进行内存级调整（不修改config.py）"""

    def __init__(self, feedback_store=None):
        self.feedback_store = feedback_store
        self.overrides: Dict[str, Any] = {}
        self.applied_suggestions: List = []
        self.run_count = 0

    def apply_adjustments(self, quality_thresholds: Optional[Dict] = None) -> Dict:
        """应用已学习的阈值调整，返回调整后的阈值"""
        thresholds = dict(quality_thresholds or QUALITY_THRESHOLDS)

        if not self.feedback_store:
            return thresholds

        suggestions = self.feedback_store.suggest_threshold_adjustments()
        self.run_count += 1

        for s in suggestions:
            if s.confidence >= AUTO_APPLY_THRESHOLD_CONFIDENCE and s.evidence_count >= 3:
                self._apply_suggestion(thresholds, s)

        return thresholds

    def _apply_suggestion(self, thresholds: Dict, suggestion):
        """将建议应用到阈值dict"""
        key = suggestion.config_key

        # 解析嵌套key（如 "QUALITY_THRESHOLDS.table_residue_digit_ratio"）
        parts = key.split('.')
        target = thresholds
        for part in parts[:-1]:
            if part in target:
                target = target[part]
            else:
                return

        final_key = parts[-1]
        if final_key in target:
            old_val = target[final_key]
            target[final_key] = suggestion.suggested_value
            self.overrides[key] = suggestion.suggested_value
            self.applied_suggestions.append(suggestion)

            print(f"  [ADAPTIVE] {key}: {old_val} → {suggestion.suggested_value} "
                  f"(置信度{suggestion.confidence:.0%}, {suggestion.evidence_count}条证据)")

    def get_report(self) -> str:
        """生成可读的调整报告"""
        if not self.feedback_store:
            return "反馈存储未初始化，无法生成建议"

        suggestions = self.feedback_store.suggest_threshold_adjustments()
        if not suggestions:
            return "暂无阈值调整建议（需要至少3条修正记录）"

        lines = [f"阈值调整建议 ({len(suggestions)}条, 运行次数{self.run_count}):", ""]
        for s in suggestions:
            status = "✓ 已自动应用" if s.confidence >= AUTO_APPLY_THRESHOLD_CONFIDENCE else "○ 建议人工审核"
            lines.append(f"  {status}")
            lines.append(f"    参数: {s.config_key}")
            lines.append(f"    当前: {s.current_value} → 建议: {s.suggested_value}")
            lines.append(f"    置信度: {s.confidence:.0%}, 证据: {s.evidence_count}条修正")
            lines.append(f"    说明: {s.description}")
            lines.append("")

        applied = [s for s in suggestions if s.confidence >= AUTO_APPLY_THRESHOLD_CONFIDENCE]
        if applied:
            lines.append(f"本次自动应用了 {len(applied)}/{len(suggestions)} 条建议")

        return '\n'.join(lines)

    def reset(self):
        """重置所有调整"""
        self.overrides = {}
        self.applied_suggestions = []


if __name__ == "__main__":
    from config import FEEDBACK_STORE_PATH
    from feedback_store import FeedbackStore

    store = FeedbackStore(FEEDBACK_STORE_PATH)
    manager = AdaptiveThresholdManager(feedback_store=store)

    adjusted = manager.apply_adjustments()
    print(f"调整后的阈值: {adjusted.get('table_residue_digit_ratio')}")
    print(manager.get_report())
