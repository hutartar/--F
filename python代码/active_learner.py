"""
主动学习循环（v1）：基于不确定性采样的模型持续改进

核心思路：
1. 对未标注/低置信度样本进行不确定性评估
2. 优先选择模型最不确定的样本供人工审核
3. 将审核后的样本加入训练集，增量更新模型
4. 循环迭代直到模型收敛

工作流程：
    提取结果 → ML质量评分 → 低置信度样本 → 人工审核 → 更新训练集 → 重新训练 → 更高准确率
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import os

from feedback_store import FeedbackStore, CorrectionRecord


@dataclass
class UncertaintySample:
    """不确定性样本"""
    company_code: str
    company_name: str
    module_name: str
    text: str
    predicted_class: str
    confidence: float
    class_scores: Dict[str, float]
    uncertainty_score: float      # 0-1，越高越不确定
    flags: List[str] = field(default_factory=list)
    quality_score: float = 0.0

    @property
    def needs_review(self) -> bool:
        """是否需要人工审核"""
        return self.confidence < 0.5 or self.uncertainty_score > 0.4


@dataclass
class LearningIteration:
    """一次学习迭代的记录"""
    iteration: int
    timestamp: str
    samples_added: int
    good_added: int
    bad_added: int
    model_accuracy_before: float
    model_accuracy_after: Optional[float]
    low_confidence_count_before: int
    low_confidence_count_after: int


class ActiveLearner:
    """
    主动学习管理器

    负责：
    1. 识别不确定性样本
    2. 管理人工审核队列
    3. 协调模型更新
    4. 追踪学习进度
    """

    def __init__(
        self,
        multi_module_classifier,      # MultiModuleClassifier
        embedder,                     # ChineseTextEmbedder
        feedback_store: FeedbackStore,
        uncertainty_threshold: float = 0.4,
        confidence_threshold: float = 0.5,
        max_samples_per_iteration: int = 20,
    ):
        self.mclf = multi_module_classifier
        self.embedder = embedder
        self.feedback_store = feedback_store
        self.uncertainty_threshold = uncertainty_threshold
        self.confidence_threshold = confidence_threshold
        self.max_samples_per_iteration = max_samples_per_iteration

        self.iteration: int = 0
        self.history: List[LearningIteration] = []
        self.review_queue: List[UncertaintySample] = []

    def compute_uncertainty(
        self, class_scores: Dict[str, float], method: str = "margin"
    ) -> float:
        """
        计算不确定性分数

        方法：
        - "margin": 最佳与次佳分数之差（margin小→不确定）
        - "entropy": 熵（高熵→不确定）
        - "least_confidence": 1 - 最佳分数（低置信度→不确定）
        """
        scores = list(class_scores.values())
        if not scores:
            return 1.0

        if method == "margin":
            if len(scores) > 1:
                sorted_scores = sorted(scores, reverse=True)
                margin = sorted_scores[0] - sorted_scores[1]
                return float(np.clip(1.0 - margin * 3.0, 0.0, 1.0))
            return 1.0 - max(scores)

        elif method == "entropy":
            # 归一化为概率分布
            exp_scores = np.exp(np.array(scores) * 3.0)
            probs = exp_scores / exp_scores.sum()
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            max_entropy = np.log(len(scores))
            return float(entropy / max_entropy) if max_entropy > 0 else 0.0

        elif method == "least_confidence":
            return float(1.0 - max(scores))

        return 0.5

    def identify_uncertain_samples(
        self, results: List[Dict]
    ) -> List[UncertaintySample]:
        """
        从当前批次结果中识别不确定性样本

        Args:
            results: 提取结果列表（含_quality字段）

        Returns:
            按不确定性降序排列的样本列表
        """
        uncertain_samples = []

        for record in results:
            quality = record.get("_quality", {})
            for mod_name in [
                "module1", "module2", "module3",
                "module4", "module5", "policy_info",
            ]:
                text = record.get(mod_name, "")
                if not text or len(text) < 3:
                    continue

                # 使用ML分类器预测
                try:
                    quality_label, confidence, result = self.mclf.classify(mod_name, text)
                except Exception:
                    continue

                if result is None:
                    continue

                uncertainty = self.compute_uncertainty(
                    result.class_scores, method="margin"
                )

                # 收集低置信度/高不确定性样本
                if confidence < self.confidence_threshold or uncertainty > self.uncertainty_threshold:
                    sample = UncertaintySample(
                        company_code=str(record.get("code", "")),
                        company_name=str(record.get("name", "")),
                        module_name=mod_name,
                        text=text[:500],
                        predicted_class=quality_label,
                        confidence=confidence,
                        class_scores=result.class_scores,
                        uncertainty_score=uncertainty,
                        flags=quality.get(mod_name, {}).get("flags", []),
                        quality_score=quality.get(mod_name, {}).get("overall_score", 0),
                    )
                    uncertain_samples.append(sample)

        # 按不确定性降序排列
        uncertain_samples.sort(
            key=lambda x: (x.uncertainty_score, -x.confidence),
            reverse=True,
        )

        return uncertain_samples

    def build_review_queue(self, results: List[Dict]):
        """构建人工审核队列"""
        uncertain = self.identify_uncertain_samples(results)
        self.review_queue = uncertain[:self.max_samples_per_iteration]

        print(f"  [AL] 识别到 {len(uncertain)} 个不确定样本，"
              f"前 {len(self.review_queue)} 个加入审核队列")

    def get_review_queue(self) -> List[UncertaintySample]:
        """获取当前审核队列"""
        return self.review_queue

    def submit_review(
        self,
        reviewed_samples: List[Tuple[UncertaintySample, str]],
    ) -> int:
        """
        提交人工审核结果

        Args:
            reviewed_samples: [(UncertaintySample, corrected_label), ...]
                            label: "GOOD" 表示提取正确，"BAD" 表示提取有问题

        Returns:
            添加的训练样本数
        """
        new_good = []
        new_bad = []

        for sample, label in reviewed_samples:
            if label == "GOOD":
                new_good.append(sample.text)
            elif label == "BAD":
                new_bad.append(sample.text)

            # 记录到反馈存储
            try:
                cr = CorrectionRecord(
                    timestamp=datetime.now().isoformat(),
                    company_code=sample.company_code,
                    company_name=sample.company_name,
                    module_name=sample.module_name,
                    original_text=sample.text,
                    corrected_text="" if label == "BAD" else sample.text,
                    quality_before=sample.quality_score,
                    quality_after=85.0 if label == "GOOD" else 15.0,
                    correction_type="REVIEW_FEEDBACK",
                    pattern_triggered=f"predicted={sample.predicted_class}, actual={label}",
                    original_length=len(sample.text),
                    corrected_length=0 if label == "BAD" else len(sample.text),
                )
                self.feedback_store.record_correction(cr)
            except Exception:
                pass

        # 更新分类器
        if new_good or new_bad:
            for mod_name in self.mclf.module_names:
                # 按模块分组
                mod_good = [s.text for s, l in reviewed_samples
                           if s.module_name == mod_name and l == "GOOD"]
                mod_bad = [s.text for s, l in reviewed_samples
                          if s.module_name == mod_name and l == "BAD"]

                if mod_good or mod_bad:
                    self.mclf.add_examples_from_feedback(
                        mod_name, mod_good, mod_bad
                    )

        total_added = len(new_good) + len(new_bad)
        return total_added

    def run_iteration(
        self, results: List[Dict], auto_apply: bool = False
    ) -> Tuple[int, int, int]:
        """
        运行一次主动学习迭代

        Args:
            results: 当前批次提取结果
            auto_apply: 是否自动应用高置信度分类结果

        Returns:
            (新增GOOD数, 新增BAD数, 低置信度样本数)
        """
        self.iteration += 1
        timestamp = datetime.now().isoformat()

        # 统计迭代前状态
        low_conf_before = sum(
            1 for r in results
            for mod in self.mclf.module_names
            if r.get("_quality", {}).get(mod, {}).get("confidence", "HIGH") in ("LOW", "MEDIUM")
        )

        # 如果自动应用，将高置信度分类结果用于增强训练
        if auto_apply:
            auto_good = []
            auto_bad = []
            for record in results:
                for mod_name in self.mclf.module_names:
                    text = record.get(mod_name, "")
                    if not text or len(text) < 3:
                        continue
                    try:
                        quality, conf, result = self.mclf.classify(mod_name, text)
                        if conf > 0.85:  # 高置信度
                            if quality == "GOOD":
                                auto_good.append((mod_name, text))
                            elif quality == "BAD":
                                auto_bad.append((mod_name, text))
                    except Exception:
                        continue

            for mod_name in self.mclf.module_names:
                mod_good = [t for m, t in auto_good if m == mod_name]
                mod_bad = [t for m, t in auto_bad if m == mod_name]
                if mod_good or mod_bad:
                    self.mclf.add_examples_from_feedback(mod_name, mod_good, mod_bad)

        # 构建审核队列
        self.build_review_queue(results)

        # 统计迭代后状态
        low_conf_after = len(self.review_queue)

        learning_iter = LearningIteration(
            iteration=self.iteration,
            timestamp=timestamp,
            samples_added=low_conf_before,
            good_added=low_conf_before - low_conf_after,
            bad_added=0,
            model_accuracy_before=0.0,
            model_accuracy_after=None,
            low_confidence_count_before=low_conf_before,
            low_confidence_count_after=low_conf_after,
        )
        self.history.append(learning_iter)

        return 0, 0, low_conf_after

    def get_progress_report(self) -> str:
        """生成学习进度报告"""
        if not self.history:
            return "尚未开始主动学习"

        lines = ["主动学习进度报告", "=" * 40]
        lines.append(f"总迭代次数: {self.iteration}")

        stats = self.mclf.get_training_stats()
        lines.append("\n各模块训练状态:")
        for mod_name, s in stats.items():
            status = "✓ 已训练" if s["is_trained"] else "○ 待训练"
            lines.append(f"  {mod_name}: {status} "
                        f"(GOOD={s['good_count']}, BAD={s['bad_count']}, "
                        f"总计={s['total_examples']})")

        if self.review_queue:
            lines.append(f"\n当前审核队列: {len(self.review_queue)} 个样本")
            for i, sample in enumerate(self.review_queue[:5]):
                lines.append(f"  [{i+1}] {sample.module_name} "
                           f"({sample.company_name}): "
                           f"置信度={sample.confidence:.2f}, "
                           f"不确定度={sample.uncertainty_score:.2f}")

        # 学习趋势
        if len(self.history) >= 2:
            first = self.history[0]
            last = self.history[-1]
            lines.append(f"\n学习趋势:")
            lines.append(f"  低置信度样本: {first.low_confidence_count_before} → "
                        f"{last.low_confidence_count_after}")
            if first.low_confidence_count_before > 0:
                improvement = (
                    (first.low_confidence_count_before - last.low_confidence_count_after)
                    / first.low_confidence_count_before * 100
                )
                lines.append(f"  改善幅度: {improvement:.0f}%")

        return '\n'.join(lines)

    def save_state(self, path: str):
        """保存主动学习状态"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "iteration": self.iteration,
            "uncertainty_threshold": self.uncertainty_threshold,
            "confidence_threshold": self.confidence_threshold,
            "review_queue": [
                {
                    "company_code": s.company_code,
                    "company_name": s.company_name,
                    "module_name": s.module_name,
                    "text": s.text[:200],
                    "predicted_class": s.predicted_class,
                    "confidence": s.confidence,
                    "uncertainty_score": s.uncertainty_score,
                }
                for s in self.review_queue
            ],
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_state(cls, path: str, mclf, embedder, feedback_store) -> "ActiveLearner":
        """加载状态"""
        learner = cls(mclf, embedder, feedback_store)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            learner.iteration = state.get("iteration", 0)
        return learner


if __name__ == "__main__":
    from ml_embeddings import ChineseTextEmbedder, MODULE_REFERENCE_TEXTS
    from few_shot_classifier import MultiModuleClassifier
    from feedback_store import FeedbackStore
    from config import FEEDBACK_STORE_PATH

    print("=== 主动学习测试 ===\n")

    embedder = ChineseTextEmbedder()
    mclf = MultiModuleClassifier(embedder)
    mclf.build_from_reference_texts(MODULE_REFERENCE_TEXTS)

    store = FeedbackStore(FEEDBACK_STORE_PATH)

    learner = ActiveLearner(mclf, embedder, store)

    # 模拟一些提取结果
    test_results = [
        {
            "code": "3000", "name": "测试公司A",
            "module1": "汇率风险：公司面临美元汇率波动。",
            "module2": "公司报告期不存在衍生品投资。",
            "module4": "4、记账本位币本公司以人民币为记账本位币。",
            "_quality": {
                "module1": {"overall_score": 55, "confidence": "MEDIUM", "flags": []},
                "module2": {"overall_score": 80, "confidence": "HIGH", "flags": []},
                "module4": {"overall_score": 45, "confidence": "MEDIUM", "flags": ["HIGH_BOILERPLATE"]},
            },
        },
    ]

    # 识别不确定样本
    learner.build_review_queue(test_results)
    print(learner.get_progress_report())

    # 模拟人工审核
    review_results = []
    for sample in learner.review_queue[:3]:
        # 假设人工判断：module1的提取是正确的
        review_results.append((sample, "GOOD"))

    if review_results:
        n = learner.submit_review(review_results)
        print(f"\n提交了 {n} 条审核结果")

    print("\n" + learner.get_progress_report())
