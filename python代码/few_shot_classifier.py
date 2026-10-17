"""
少样本原型分类器（v1）：基于原型网络（Prototypical Networks）的文本质量分类

核心原理：
1. 使用中文Sentence-Transformer将文本映射到语义向量空间
2. 对每个类别（GOOD/BAD）计算原型向量（类别中心）
3. 新文本通过比较与各原型的距离进行分类
4. 支持增量学习（添加新样本后重新计算原型）

优势：
- 每类仅需5-20个样本即可有效分类
- 训练零成本（无需梯度下降）
- 支持增量更新
- 可解释性强（能看出与哪些训练样本最相似）

参考：
- Snell et al., "Prototypical Networks for Few-shot Learning" (NeurIPS 2017)
- SetFit: Efficient Few-Shot Learning Without Prompts (Hugging Face, 2022)
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import json
import os
from datetime import datetime


@dataclass
class ClassificationResult:
    """分类结果"""
    predicted_class: str          # "GOOD" / "BAD" / "UNKNOWN"
    confidence: float             # 0.0-1.0，分类置信度
    class_scores: Dict[str, float]  # 各类别的相似度
    nearest_examples: List[Tuple[str, float, str]] = field(default_factory=list)  # (label, sim, example)
    is_reliable: bool = True      # 分类是否可靠（置信度>阈值）


class PrototypeClassifier:
    """
    单类别原型分类器

    用于二分类问题（如：GOOD提取 vs BAD提取）
    """

    def __init__(
        self,
        embedder,  # ChineseTextEmbedder
        class_names: List[str] = None,
        confidence_threshold: float = 0.3,
        min_examples_per_class: int = 2,
    ):
        self.embedder = embedder
        self.class_names = class_names or ["GOOD", "BAD"]
        self.confidence_threshold = confidence_threshold
        self.min_examples_per_class = min_examples_per_class

        # 训练数据 {class_name: [text1, text2, ...]}
        self.examples: Dict[str, List[str]] = {c: [] for c in self.class_names}

        # 原型向量 {class_name: ndarray}
        self.prototypes: Dict[str, np.ndarray] = {}

        # 所有训练样本的嵌入 {class_name: [vec1, vec2, ...]}
        self.example_embeddings: Dict[str, List[np.ndarray]] = {c: [] for c in self.class_names}

        # 训练统计
        self.train_count: int = 0
        self.last_updated: Optional[str] = None

    @property
    def is_trained(self) -> bool:
        """是否有足够的训练数据"""
        return all(
            len(self.examples.get(c, [])) >= self.min_examples_per_class
            for c in self.class_names
        )

    @property
    def total_examples(self) -> int:
        return sum(len(v) for v in self.examples.values())

    def add_example(self, text: str, label: str):
        """添加一个训练样本"""
        if label not in self.examples:
            self.examples[label] = []
            self.example_embeddings[label] = []
        if text and text not in self.examples[label]:
            self.examples[label].append(text)

    def add_examples(self, texts: List[str], labels: List[str]):
        """批量添加训练样本"""
        for text, label in zip(texts, labels):
            self.add_example(text, label)

    def fit(self, rebuild_embeddings: bool = True):
        """
        训练/更新分类器
        - 计算所有训练样本的嵌入
        - 计算每个类别的原型向量
        """
        if rebuild_embeddings or not any(bool(v) for v in self.example_embeddings.values()):
            for cls_name in self.class_names:
                texts = self.examples.get(cls_name, [])
                if texts:
                    self.example_embeddings[cls_name] = [
                        self.embedder.embed(t) for t in texts if t and len(t.strip()) > 0
                    ]
                else:
                    self.example_embeddings[cls_name] = []

        # 计算原型
        for cls_name in self.class_names:
            embs = self.example_embeddings.get(cls_name, [])
            if embs:
                proto = np.mean(embs, axis=0)
                norm = np.linalg.norm(proto)
                if norm > 0:
                    proto = proto / norm
                self.prototypes[cls_name] = proto
            else:
                self.prototypes.pop(cls_name, None)

        self.train_count += 1
        self.last_updated = datetime.now().isoformat()

    def predict(self, text: str) -> ClassificationResult:
        """
        对文本进行分类

        Returns:
            ClassificationResult with predicted_class, confidence, etc.
        """
        if not text or len(text.strip()) < 3:
            return ClassificationResult(
                predicted_class="BAD",
                confidence=0.95,
                class_scores={c: 0.0 for c in self.class_names},
                is_reliable=True,
            )

        if not self.is_trained:
            return ClassificationResult(
                predicted_class="UNKNOWN",
                confidence=0.0,
                class_scores={c: 0.5 for c in self.class_names},
                is_reliable=False,
            )

        vec = self.embedder.embed(text)

        # 计算与各原型的相似度
        class_scores = {}
        for cls_name, proto in self.prototypes.items():
            class_scores[cls_name] = float(np.dot(vec, proto))

        # 预测类别
        best_cls = max(class_scores, key=class_scores.get)
        best_score = class_scores[best_cls]

        # 置信度计算（边距法）
        sorted_scores = sorted(class_scores.values(), reverse=True)
        if len(sorted_scores) > 1:
            margin = sorted_scores[0] - sorted_scores[1]
            # 将margin映射到0-1（经验：margin 0.05->0.1, 0.1->0.5, 0.2->0.9）
            confidence = float(np.clip(margin * 5.0, 0.0, 1.0))
        else:
            confidence = float(np.clip(best_score, 0.0, 1.0))

        # 找最近邻
        nearest = []
        for cls_name in self.class_names:
            embs = self.example_embeddings.get(cls_name, [])
            texts = self.examples.get(cls_name, [])
            for emb, t in zip(embs, texts):
                sim = float(np.dot(vec, emb))
                nearest.append((cls_name, sim, t[:100]))
        nearest.sort(key=lambda x: x[1], reverse=True)

        is_reliable = confidence >= self.confidence_threshold

        return ClassificationResult(
            predicted_class=best_cls,
            confidence=confidence,
            class_scores=class_scores,
            nearest_examples=nearest[:5],
            is_reliable=is_reliable,
        )

    def predict_proba(self, text: str) -> Dict[str, float]:
        """
        返回各类别的概率分布（softmax over similarities）

        注意：这不是真正的概率，而是归一化后的相似度
        """
        result = self.predict(text)
        scores = result.class_scores
        # Softmax
        exp_scores = {k: np.exp(v * 5.0) for k, v in scores.items()}  # temperature=0.2
        total = sum(exp_scores.values())
        if total > 0:
            return {k: round(v / total, 4) for k, v in exp_scores.items()}
        return {k: 1.0 / len(scores) for k in scores}

    def remove_example(self, text: str, label: str):
        """移除一个错误标注的样本"""
        if label in self.examples and text in self.examples[label]:
            self.examples[label].remove(text)
            # 需要重新fit
            self.fit(rebuild_embeddings=True)

    def save(self, path: str):
        """保存分类器状态"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "class_names": self.class_names,
            "confidence_threshold": self.confidence_threshold,
            "min_examples_per_class": self.min_examples_per_class,
            "examples": self.examples,
            "train_count": self.train_count,
            "last_updated": self.last_updated,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str, embedder) -> "PrototypeClassifier":
        """加载分类器状态"""
        with open(path, 'r', encoding='utf-8') as f:
            state = json.load(f)

        clf = cls(
            embedder=embedder,
            class_names=state["class_names"],
            confidence_threshold=state.get("confidence_threshold", 0.3),
            min_examples_per_class=state.get("min_examples_per_class", 2),
        )
        clf.examples = state["examples"]
        clf.train_count = state.get("train_count", 0)
        clf.last_updated = state.get("last_updated")
        clf.fit(rebuild_embeddings=True)
        return clf


class MultiModuleClassifier:
    """
    多模块质量分类器

    为每个模块（module1-5, policy_info）训练独立的二分类器
    """

    def __init__(self, embedder, confidence_threshold: float = 0.3):
        self.embedder = embedder
        self.confidence_threshold = confidence_threshold
        self.classifiers: Dict[str, PrototypeClassifier] = {}

        # 模块列表
        self.module_names = [
            "module1", "module2", "module3",
            "module4", "module5", "policy_info",
        ]

        for name in self.module_names:
            self.classifiers[name] = PrototypeClassifier(
                embedder=embedder,
                class_names=["GOOD", "BAD"],
                confidence_threshold=confidence_threshold,
            )

    def build_from_reference_texts(self, reference_texts: Dict[str, Dict[str, List[str]]]):
        """
        从参考文本构建各模块分类器

        Args:
            reference_texts: {
                "module1": {"GOOD": [...], "BAD": [...]},
                "module2": {"GOOD": [...], "BAD": [...]},
                ...
            }
        """
        for mod_name in self.module_names:
            mod_refs = reference_texts.get(mod_name, {})
            good_texts = mod_refs.get("GOOD", [])
            bad_texts = mod_refs.get("BAD", [])

            clf = self.classifiers[mod_name]
            clf.add_examples(good_texts, ["GOOD"] * len(good_texts))
            clf.add_examples(bad_texts, ["BAD"] * len(bad_texts))
            clf.fit()

        print(f"  [ML] 多模块分类器已构建: {sum(c.total_examples for c in self.classifiers.values())} 个训练样本")

    def add_examples_from_feedback(
        self, module_name: str, good_texts: List[str], bad_texts: List[str]
    ):
        """从人工反馈添加训练样本"""
        if module_name not in self.classifiers:
            return

        clf = self.classifiers[module_name]
        clf.add_examples(good_texts, ["GOOD"] * len(good_texts))
        clf.add_examples(bad_texts, ["BAD"] * len(bad_texts))
        clf.fit()
        print(f"  [ML] {module_name}: 新增 {len(good_texts)} GOOD + {len(bad_texts)} BAD 样本")

    def classify(self, module_name: str, text: str) -> Tuple[str, float, ClassificationResult]:
        """
        对单个模块的提取结果进行分类

        Returns:
            (predicted_quality, confidence, full_result)
            quality: "GOOD" / "BAD" / "UNKNOWN"
        """
        if module_name not in self.classifiers:
            return "UNKNOWN", 0.0, None

        clf = self.classifiers[module_name]
        result = clf.predict(text)

        quality = result.predicted_class
        confidence = result.confidence

        return quality, confidence, result

    def classify_all_modules(
        self, extracted: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        对所有模块的提取结果进行质量分类

        Returns:
            {
                "module1": {"quality": "GOOD", "confidence": 0.85, "details": ...},
                ...
            }
        """
        results = {}
        for mod_name in self.module_names:
            text = extracted.get(mod_name, "")
            quality, confidence, result = self.classify(mod_name, text)
            results[mod_name] = {
                "quality": quality,
                "confidence": confidence,
                "predicted_class": result.predicted_class if result else "UNKNOWN",
                "class_scores": result.class_scores if result else {},
                "nearest_examples": result.nearest_examples[:3] if result else [],
            }
        return results

    def get_overall_quality(self, extracted: Dict[str, str]) -> Tuple[float, str]:
        """
        计算整体质量分 (0-100) 和总体置信度
        """
        module_results = self.classify_all_modules(extracted)
        scores = []
        weights = {
            "module1": 1.0, "module2": 1.5, "module3": 1.0,
            "module4": 1.0, "module5": 1.5, "policy_info": 1.0,
        }

        for mod_name, result in module_results.items():
            w = weights.get(mod_name, 1.0)
            if result["quality"] == "GOOD":
                scores.append(100 * w * result["confidence"])
            elif result["quality"] == "BAD":
                scores.append(30 * w * result.get("confidence", 0.5))
            else:
                scores.append(50 * w)

        if scores:
            total_weight = sum(weights.values())
            overall = sum(scores) / total_weight
        else:
            overall = 50.0

        # 确定总体置信度
        all_reliable = all(
            r["confidence"] >= self.confidence_threshold or r["quality"] == "UNKNOWN"
            for r in module_results.values()
        )

        if overall >= 75:
            band = "HIGH"
        elif overall >= 45:
            band = "MEDIUM"
        else:
            band = "LOW"

        return round(overall, 1), band

    def save(self, base_path: str):
        """保存所有模块分类器"""
        os.makedirs(base_path, exist_ok=True)
        for mod_name, clf in self.classifiers.items():
            path = os.path.join(base_path, f"{mod_name}_classifier.json")
            clf.save(path)
        print(f"  [ML] 多模块分类器已保存到: {base_path}")

    @classmethod
    def load(cls, base_path: str, embedder) -> "MultiModuleClassifier":
        """加载所有模块分类器"""
        mclf = cls(embedder=embedder)
        for mod_name in mclf.module_names:
            path = os.path.join(base_path, f"{mod_name}_classifier.json")
            if os.path.exists(path):
                try:
                    mclf.classifiers[mod_name] = PrototypeClassifier.load(path, embedder)
                except Exception as e:
                    print(f"  [WARN] 加载 {mod_name} 分类器失败: {e}")
        return mclf

    @property
    def total_training_examples(self) -> int:
        return sum(c.total_examples for c in self.classifiers.values())

    def get_training_stats(self) -> Dict[str, Dict]:
        """获取各模块的训练统计"""
        stats = {}
        for mod_name in self.module_names:
            clf = self.classifiers[mod_name]
            stats[mod_name] = {
                "total_examples": clf.total_examples,
                "good_count": len(clf.examples.get("GOOD", [])),
                "bad_count": len(clf.examples.get("BAD", [])),
                "is_trained": clf.is_trained,
                "last_updated": clf.last_updated,
            }
        return stats


if __name__ == "__main__":
    from ml_embeddings import ChineseTextEmbedder, MODULE_REFERENCE_TEXTS

    print("=== 少样本原型分类器测试 ===\n")

    embedder = ChineseTextEmbedder()

    # 测试单个分类器
    clf = PrototypeClassifier(embedder, class_names=["GOOD", "BAD"])
    clf.add_examples(
        ["汇率风险：美元波动影响公司利润。", "外汇风险敞口包括欧元和港币。"],
        ["GOOD", "GOOD"]
    )
    clf.add_examples(
        ["会计准则要求...", "公司治理结构完善..."],
        ["BAD", "BAD"]
    )
    clf.fit()

    test_texts = [
        "汇率波动风险：公司出口业务主要以美元结算，面临较大的汇率波动风险。",
        "公司以人民币为记账本位币，遵循企业会计准则。",
        "报告期内外汇衍生品实现收益285万元。",
    ]

    for text in test_texts:
        result = clf.predict(text)
        print(f"  文本: {text[:50]}...")
        print(f"  预测: {result.predicted_class}, 置信度: {result.confidence:.3f}")
        print(f"  分数: {result.class_scores}")
        print(f"  最近邻: {[(n[0], f'{n[1]:.3f}', n[2][:40]) for n in result.nearest_examples[:2]]}")
        print()

    # 测试多模块分类器
    print("\n=== 多模块分类器测试 ===")
    mclf = MultiModuleClassifier(embedder)
    mclf.build_from_reference_texts(MODULE_REFERENCE_TEXTS)
    print(f"  总训练样本: {mclf.total_training_examples}")
    for mod_name, stats in mclf.get_training_stats().items():
        print(f"  {mod_name}: {stats}")
