"""
ML质量评分引擎（v2）：语义嵌入 + 少样本分类 + TF-IDF回退

升级内容：
- 主方案：基于Chinese Sentence-BERT的语义原型分类
- 回退方案：原TF-IDF锅炉板检测（当嵌入模型不可用时）
- 融合方案：语义分 + 启发式维度分数的加权综合
- 增量学习：支持从反馈中持续改进

解决欠拟合问题的关键：
1. 预训练模型提供丰富的语义先验知识（无需大量标注）
2. 原型网络在小样本下表现优异（每类5-20个样本即可）
3. 数据增强扩充训练集
4. 主动学习持续改进
"""
import os
import re
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple

import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from config import TEMPLATE_NGRAMS, QUALITY_THRESHOLDS, MODULE_QUALITY_WEIGHTS

# 可选的ML组件
_HAS_EMBEDDINGS = False
_HAS_CLASSIFIER = False
_ChineseTextEmbedder = None
_MultiModuleClassifier = None
_MODULE_REFERENCE_TEXTS = None
_SemanticBoilerplateDetector = None
_PREDEFINED_BOILERPLATE = None

try:
    from ml_embeddings import (
        ChineseTextEmbedder,
        SemanticBoilerplateDetector,
        MODULE_REFERENCE_TEXTS,
        PREDEFINED_BOILERPLATE,
    )
    _ChineseTextEmbedder = ChineseTextEmbedder
    _SemanticBoilerplateDetector = SemanticBoilerplateDetector
    _MODULE_REFERENCE_TEXTS = MODULE_REFERENCE_TEXTS
    _PREDEFINED_BOILERPLATE = PREDEFINED_BOILERPLATE
    _HAS_EMBEDDINGS = True
except ImportError:
    pass

try:
    from few_shot_classifier import MultiModuleClassifier, PrototypeClassifier
    _MultiModuleClassifier = MultiModuleClassifier
    _HAS_CLASSIFIER = True
except ImportError:
    pass


# ── 全局单例 ──
_embedder_instance = None
_classifier_instance = None
_bp_detector_instance = None


def _get_embedder() -> Optional[Any]:
    """获取全局嵌入器实例（延迟加载）"""
    global _embedder_instance
    if _embedder_instance is None:
        # 方案1：优先加载本地缓存的 BGE 语义嵌入模型
        if _HAS_EMBEDDINGS:
            try:
                from sentence_transformers import SentenceTransformer
                # 模型已缓存在本地，直接离线加载，不联网
                model = SentenceTransformer("BAAI/bge-small-zh-v1.5", local_files_only=True)
                if model:
                    _embedder_instance = _ChineseTextEmbedder(
                        model_name="BAAI/bge-small-zh-v1.5",
                        cache_embeddings=True,
                    )
                    _embedder_instance._model = model
                    _embedder_instance._ready = True
                    print("  [ML] 使用 BGE-small-zh 语义嵌入模型（本地缓存离线加载）")
                    return _embedder_instance
            except Exception as e:
                print(f"  [INFO] BGE本地模型加载失败，降级TF-IDF: {e}")

        # 方案2：轻量级TF-IDF+SVD（纯本地回退）
        try:
            from lightweight_embedder import LightweightChineseEmbedder
            _embedder_instance = LightweightChineseEmbedder(
                svd_components=256, max_features=8000
            )
            print("  [ML] 使用轻量级TF-IDF+SVD嵌入器（纯本地回退）")
            return _embedder_instance
        except Exception:
            pass

    return _embedder_instance


def _get_classifier() -> Optional[Any]:
    """获取全局分类器实例（延迟加载）"""
    global _classifier_instance
    if _classifier_instance is None and _HAS_CLASSIFIER:
        embedder = _get_embedder()
        if embedder is not None:
            try:
                _classifier_instance = _MultiModuleClassifier(
                    embedder, confidence_threshold=0.3
                )
                # 用预定义的参考文本初始化（仅在嵌入器ready后才有效）
                if _MODULE_REFERENCE_TEXTS and embedder.is_ready():
                    _classifier_instance.build_from_reference_texts(_MODULE_REFERENCE_TEXTS)
                    print(f"  [ML] 分类器已用参考文本初始化")
            except Exception as e:
                print(f"  [WARN] 分类器创建失败: {e}")
                _classifier_instance = None
    return _classifier_instance


def _get_bp_detector() -> Optional[Any]:
    """获取全局语义锅炉板检测器"""
    global _bp_detector_instance
    if _bp_detector_instance is None:
        embedder = _get_embedder()
        if embedder and embedder.is_ready():
            if _HAS_EMBEDDINGS and _SemanticBoilerplateDetector:
                try:
                    _bp_detector_instance = _SemanticBoilerplateDetector(embedder)
                    if _PREDEFINED_BOILERPLATE:
                        _bp_detector_instance.build_prototypes(_PREDEFINED_BOILERPLATE)
                except Exception:
                    _bp_detector_instance = None
    return _bp_detector_instance


# ── 数据结构 ──

@dataclass
class QualityResult:
    """单个模块的质量评估结果"""
    module_name: str
    overall_score: float          # 0-100
    dimensions: Dict[str, float]  # 各维度分数
    flags: List[str]              # 警告标签
    confidence: str               # HIGH / MEDIUM / LOW
    boilerplate_phrases: List[str] = field(default_factory=list)
    ml_score: Optional[float] = None      # ML语义分
    heuristic_score: Optional[float] = None  # 启发式分
    ml_details: Optional[Dict] = None     # ML分类详情

    @property
    def is_low_confidence(self) -> bool:
        return self.confidence == "LOW"

    @property
    def needs_refinement(self) -> bool:
        return self.confidence in ("LOW", "MEDIUM")


# ── TF-IDF 锅炉板检测器（回退方案）──

class BoilerplateDetector:
    """基于TF-IDF的锅炉板/模板文本检测器（v1回退方案）"""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            tokenizer=self._tokenize,
            ngram_range=(1, 3),
            max_features=5000,
            min_df=2,
            max_df=0.85,
            token_pattern=None,
        )
        self.idf = None
        self.vocabulary = None
        self.corpus_tfidf = None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        words = jieba.lcut(text)
        return [w.strip() for w in words if len(w.strip()) >= 1]

    def fit(self, documents: List[str]):
        docs = [d for d in documents if d and len(d) > 10]
        if not docs:
            return
        self.corpus_tfidf = self.vectorizer.fit_transform(docs)
        self.idf = dict(zip(self.vectorizer.get_feature_names_out(),
                            self.vectorizer.idf_))
        self.vocabulary = self.vectorizer.vocabulary_

    def score_boilerplate(self, text: str) -> float:
        if not text or len(text) < 5:
            return 1.0
        try:
            vec = self.vectorizer.transform([text])
            nonzero = vec.data
            if len(nonzero) == 0:
                return 1.0
            mean_tfidf = float(nonzero.mean())
            tfidf_score = max(0, 1 - mean_tfidf / 4.0)
        except Exception:
            tfidf_score = 0.5

        template_hits = sum(1 for ng in TEMPLATE_NGRAMS if ng in text)
        template_score = min(template_hits / max(1, len(TEMPLATE_NGRAMS) * 0.1), 1.0)

        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if len(lines) >= 3:
            char_repetition = 1 - len(set(text)) / max(1, len(text))
            repetition_score = min(char_repetition * 3, 1.0)
        else:
            repetition_score = 0

        score = tfidf_score * 0.45 + template_score * 0.35 + repetition_score * 0.20
        return min(score, 1.0)

    def detect_template_phrases(self, text: str) -> List[str]:
        found = []
        for ng in TEMPLATE_NGRAMS:
            if ng in text:
                found.append(ng)
        return found


# ── 混合质量评分器 ──

class TextQualityScorer:
    """
    多维度文本质量评分器（v2升级版）

    评分策略：
    1. 如果嵌入模型可用：语义ML分（60%权重） + 启发式分（40%权重）
    2. 如果不可用：纯启发式评分（原v1逻辑）
    """

    def __init__(self, boilerplate_detector=None):
        self.bp = boilerplate_detector  # TF-IDF检测器（回退用）
        self.semantic_bp = None         # 语义检测器
        self.classifier = None          # 少样本分类器
        self._init_ml_components()

        self.module_keywords = {
            "module1": ["汇率", "外汇", "展望", "风险", "发展"],
            "module2": ["衍生品", "套期保值", "远期", "期权", "掉期", "互换", "汇率", "外汇"],
            "module3": ["美元", "欧元", "港币", "日元", "外币", "币种"],
            "module4": ["记账本位币", "境外经营", "本位币", "人民币"],
            "module5": ["汇率风险", "外汇风险", "市场风险", "敏感性", "公允价值"],
            "policy_info": ["衍生品", "套期保值", "外汇", "远期结售汇", "风险管理制度", "管理办法"],
        }

    def _init_ml_components(self):
        """初始化ML组件"""
        try:
            self.semantic_bp = _get_bp_detector()
            self.classifier = _get_classifier()
            if self.semantic_bp:
                print("  [ML] 语义嵌入模型已启用（主方案）")
            if self.classifier:
                print("  [ML] 少样本原型分类器已启用")
        except Exception as e:
            print(f"  [INFO] ML组件初始化跳过: {e}，使用TF-IDF回退方案")

    def score_module(
        self, module_name: str, text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> QualityResult:
        """
        对单个模块的提取结果进行全面评分

        融合语义ML分和启发式维度分
        """
        # 空文本处理
        if not text or len(text.strip()) < 3:
            return QualityResult(
                module_name=module_name, overall_score=0,
                dimensions={"completeness": 0, "relevance": 0, "cleanliness": 100,
                           "boilerplate": 100, "informativeness": 0},
                flags=["EMPTY"], confidence="LOW",
            )

        weights = MODULE_QUALITY_WEIGHTS.get(module_name,
            {"completeness": 0.2, "relevance": 0.2, "cleanliness": 0.2,
             "boilerplate": 0.2, "informativeness": 0.2})

        # 1. 启发式维度评分（始终计算）
        dims = {
            "completeness": self._score_completeness(text, module_name),
            "relevance": self._score_relevance(text, module_name),
            "cleanliness": self._score_cleanliness(text),
            "boilerplate": self._score_boilerplate(text),
            "informativeness": self._score_informativeness(text),
        }
        heuristic_score = sum(dims[k] * weights[k] for k in weights)

        # 2. 语义ML评分（如果可用）
        ml_score = None
        ml_details = None

        if self.classifier is not None:
            try:
                # 少样本原型分类
                quality_label, ml_conf, clf_result = self.classifier.classify(
                    module_name, text
                )

                if clf_result is not None:
                    # 锅炉板分（如果有语义检测器）
                    sem_bp_score = 0.0
                    sem_quality = 50.0
                    if self.semantic_bp is not None:
                        try:
                            sem_bp_score = self.semantic_bp.score_boilerplate(text)
                            sem_quality = self.semantic_bp.score_module_quality(module_name, text)
                        except Exception:
                            pass

                    # 综合语义分：分类结果 + 置信度
                    if quality_label == "GOOD":
                        ml_score = 85 * ml_conf + 15 * 0.5  # 基础分 + 置信度加成
                    elif quality_label == "BAD":
                        ml_score = 25 * ml_conf + 10 * 0.5
                    else:
                        ml_score = 50.0

                    # 锅炉板检测调整
                    if sem_bp_score > 0.5:
                        ml_score = ml_score * 0.7 + (1 - sem_bp_score) * 100 * 0.3

                    ml_score = min(100, max(0, ml_score))

                    ml_details = {
                        "quality_label": quality_label,
                        "confidence": round(ml_conf, 3),
                        "class_scores": {
                            k: round(v, 3)
                            for k, v in clf_result.class_scores.items()
                        } if clf_result.class_scores else {},
                        "nearest_examples": clf_result.nearest_examples[:2] if clf_result.nearest_examples else [],
                    }
            except Exception:
                pass  # ML评分失败，使用启发式评分

        # 3. 融合评分
        if ml_score is not None and ml_details is not None:
            # ML分权重随置信度动态调整
            ml_weight = min(0.7, 0.35 + ml_details["confidence"] * 0.5)
            heuristic_weight = 1.0 - ml_weight
            overall = ml_score * ml_weight + heuristic_score * heuristic_weight
        else:
            overall = heuristic_score

        # 4. 收集标志
        flags = self._collect_flags(text, dims, module_name)

        # 添加ML特定标志
        if ml_details:
            if ml_details.get("quality_label") == "BAD":
                if "BAD" not in flags:
                    flags.append("ML_LOW_QUALITY")
            if ml_details.get("sem_bp_score", 0) > 0.6:
                if "HIGH_BOILERPLATE" not in flags:
                    flags.append("HIGH_BOILERPLATE")

        bp_phrases = []
        if dims["boilerplate"] < 50:
            if self.bp:
                bp_phrases = self.bp.detect_template_phrases(text)

        # 5. 确定置信度等级
        if overall >= 70:
            confidence = "HIGH"
        elif overall >= 40:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        return QualityResult(
            module_name=module_name,
            overall_score=round(overall, 1),
            dimensions={k: round(v, 1) for k, v in dims.items()},
            flags=flags,
            confidence=confidence,
            boilerplate_phrases=bp_phrases,
            ml_score=round(ml_score, 1) if ml_score is not None else None,
            heuristic_score=round(heuristic_score, 1),
            ml_details=ml_details,
        )

    # ── 启发式维度评分（保留原v1逻辑）──

    def _score_completeness(self, text: str, module_name: str) -> float:
        min_len = QUALITY_THRESHOLDS["min_text_length"].get(module_name, 10)
        if len(text.strip()) < min_len:
            return max(0, len(text.strip()) / max(1, min_len) * 100)
        score = 100.0
        truncation_patterns = QUALITY_THRESHOLDS.get("truncation_patterns", [])
        for pat in truncation_patterns:
            if re.search(pat, text[-50:]):
                score -= 30
                break
        if re.match(r'^[）\)\]\}】」]', text) or re.match(r'^[、，,]\s*', text):
            score -= 25
        if re.match(r'^.{0,5}[、，,]\s*[一-鿿]', text):
            score -= 15
        if module_name == "module3":
            if text == "无":
                score = 100
            elif len(text) < 2:
                score = max(score - 50, 0)
        return max(score, 0)

    def _score_relevance(self, text: str, module_name: str) -> float:
        keywords = self.module_keywords.get(module_name, ["汇率", "外汇"])
        kw_hits = sum(1 for kw in keywords if kw in text)
        kw_score = min(kw_hits / max(1, len(keywords) * 0.25) * 100, 100)
        if module_name == "module2":
            if "不存在衍生品投资" in text or "不适用" in text[:100]:
                return 100
        if module_name == "module5":
            has_substantive = bool(re.search(
                r'(?:汇率风险|外汇风险).{10,}(?:美元|欧元|港币|日元|人民币|敏感性|敞口)',
                text
            ))
            if not has_substantive and len(text) < 100:
                kw_score *= 0.5
        if module_name == "policy_info":
            from config import POLICY_BLACKLIST_PHRASES
            blacklisted = sum(1 for bp in POLICY_BLACKLIST_PHRASES if bp in text)
            total_entries = len(text.split(';'))
            if total_entries > 0 and blacklisted >= total_entries:
                kw_score *= 0.3
        return min(kw_score, 100)

    def _score_cleanliness(self, text: str) -> float:
        score = 100.0
        lines = text.split('\n')
        table_row_count = 0
        for line in lines:
            s = line.strip()
            if not s:
                continue
            if re.match(r'^[\w一-鿿\-–]{1,30}\s{3,}\d[\d,.]*', s):
                table_row_count += 1
            if re.match(r'^(?:项目|币种|单位)\s+(?:期末|期初|外币|人民币)', s):
                table_row_count += 2
            if re.match(r'^(?:合计|小计)\s+\d', s):
                table_row_count += 1
            digit_ratio = len(re.findall(r'[\d\s,，.￥$€%\-+\s]', s)) / max(1, len(s))
            if digit_ratio > QUALITY_THRESHOLDS["table_residue_digit_ratio"] and len(s) > 30:
                table_row_count += 1
        score -= min(table_row_count * 8, 60)
        artifact_lines = sum(1 for l in lines
                           if re.search(r'(?:股份有限公司|年度报告全文|年年度报告)', l) and len(l) < 80)
        score -= min(artifact_lines * 10, 20)
        short_lines = sum(1 for l in lines if 1 <= len(l.strip()) <= 15)
        if len(lines) > 5:
            short_ratio = short_lines / len(lines)
            if short_ratio > 0.4:
                score -= short_ratio * 30
        return max(score, 0)

    def _score_boilerplate(self, text: str) -> float:
        """锅炉板评分：综合TF-IDF和语义检测"""
        # TF-IDF评分
        if self.bp:
            bp_score = self.bp.score_boilerplate(text)
            tfidf_bp = (1 - bp_score) * 100
        else:
            tfidf_bp = 50.0

        # 语义评分（如果可用）
        if self.semantic_bp:
            try:
                sem_bp = self.semantic_bp.score_boilerplate(text)
                sem_bp_score = (1 - sem_bp) * 100
                # 加权综合
                return tfidf_bp * 0.4 + sem_bp_score * 0.6
            except Exception:
                pass

        return tfidf_bp

    def _score_informativeness(self, text: str) -> float:
        score = 100.0
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if not lines:
            return 0
        words = jieba.lcut(text)
        unique_ratio = len(set(words)) / max(1, len(words))
        if unique_ratio < 0.3:
            score -= 30
        quantitative_count = len(re.findall(r'\d[\d,.]*\s*(?:%|元|美元|欧元|人民币|港币|万)', text))
        if quantitative_count >= 2:
            score = min(score + 10, 100)
        entity_count = len(re.findall(
            r'(?:《[^》]+》|美元|欧元|港币|日元|人民币|有限公司|董事会|管理层)',
            text
        ))
        if entity_count >= 1:
            score = min(score + 5, 100)
        sent_lengths = [len(l) for l in lines]
        if len(sent_lengths) >= 3:
            mean_len = sum(sent_lengths) / len(sent_lengths)
            if mean_len < 8:
                score -= 20
            elif mean_len > 300:
                score -= 10
        return max(score, 0)

    def _collect_flags(self, text: str, dims: Dict[str, float],
                       module_name: str) -> List[str]:
        flags = []
        if dims["completeness"] < 50:
            flags.append("TRUNCATED")
        if dims["relevance"] < 40:
            flags.append("LOW_RELEVANCE")
        if dims["cleanliness"] < 50:
            flags.append("TABLE_RESIDUE")
        if dims["boilerplate"] < 40:
            flags.append("HIGH_BOILERPLATE")
        if dims["informativeness"] < 30:
            flags.append("LOW_INFO")
        if module_name == "policy_info":
            from config import POLICY_BLACKLIST_PHRASES
            entries = text.split(';')
            blacklisted = sum(1 for e in entries
                            if any(bp in e for bp in POLICY_BLACKLIST_PHRASES))
            if blacklisted > 0 and blacklisted >= len(entries) * 0.5:
                flags.append("CSRC_GENERIC_ONLY")
        if module_name == "module5":
            if len(text) < 150 and "市场风险" in text:
                flags.append("PREAMBLE_ONLY")
        if module_name == "module4":
            if re.search(r'(?:遵循企业会计准则|会计年度|营业周期|重要性标准)', text[:200]):
                flags.append("ACCOUNTING_POLICY_MIXED")
        return flags


# ── 批量评分入口 ──

def score_all_modules(results: List[Dict]) -> List[Dict]:
    """
    对整个提取结果列表进行质量评分（v2升级版）

    策略：
    1. 首先尝试使用语义嵌入+少样本分类器
    2. 回退到TF-IDF + 启发式评分
    3. 融合两种方法的分数
    """
    # 收集所有模块文本
    all_texts = []
    for record in results:
        for mod in ["module1", "module2", "module3", "module4", "module5", "policy_info"]:
            text = record.get(mod, "")
            if text and len(text) > 10:
                all_texts.append(text)

    # 初始化TF-IDF锅炉板检测器（回退方案）
    bp_detector = BoilerplateDetector()
    bp_detector.fit(all_texts)

    # 初始化/训练嵌入器并重建分类器
    try:
        embedder = _get_embedder()
        if embedder is not None and not embedder.is_ready():
            # 轻量级嵌入器需要在语料上训练
            if hasattr(embedder, 'fit_on_corpus'):
                # 扩大训练语料：加入缓存中的所有年报文本
                corpus_texts = list(all_texts)
                try:
                    import os as _os
                    cache_dir = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "cache")
                    if _os.path.exists(cache_dir):
                        for f in sorted(_os.listdir(cache_dir))[:20]:
                            if f.endswith('.txt'):
                                try:
                                    with open(_os.path.join(cache_dir, f), 'r', encoding='utf-8') as fp:
                                        raw = fp.read()
                                    # 分块加入
                                    lines = raw.split('\n')
                                    chunk = []
                                    chunk_len = 0
                                    for line in lines:
                                        s = line.strip()
                                        if not s: continue
                                        chunk.append(s)
                                        chunk_len += len(s)
                                        if chunk_len >= 300:
                                            corpus_texts.append(''.join(chunk))
                                            chunk = []
                                            chunk_len = 0
                                except Exception:
                                    pass
                except Exception:
                    pass
                embedder.fit_on_corpus(corpus_texts)

            # 重建分类器并加载历史反馈（增量学习）
            global _classifier_instance, _bp_detector_instance
            _classifier_instance = None
            _bp_detector_instance = None
            if embedder.is_ready():
                clf = _get_classifier()
                n_feedback = _load_feedback_into_classifier(clf)
                if n_feedback > 0:
                    print(f"  [ML] 从历史反馈加载了 {n_feedback} 条训练样本")
    except Exception as e:
        pass  # 静默失败，将使用TF-IDF回退

    # 初始化混合评分器
    scorer = TextQualityScorer(bp_detector)

    # 逐条评分
    for record in results:
        quality_meta = {}
        for mod in ["module1", "module2", "module3", "module4", "module5", "policy_info"]:
            text = record.get(mod, "")
            qr = scorer.score_module(mod, text)
            quality_meta[mod] = {
                "overall_score": qr.overall_score,
                "confidence": qr.confidence,
                "flags": qr.flags,
                "dimensions": qr.dimensions,
            }
            # 添加ML详情（如果有）
            if qr.ml_score is not None:
                quality_meta[mod]["ml_score"] = qr.ml_score
                quality_meta[mod]["heuristic_score"] = qr.heuristic_score
            if qr.ml_details is not None:
                quality_meta[mod]["ml_details"] = qr.ml_details

        record["_quality"] = quality_meta

        # 计算综合置信度
        scores = [m["overall_score"] for m in quality_meta.values()]
        record["_overall_quality"] = round(sum(scores) / len(scores), 1) if scores else 0

    # 打印ML状态
    ml_contributions = sum(
        1 for r in results
        for m in ["module1", "module2", "module3", "module4", "module5", "policy_info"]
        if r.get("_quality", {}).get(m, {}).get("ml_score") is not None
    )
    if ml_contributions > 0:
        print(f"  [ML] 语义嵌入评分贡献: {ml_contributions} 个模块评估")
    else:
        print(f"  [ML] 使用TF-IDF回退方案评分")

    # 自动记录高置信度GOOD/BAD样本到反馈存储（供后续运行增量学习）
    try:
        from feedback_store import FeedbackStore
        from config import FEEDBACK_STORE_PATH
        from datetime import datetime
        store = FeedbackStore(FEEDBACK_STORE_PATH)
        new_records = 0
        for record in results:
            quality = record.get("_quality", {})
            for mod in ["module1", "module2", "module3", "module4", "module5", "policy_info"]:
                text = record.get(mod, "")
                if not text or len(text) < 3:
                    continue
                q = quality.get(mod, {})
                ml_det = q.get("ml_details", {})
                label = ml_det.get("quality_label", "")
                conf = ml_det.get("confidence", 0)

                # 高置信度GOOD → 强化正样本
                if label == "GOOD" and conf > 0.8 and q.get("overall_score", 0) > 70:
                    from feedback_store import CorrectionRecord
                    cr = CorrectionRecord(
                        timestamp=datetime.now().isoformat(),
                        company_code=str(record.get("code", "")),
                        company_name=str(record.get("name", "")),
                        module_name=mod,
                        original_text=text,
                        corrected_text=text,
                        quality_before=85.0,
                        quality_after=90.0,
                        correction_type="AUTO_GOOD",
                        pattern_triggered=f"confidence={conf:.2f}",
                        original_length=len(text),
                        corrected_length=len(text),
                    )
                    store.record_correction(cr)
                    new_records += 1
        if new_records > 0:
            print(f"  [ML] 自动记录了 {new_records} 条高置信度样本供后续学习")
    except Exception:
        pass

    return results


# ── 模型管理 ──

def _load_feedback_into_classifier(classifier) -> int:
    """从反馈存储加载历史修正记录，增强分类器训练数据"""
    try:
        from feedback_store import FeedbackStore
        from config import FEEDBACK_STORE_PATH
        store = FeedbackStore(FEEDBACK_STORE_PATH)
        corrections = store.get_all_corrections()
        if not corrections:
            return 0

        # 按模块分组
        by_module = {}
        for c in corrections:
            mod = c.module_name
            if mod not in by_module:
                by_module[mod] = {"GOOD": [], "BAD": []}

            # 修正类型为锅炉板移除/表格清理/相关性过滤 → 原文是BAD
            if c.correction_type in ("BOILERPLATE_REMOVAL", "TABLE_CLEANUP",
                                      "RELEVANCE_FILTER", "TRUNCATION_FIX", "ML_LOW_QUALITY"):
                if c.original_text and len(c.original_text) > 10:
                    by_module[mod]["BAD"].append(c.original_text)

            # AUTO_GOOD：高置信度GOOD样本
            if c.correction_type == "AUTO_GOOD":
                if c.corrected_text and len(c.corrected_text) > 10:
                    by_module[mod]["GOOD"].append(c.corrected_text)

            # 修正后的文本 → GOOD
            if c.corrected_text and len(c.corrected_text) > 10:
                # 避免重复添加（AUTO_GOOD已经加过了）
                if c.correction_type != "AUTO_GOOD":
                    by_module[mod]["GOOD"].append(c.corrected_text)

                # 如果修正前后不同且质量分提高很多，原文本也是BAD
                if c.quality_before < 50 and (c.quality_after or 0) > 70:
                    if c.original_text and c.original_text != c.corrected_text:
                        by_module[mod]["BAD"].append(c.original_text)

        # 注入分类器
        total_added = 0
        for mod_name, data in by_module.items():
            if mod_name in classifier.classifiers:
                clf = classifier.classifiers[mod_name]
                good = list(set(data["GOOD"]))
                bad = list(set(data["BAD"]))
                if good or bad:
                    classifier.add_examples_from_feedback(mod_name, good, bad)
                    total_added += len(good) + len(bad)

        if total_added > 0:
            print(f"  [ML] 从反馈历史加载了 {total_added} 条修正记录，增强训练数据")
        return total_added
    except Exception as e:
        return 0


def save_ml_models(path: str):
    """保存所有ML模型状态"""
    clf = _get_classifier()
    if clf is not None:
        clf.save(path)
        print(f"  [ML] 模型已保存到: {path}")

    embedder = _get_embedder()
    if embedder is not None:
        import os
        cache_path = os.path.join(path, "embedding_cache.pkl")
        embedder.save_cache(cache_path)


def load_ml_models(path: str) -> bool:
    """加载ML模型状态"""
    global _classifier_instance
    import os

    embedder = _get_embedder()
    if embedder is None:
        return False

    # 加载嵌入缓存
    cache_path = os.path.join(path, "embedding_cache.pkl")
    if os.path.exists(cache_path):
        embedder.load_cache(cache_path)

    # 加载分类器
    if _HAS_CLASSIFIER and embedder.is_ready():
        try:
            _classifier_instance = _MultiModuleClassifier.load(path, embedder)
            print(f"  [ML] 已加载分类器模型")
            return True
        except Exception as e:
            print(f"  [WARN] 加载分类器失败: {e}")

    return False


def add_training_examples(
    module_name: str, good_texts: List[str], bad_texts: List[str]
):
    """增量添加训练样本"""
    clf = _get_classifier()
    if clf is not None:
        clf.add_examples_from_feedback(module_name, good_texts, bad_texts)


# ── 测试 ──

if __name__ == "__main__":
    print("=== ML质量评分引擎 v2 测试 ===\n")

    test_texts = [
        "4、记账本位币人民币为本公司及境内子公司经营所处的主要经济环境中的货币，本公司及子公司均以人民币为记账本位币。",
        "1、遵循企业会计准则的声明本财务报表符合财政部颁布的企业会计准则的要求。2、会计期间会计年度自公历1月1日起至12月31日止。3、营业周期正常营业周期是指本公司从购买用于加工的资产起至实现现金或现金等价物的期间。4、记账本位币本公司采用人民币为记账本位币。",
        "公司报告期不存在衍生品投资。",
        "《企业内部控制基本规范》; 《2023年度内部控制自我评价报告》",
    ]

    # TF-IDF回退测试
    print("--- TF-IDF回退方案 ---")
    bp = BoilerplateDetector()
    bp.fit(test_texts)
    for t in test_texts:
        score = bp.score_boilerplate(t)
        phrases = bp.detect_template_phrases(t)
        print(f"  BP={score:.2f} | phrases={phrases} | text={t[:80]}...")

    print("\n--- 混合评分器 ---")
    scorer = TextQualityScorer(bp)
    for i, t in enumerate(test_texts):
        mod = ["module4", "module4", "module2", "policy_info"][i]
        qr = scorer.score_module(mod, t)
        print(f"  {mod}: score={qr.overall_score} conf={qr.confidence} "
              f"ml={qr.ml_score} heur={qr.heuristic_score} flags={qr.flags}")
