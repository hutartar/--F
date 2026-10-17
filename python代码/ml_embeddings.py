"""
中文金融文本嵌入模型（v1）：基于Sentence-Transformers的语义向量化
用于替代TF-IDF，实现语义级别的文本质量评估和相似度计算

支持模型：
- shibing624/text2vec-base-chinese (轻量中文专用，推荐)
- BAAI/bge-small-zh-v1.5 (中文SOTA，精度更高)
- paraphrase-multilingual-MiniLM-L12-v2 (多语言备选)
"""
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import os
import pickle
import hashlib
import json


@dataclass
class EmbeddingCache:
    """嵌入向量缓存，避免重复计算"""
    cache: Dict[str, np.ndarray]

    def __init__(self):
        self.cache = {}

    def _key(self, text: str) -> str:
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def get(self, text: str) -> Optional[np.ndarray]:
        return self.cache.get(self._key(text))

    def put(self, text: str, vec: np.ndarray):
        self.cache[self._key(text)] = vec

    def size(self) -> int:
        return len(self.cache)

    def clear(self):
        self.cache.clear()


class ChineseTextEmbedder:
    """
    中文文本嵌入器：使用预训练Sentence-Transformer模型
    将中文年报文本转换为稠密语义向量
    """

    def __init__(
        self,
        model_name: str = "shibing624/text2vec-base-chinese",
        device: str = "cpu",
        cache_embeddings: bool = True,
        max_seq_length: int = 512,
        batch_size: int = 32,
    ):
        """
        Args:
            model_name: HuggingFace模型名称
            device: "cpu" 或 "cuda"
            cache_embeddings: 是否缓存嵌入向量
            max_seq_length: 最大序列长度（超过会截断）
            batch_size: 批量编码大小
        """
        self.model_name = model_name
        self.device = device
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.cache = EmbeddingCache() if cache_embeddings else None
        self._model = None
        self._ready = False
        self._using_lightweight = False
        self._lightweight = None

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self):
        """延迟加载模型"""
        try:
            from sentence_transformers import SentenceTransformer
            print(f"  [ML] 加载嵌入模型: {self.model_name} ...")
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
            )
            # 设置最大序列长度
            if hasattr(self._model, 'max_seq_length'):
                self._model.max_seq_length = self.max_seq_length
            self._ready = True
            print(f"  [ML] 模型加载完成，维度: {self._model.get_sentence_embedding_dimension()}")
        except ImportError:
            print("  [ERROR] sentence-transformers 未安装。运行: pip install sentence-transformers")
            raise
        except Exception as e:
            print(f"  [WARN] 模型 {self.model_name} 加载失败: {e}")
            print(f"  [INFO] 尝试回退到轻量模型...")
            self._load_fallback()

    def _load_fallback(self):
        """回退到最轻量的模型或多语言模型"""
        # 首先尝试多语言MiniLM（更兼容）
        try:
            from sentence_transformers import SentenceTransformer
            # 尝试显式设置缓存文件夹
            import os as _os
            _cache = _os.path.join(_os.path.expanduser("~"), ".cache", "huggingface")
            self._model = SentenceTransformer(
                "paraphrase-multilingual-MiniLM-L12-v2",
                device=self.device,
                cache_folder=_cache,
            )
            self._ready = True
            print(f"  [ML] 回退模型加载完成")
        except Exception:
            # 最后回退到轻量级TF-IDF+SVD方案
            print(f"  [INFO] 所有预训练模型加载失败，启用轻量级TF-IDF+SVD嵌入")
            self._load_lightweight_fallback()

    def _load_lightweight_fallback(self):
        """使用纯sklearn的轻量级嵌入"""
        try:
            from lightweight_embedder import LightweightChineseEmbedder
            self._lightweight = LightweightChineseEmbedder(
                svd_components=256,
                max_features=8000,
            )
            self._ready = False  # 需要先fit
            self._using_lightweight = True
        except Exception as e:
            print(f"  [ERROR] 轻量级嵌入也加载失败: {e}")
            self._ready = False
            raise

    def is_ready(self) -> bool:
        if self._using_lightweight and self._lightweight:
            return self._lightweight.is_ready()
        return self._ready

    def fit_on_corpus(self, documents: List[str]):
        """在语料上训练轻量级嵌入模型（仅轻量级方案需要）"""
        if self._using_lightweight and self._lightweight:
            self._lightweight.fit_on_corpus(documents)
            self._ready = True

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        将文本转换为嵌入向量

        Args:
            texts: 单个文本字符串或文本列表

        Returns:
            numpy数组，shape为 (n_texts, embedding_dim)
        """
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        # 轻量级方案
        if self._using_lightweight and self._lightweight:
            return self._lightweight.embed(texts)

        # 从缓存中获取
        if self.cache:
            vectors = []
            uncached_texts = []
            uncached_indices = []
            for i, text in enumerate(texts):
                vec = self.cache.get(text) if text else None
                if vec is not None:
                    vectors.append((i, vec))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
            vectors = []

        # 编码未缓存的文本
        if uncached_texts:
            new_vectors = self.model.encode(
                uncached_texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,  # L2归一化，便于余弦相似度计算
            )
            # 存入缓存
            if self.cache:
                for text, vec in zip(uncached_texts, new_vectors):
                    self.cache.put(text, vec)
            for idx, vec in zip(uncached_indices, new_vectors):
                vectors.append((idx, vec))

        # 按原始顺序排列
        vectors.sort(key=lambda x: x[0])
        result = np.stack([v[1] for v in vectors])

        return result[0] if single_input else result

    def similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的余弦相似度 (0-1)"""
        if not text1 or not text2:
            return 0.0
        vecs = self.embed([text1, text2])
        sim = float(np.dot(vecs[0], vecs[1]))  # L2归一化后点积=余弦相似度
        return max(0.0, min(1.0, sim))

    def batch_similarity(self, query: str, candidates: List[str]) -> np.ndarray:
        """计算查询文本与多个候选文本的相似度"""
        if not query or not candidates:
            return np.zeros(len(candidates))
        all_texts = [query] + list(candidates)
        vecs = self.embed(all_texts)
        query_vec = vecs[0]
        candidate_vecs = vecs[1:]
        sims = np.dot(candidate_vecs, query_vec)
        return np.clip(sims, 0.0, 1.0)

    def find_most_similar(
        self, query: str, candidates: List[str], top_k: int = 5
    ) -> List[Tuple[int, float, str]]:
        """在候选文本中找到与查询最相似的top_k个"""
        sims = self.batch_similarity(query, candidates)
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(int(i), float(sims[i]), candidates[i]) for i in top_indices]

    def compute_prototype(
        self, texts: List[str], method: str = "mean"
    ) -> np.ndarray:
        """
        计算一组文本的原型向量（类中心）

        Args:
            texts: 同类文本列表
            method: "mean"（均值）或 "median"（中位数）

        Returns:
            原型向量
        """
        if not texts:
            raise ValueError("texts 不能为空")
        vecs = self.embed(texts)
        if method == "mean":
            proto = np.mean(vecs, axis=0)
        elif method == "median":
            proto = np.median(vecs, axis=0)
        else:
            raise ValueError(f"未知方法: {method}")
        # L2归一化
        norm = np.linalg.norm(proto)
        if norm > 0:
            proto = proto / norm
        return proto

    def classify_by_prototype(
        self, text: str, prototypes: Dict[str, np.ndarray]
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        基于原型向量的分类

        Args:
            text: 待分类文本
            prototypes: {类别名: 原型向量}

        Returns:
            (预测类别, 置信度, {类别名: 相似度})
        """
        vec = self.embed(text)
        sims = {}
        for cls_name, proto in prototypes.items():
            sims[cls_name] = float(np.dot(vec, proto))

        best_cls = max(sims, key=sims.get)
        best_sim = sims[best_cls]

        # 置信度：最佳相似度减去次佳相似度（越大越确定）
        sorted_sims = sorted(sims.values(), reverse=True)
        margin = sorted_sims[0] - sorted_sims[1] if len(sorted_sims) > 1 else sorted_sims[0]
        confidence = float(np.clip(margin * 2.0, 0.0, 1.0))

        return best_cls, confidence, sims

    def compute_coherence(self, texts: List[str]) -> float:
        """
        计算一组文本的语义一致性（类内内聚度）
        高值 → 文本语义一致；低值 → 文本语义分散（可能是噪声）
        """
        if len(texts) < 2:
            return 1.0
        vecs = self.embed(texts)
        # 平均余弦相似度
        sim_matrix = np.dot(vecs, vecs.T)
        n = len(vecs)
        # 排除对角线
        total_sim = sim_matrix.sum() - np.trace(sim_matrix)
        n_pairs = n * (n - 1)
        return float(total_sim / max(1, n_pairs))

    def save_cache(self, path: str):
        """保存嵌入缓存到磁盘"""
        if self.cache and self.cache.size() > 0:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump(self.cache.cache, f)
            print(f"  [ML] 嵌入缓存已保存: {path} ({self.cache.size()} 条)")

    def load_cache(self, path: str):
        """从磁盘加载嵌入缓存"""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.cache.cache = pickle.load(f)
            print(f"  [ML] 嵌入缓存已加载: {path} ({self.cache.size()} 条)")


class SemanticBoilerplateDetector:
    """
    基于语义嵌入的锅炉板/模板文本检测器（升级版，替代TF-IDF方案）

    核心思路：
    1. 用人工标注+自动收集构建"锅炉板文本"原型
    2. 对任意文本计算与锅炉板原型的语义相似度
    3. 高相似度 → 可能是锅炉板
    """

    def __init__(self, embedder: ChineseTextEmbedder):
        self.embedder = embedder
        self.boilerplate_prototypes: Dict[str, np.ndarray] = {}
        self.module_specific_prototypes: Dict[str, Dict[str, np.ndarray]] = {}
        self._built = False

    def build_prototypes(self, boilerplate_texts: Dict[str, List[str]]):
        """
        构建锅炉板原型

        Args:
            boilerplate_texts: {
                "通用": ["模板文字1", "模板文字2", ...],
                "会计政策类": ["遵循企业会计准则...", ...],
                "CSRC规范类": ["企业内部控制基本规范...", ...],
                ...
            }
        """
        for category, texts in boilerplate_texts.items():
            if texts:
                self.boilerplate_prototypes[category] = (
                    self.embedder.compute_prototype(texts)
                )
        self._built = True
        print(f"  [ML] 构建了 {len(self.boilerplate_prototypes)} 个锅炉板原型")

    def build_module_prototypes(
        self, module_name: str, good_texts: List[str], bad_texts: List[str]
    ):
        """为特定模块构建好/坏文本原型"""
        prototypes = {}
        if good_texts:
            prototypes["GOOD"] = self.embedder.compute_prototype(good_texts)
        if bad_texts:
            prototypes["BAD"] = self.embedder.compute_prototype(bad_texts)
        self.module_specific_prototypes[module_name] = prototypes

    def score_boilerplate(self, text: str) -> float:
        """
        返回0.0-1.0的锅炉板可能性评分
        1.0 = 肯定是模板/锅炉板文本
        """
        if not text or len(text) < 5:
            return 1.0

        if not self._built:
            return 0.5  # 未构建原型时返回中性值

        # 计算与所有锅炉板原型的最大相似度
        max_sim = 0.0
        for category, proto in self.boilerplate_prototypes.items():
            vec = self.embedder.embed(text)
            sim = float(np.dot(vec, proto))
            max_sim = max(max_sim, sim)

        # 调整：模板文本通常在0.6-0.95范围
        # 低于0.4的基本不是模板
        if max_sim < 0.35:
            return max_sim * 0.5
        elif max_sim < 0.55:
            return max_sim * 0.8
        else:
            return min(max_sim, 1.0)

    def score_module_quality(self, module_name: str, text: str) -> float:
        """
        基于模块特定的好/坏原型评估文本质量
        返回0-100的质量分
        """
        prototypes = self.module_specific_prototypes.get(module_name, {})
        if not prototypes or not text:
            return 50.0  # 无原型时返回中性分

        vec = self.embedder.embed(text)

        if "GOOD" in prototypes and "BAD" in prototypes:
            good_sim = float(np.dot(vec, prototypes["GOOD"]))
            bad_sim = float(np.dot(vec, prototypes["BAD"]))
            # 质量分：与GOOD相似减去与BAD相似
            raw_score = (good_sim - bad_sim + 1.0) / 2.0
            return round(raw_score * 100, 1)
        elif "GOOD" in prototypes:
            good_sim = float(np.dot(vec, prototypes["GOOD"]))
            return round(good_sim * 100, 1)
        elif "BAD" in prototypes:
            bad_sim = float(np.dot(vec, prototypes["BAD"]))
            return round((1.0 - bad_sim) * 100, 1)
        return 50.0

    def is_likely_boilerplate(self, text: str, threshold: float = 0.6) -> bool:
        """判断文本是否可能是锅炉板"""
        return self.score_boilerplate(text) >= threshold


# ── 预定义的锅炉板文本库（覆盖年报常见模板）──
PREDEFINED_BOILERPLATE = {
    "会计政策通用": [
        "1、遵循企业会计准则的声明本财务报表符合财政部颁布的企业会计准则的要求，真实、完整地反映了公司的财务状况、经营成果和现金流量等有关信息。",
        "2、会计期间会计年度自公历1月1日起至12月31日止。",
        "3、营业周期正常营业周期是指本公司从购买用于加工的资产起至实现现金或现金等价物的期间。",
        "重要性标准确定方法和选择依据",
        "本公司以人民币为记账本位币。",
        "本公司及境内子公司以人民币为记账本位币。",
        "除有特别说明外，均以人民币元为单位表示",
        "本公司的主要经营位于中国境内，主要业务以人民币结算。",
    ],
    "CSRC规范文档": [
        "企业内部控制基本规范",
        "企业内部控制应用指引",
        "企业内部控制评价指引",
        "上市公司内部控制指引",
        "上市公司治理准则",
        "公开发行证券的公司信息披露内容与格式准则",
        "公开发行证券的公司信息披露编报规则",
        "证券交易所股票上市规则",
    ],
    "模板指令语": [
        "包括对于重要的境外经营实体，应披露其境外主要经营地、记账本位币及选择依据，记账本位币发生变化的还应披露原因。",
        "应披露其境外主要经营地",
        "发生变化的还应披露原因",
        "报告期内以套期保值为目的的衍生品投资",
        "独立董事对公司衍生品投资及风险控制情况的专项意见",
    ],
    "通用报告文字": [
        "本公司董事会及全体董事保证本公告内容不存在任何虚假记载、误导性陈述或者重大遗漏",
        "并对其内容的真实性、准确性和完整性承担法律责任",
        "股份有限公司2023年年度报告全文",
        "年年度报告全文",
    ],
}

# 各模块的质量参考文本（好/坏示例）
MODULE_REFERENCE_TEXTS = {
    "module1": {
        "GOOD": [
            "汇率波动风险：公司出口业务主要以美元结算，人民币对美元汇率波动将对公司出口产品的价格竞争力及汇兑损益产生影响。为应对汇率风险，公司将积极关注汇率变动趋势，适时采用远期结售汇等金融工具锁定汇率。",
            "外汇风险：本集团面临的外汇风险主要来自以美元、欧元计价的出口销售。本集团通过密切关注汇率走势，灵活调整销售结算币种，并适时开展远期外汇交易等方式管理外汇风险敞口。",
            "（2）汇率波动风险公司境外销售收入占比较高，汇率波动对公司经营业绩有较大影响。2023年人民币汇率呈现双向波动，公司通过以下措施管理汇率风险：一是优化外币资产负债结构；二是适时运用外汇衍生品进行套期保值。",
        ],
        "BAD": [
            "公司遵循企业会计准则，真实、完整地反映了公司的财务状况。2023年度公司实现营业收入稳定增长。",
            "公司不存在对生产经营产生重大影响的汇率风险。公司主要业务在境内开展。",
            "详见本报告第十节财务报告之与金融工具相关的风险。",
        ],
    },
    "module2": {
        "GOOD": [
            "1）报告期内以套期保值为目的的衍生品投资：适用衍生品投资类型：远期外汇合约。报告期实际损益情况的说明：2023年度公司远期结售汇合约产生投资收益285万元。套期保值效果的说明：公司通过远期结售汇锁定汇率，有效降低了汇率波动对出口业务利润的影响。",
            "衍生品投资类型：远期结售汇、外汇期权。套期保值效果的说明：报告期内公司外汇衍生品交易均以套期保值为目的，与公司业务规模相匹配，有效规避了汇率波动风险。",
            "公司报告期不存在衍生品投资。",
            "公司报告期不存在衍生品投资。以套期保值为目的的衍生品投资：不适用。",
        ],
        "BAD": [
            "期货套期保值业务：公司开展铜、铝期货套期保值业务，以规避原材料价格波动风险。",
            "商品期货投资：公司参与螺纹钢、铁矿石期货交易。",
            "股票期权激励计划：公司实施限制性股票激励计划。",
        ],
    },
    "module3": {
        "GOOD": [
            "美元、欧元、港币",
            "美元、日元",
            "美元",
            "美元、欧元、港币、日元、韩元、英镑",
            "无",
        ],
        "BAD": [
            "项目 期末余额 期初余额",
            "货币资金 应收账款 应付账款",
            "合计 小计 总计",
        ],
    },
    "module4": {
        "GOOD": [
            "4、记账本位币人民币为本公司及境内子公司经营所处的主要经济环境中的货币，本公司及境内子公司以人民币为记账本位币。境外子公司XX公司以美元为记账本位币。",
            "境外经营实体说明：本公司之子公司XX有限公司，其主要经营地在香港，以港币为记账本位币。",
            "4、记账本位币本公司及境内子公司以人民币为记账本位币。境外子公司以先令为记账本位币。",
            "本公司以人民币为记账本位币。境外子公司XX有限公司以美元为记账本位币。",
        ],
        "BAD": [
            "4、记账本位币1、遵循企业会计准则的声明本财务报表符合企业会计准则的要求。2、会计期间自公历1月1日起至12月31日。3、营业周期12个月。4、记账本位币本公司以人民币为记账本位币。",
            "包括对于重要的境外经营实体，应披露其境外主要经营地、记账本位币及选择依据，记账本位币发生变化的还应披露原因。不适用",
        ],
    },
    "module5": {
        "GOOD": [
            "1）汇率风险本公司面临的汇率风险主要来源于以美元计价的金融资产和金融负债。于2023年12月31日，本公司持有美元金融资产折合人民币5,280万元，美元金融负债折合人民币3,150万元。公司通过外汇资金集中管理、外币资产与外币负债匹配等方式管理汇率风险敞口。",
            "1）汇率风险本集团承受汇率风险主要与美元有关，外币金融资产包括美元存款及应收账款。汇率敏感性分析：若人民币对美元升值1%，将导致本集团税前利润减少约85万元。",
            "（1）市场风险\n1）汇率风险\n本集团承受的汇率风险主要来源于以美元、欧元计价的出口销售。于资产负债表日，本集团持有的外币货币性项目如下：美元应收账款1,200万元，欧元银行存款850万元。本集团通过密切监控汇率变动，适时运用远期外汇合约管理汇率风险。",
        ],
        "BAD": [
            "（1）市场风险金融工具的市场风险，是指金融工具的公允价值或未来现金流量因市场价格变动而发生波动的风险，包括汇率风险、利率风险和其他价格风险。",
            "金融工具风险包括市场风险、信用风险和流动性风险。本公司的主要经营位于中国境内，主要业务以人民币结算，但已确认的外币资产和负债及未来的外币交易依然存在外汇风险。",
            "利率风险利率风险，是指金融工具的公允价值或未来现金流量因市场利率变动而发生波动的风险。信用风险信用风险，是指金融工具的一方不履行义务，造成另一方发生财务损失的风险。",
        ],
    },
    "policy_info": {
        "GOOD": [
            "《远期结售汇业务管理制度》",
            "《外汇风险管理办法》",
            "《金融衍生品交易管理制度》",
            "《套期保值业务管理办法》",
            "《外汇套期保值业务管理制度》",
            "《远期外汇交易业务管理制度》",
            "《外汇衍生品交易业务管理制度》",
        ],
        "BAD": [
            "《企业内部控制基本规范》",
            "《企业内部控制应用指引》",
            "《企业内部控制评价指引》",
            "《上市公司内部控制指引》",
            "《上市公司治理准则》",
            "《公开发行证券的公司信息披露内容与格式准则第2号》",
        ],
    },
}


if __name__ == "__main__":
    print("=== ChineseTextEmbedder 测试 ===")
    embedder = ChineseTextEmbedder()

    texts = [
        "汇率波动风险：公司出口业务主要以美元结算，人民币对美元汇率波动将对公司出口产品的价格竞争力产生影响。",
        "公司报告期不存在衍生品投资。",
        "4、记账本位币人民币为本公司及境内子公司以人民币为记账本位币。",
        "1、遵循企业会计准则的声明本财务报表符合企业会计准则的要求。",
    ]

    vecs = embedder.embed(texts)
    print(f"嵌入维度: {vecs.shape}")

    # 相似度测试
    sim = embedder.similarity(texts[0], texts[3])
    print(f"相似度(外汇风险 vs 会计准则): {sim:.3f}")

    sim2 = embedder.similarity(texts[0], "外汇风险：美元汇率波动影响公司利润。")
    print(f"相似度(外汇风险 vs 外汇风险): {sim2:.3f}")

    # 原型分类测试
    print("\n=== SemanticBoilerplateDetector 测试 ===")
    bp = SemanticBoilerplateDetector(embedder)
    bp.build_prototypes(PREDEFINED_BOILERPLATE)

    for t in texts:
        score = bp.score_boilerplate(t)
        print(f"  锅炉板分={score:.2f} | {t[:60]}...")

    # 模块质量测试
    for mod_name in ["module4", "module5"]:
        ref = MODULE_REFERENCE_TEXTS.get(mod_name, {})
        if ref:
            bp.build_module_prototypes(mod_name, ref.get("GOOD", []), ref.get("BAD", []))
            good_score = bp.score_module_quality(mod_name, texts[2])
            bad_score = bp.score_module_quality(mod_name, texts[3])
            print(f"  {mod_name}: good_text质量分={good_score}, bad_text质量分={bad_score}")
