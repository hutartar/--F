"""
轻量级文本嵌入回退方案（v1）：基于sklearn TF-IDF + SVD的嵌入
当sentence-transformers不可用时的纯CPU方案

优势：
- 零额外依赖（仅需sklearn + jieba）
- 快速训练和推理
- 对中文年报领域自适应（在corpus上训练TF-IDF）

劣势：
- 语义理解能力弱于预训练模型
- 需要足够语料来建立有意义的TF-IDF空间
"""
import numpy as np
from typing import List, Dict, Optional, Union
import pickle
import os
import hashlib


class LightweightChineseEmbedder:
    """
    轻量级中文文本嵌入器（无需sentence-transformers）

    使用方案：
    1. 在年报语料上训练TF-IDF + TruncatedSVD
    2. 将文本映射到低维稠密向量
    3. 余弦相似度计算
    """

    def __init__(
        self,
        svd_components: int = 256,
        ngram_range: tuple = (1, 3),
        max_features: int = 8000,
    ):
        self.svd_components = svd_components
        self.ngram_range = ngram_range
        self.max_features = max_features

        self.vectorizer = None
        self.svd = None
        self._ready = False
        self._corpus_fitted = False

        # 嵌入缓存
        self._cache: Dict[str, np.ndarray] = {}

    def is_ready(self) -> bool:
        return self._ready or self._corpus_fitted

    def _tokenize(self, text: str) -> List[str]:
        """中文分词"""
        import jieba
        words = jieba.lcut(text)
        return [w.strip() for w in words if len(w.strip()) >= 1]

    def _cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def fit_on_corpus(self, documents: List[str]):
        """在年报语料上训练TF-IDF + SVD"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        docs = [d for d in documents if d and len(d) > 10]
        if not docs:
            print("  [WARN] 无语料可训练")
            return

        print(f"  [Lightweight] 在 {len(docs)} 篇文档上训练TF-IDF...")

        self.vectorizer = TfidfVectorizer(
            tokenizer=self._tokenize,
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=2,
            max_df=0.85,
            token_pattern=None,
        )

        tfidf_matrix = self.vectorizer.fit_transform(docs)
        actual_features = tfidf_matrix.shape[1]
        n_components = min(self.svd_components, actual_features - 1, len(docs) - 1)

        if n_components > 1:
            print(f"  [Lightweight] SVD降维: {actual_features} → {n_components}")
            self.svd = TruncatedSVD(n_components=n_components, random_state=42)
            self.svd.fit(tfidf_matrix)
        else:
            print(f"  [Lightweight] 特征太少({actual_features})，跳过SVD")

        self._corpus_fitted = True
        self._ready = True

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        将文本转换为嵌入向量
        """
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        vectors = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            if not text:
                vectors.append((i, np.zeros(self.svd_components if self.svd else 100)))
                continue

            key = self._cache_key(text)
            if key in self._cache:
                vectors.append((i, self._cache[key]))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        if uncached_texts and self.vectorizer:
            # TF-IDF变换
            tfidf_vecs = self.vectorizer.transform(uncached_texts)

            # SVD降维
            if self.svd:
                dense_vecs = self.svd.transform(tfidf_vecs)
            else:
                dense_vecs = tfidf_vecs.toarray()

            # L2归一化
            norms = np.linalg.norm(dense_vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1
            dense_vecs = dense_vecs / norms

            for idx, vec, text in zip(uncached_indices, dense_vecs, uncached_texts):
                key = self._cache_key(text)
                self._cache[key] = vec
                vectors.append((idx, vec))

        # 按原始顺序排列
        vectors.sort(key=lambda x: x[0])
        result = np.stack([v[1] for v in vectors])

        return result[0] if single_input else result

    def similarity(self, text1: str, text2: str) -> float:
        """余弦相似度"""
        if not text1 or not text2:
            return 0.0
        vecs = self.embed([text1, text2])
        sim = float(np.dot(vecs[0], vecs[1]))
        return max(0.0, min(1.0, sim))

    def batch_similarity(self, query: str, candidates: List[str]) -> np.ndarray:
        """批量相似度"""
        if not query or not candidates:
            return np.zeros(len(candidates))
        all_texts = [query] + list(candidates)
        vecs = self.embed(all_texts)
        query_vec = vecs[0]
        candidate_vecs = vecs[1:]
        sims = np.dot(candidate_vecs, query_vec)
        return np.clip(sims, 0.0, 1.0)

    def compute_prototype(
        self, texts: List[str], method: str = "mean"
    ) -> np.ndarray:
        """计算原型向量"""
        if not texts:
            raise ValueError("texts 不能为空")
        vecs = self.embed(texts)
        if method == "mean":
            proto = np.mean(vecs, axis=0)
        else:
            proto = np.median(vecs, axis=0)
        norm = np.linalg.norm(proto)
        if norm > 0:
            proto = proto / norm
        return proto

    def classify_by_prototype(
        self, text: str, prototypes: Dict[str, np.ndarray]
    ):
        """基于原型分类"""
        vec = self.embed(text)
        sims = {}
        for cls_name, proto in prototypes.items():
            sims[cls_name] = float(np.dot(vec, proto))
        best_cls = max(sims, key=sims.get)
        best_sim = sims[best_cls]
        sorted_sims = sorted(sims.values(), reverse=True)
        margin = sorted_sims[0] - sorted_sims[1] if len(sorted_sims) > 1 else sorted_sims[0]
        confidence = float(np.clip(margin * 2.0, 0.0, 1.0))
        return best_cls, confidence, sims

    def save(self, path: str):
        """保存模型"""
        import joblib
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_data = {
            "vectorizer": self.vectorizer,
            "svd": self.svd,
            "svd_components": self.svd_components,
        }
        joblib.dump(model_data, path)

    def load(self, path: str):
        """加载模型"""
        import joblib
        model_data = joblib.load(path)
        self.vectorizer = model_data["vectorizer"]
        self.svd = model_data["svd"]
        self.svd_components = model_data.get("svd_components", 256)
        self._corpus_fitted = True
        self._ready = True


if __name__ == "__main__":
    print("=== LightweightChineseEmbedder 测试 ===\n")

    corpus = [
        "汇率波动风险：公司出口业务主要以美元结算，面临较大的汇率波动风险。为应对汇率风险，公司采用远期结售汇锁定汇率。",
        "公司报告期不存在衍生品投资。",
        "4、记账本位币人民币为本公司及境内子公司经营所处的主要经济环境中的货币，本公司及境内子公司以人民币为记账本位币。",
        "1、遵循企业会计准则的声明本财务报表符合财政部颁布的企业会计准则的要求，真实、完整地反映了公司的财务状况。",
        "2、会计期间会计年度自公历1月1日起至12月31日止。",
        "3、营业周期正常营业周期是指本公司从购买用于加工的资产起至实现现金或现金等价物的期间。",
        "（1）市场风险金融工具的市场风险，是指金融工具的公允价值或未来现金流量因市场价格变动而发生波动的风险，包括汇率风险、利率风险和其他价格风险。",
        "1）汇率风险本集团承受汇率风险主要与美元有关，除本集团及下属进出口公司部分以美元进行采购和销售外，本集团的其它主要业务活动以人民币计价结算。",
    ]

    embedder = LightweightChineseEmbedder(svd_components=64)
    embedder.fit_on_corpus(corpus)

    t1 = "汇率风险：公司面临美元汇率波动。"
    t2 = "会计政策要求财务报表符合准则。"
    t3 = "美元外汇远期合约套期保值。"

    sim_1_2 = embedder.similarity(t1, t2)
    sim_1_3 = embedder.similarity(t1, t3)
    print(f"相似度(汇率 vs 会计): {sim_1_2:.3f}")
    print(f"相似度(汇率 vs 外汇): {sim_1_3:.3f}")
    print(f"区分度: {sim_1_3 - sim_1_2:.3f} (正值=正确区分)")

    # 原型测试
    good_texts = [corpus[0], t1, t3]
    bad_texts = [corpus[3], corpus[4], corpus[5], t2]
    proto_good = embedder.compute_prototype(good_texts)
    proto_bad = embedder.compute_prototype(bad_texts)

    test = "汇率风险主要来源于美元结算的出口业务。"
    cls, conf, sims = embedder.classify_by_prototype(
        test, {"GOOD": proto_good, "BAD": proto_bad}
    )
    print(f"\n测试: '{test}'")
    print(f"分类: {cls}, 置信度: {conf:.3f}, 分数: {sims}")
