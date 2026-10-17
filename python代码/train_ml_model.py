"""
ML模型训练脚本（v1）：小样本环境下训练语义质量分类器

工作流程：
1. 从参考文本库构建初始训练集
2. 数据增强扩充训练样本
3. 训练少样本原型分类器
4. 交叉验证评估
5. 保存模型供主流程使用

使用方法：
    python train_ml_model.py                  # 基础训练
    python train_ml_model.py --with-augment   # 含数据增强
    python train_ml_model.py --from-feedback  # 从反馈数据学习
    python train_ml_model.py --evaluate       # 评估现有模型
"""
import os
import sys
import json
import argparse
from datetime import datetime
from collections import Counter
from typing import Dict

# 添加代码目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import OUTPUT_DIR, FEEDBACK_STORE_PATH
from ml_embeddings import (
    ChineseTextEmbedder,
    SemanticBoilerplateDetector,
    MODULE_REFERENCE_TEXTS,
    PREDEFINED_BOILERPLATE,
)
from few_shot_classifier import MultiModuleClassifier, PrototypeClassifier
from data_augmentation import ChineseFinancialTextAugmenter, TemplateGenerator
from feedback_store import FeedbackStore


def build_training_data_from_references():
    """从预定义的参考文本构建训练数据"""
    train_data = {}
    for mod_name, refs in MODULE_REFERENCE_TEXTS.items():
        good = refs.get("GOOD", [])
        bad = refs.get("BAD", [])
        train_data[mod_name] = {
            "texts": good + bad,
            "labels": ["GOOD"] * len(good) + ["BAD"] * len(bad),
        }
    return train_data


def build_training_data_from_feedback(feedback_store: FeedbackStore):
    """
    从反馈存储中提取训练数据

    将人工修正记录转换为训练样本：
    - 原本低质量→修正后→原始文本作为BAD样本
    - 原本被标记为高质量→作为GOOD样本
    """
    records = feedback_store.get_all_corrections()
    if not records:
        return {}

    train_data = {}
    for mod_name in ["module1", "module2", "module3", "module4", "module5", "policy_info"]:
        good_texts = []
        bad_texts = []

        for rec in records:
            if rec.module_name != mod_name:
                continue

            # 修正类型为BOILERPLATE_REMOVAL或TABLE_CLEANUP → 原文是BAD
            if rec.correction_type in ("BOILERPLATE_REMOVAL", "TABLE_CLEANUP",
                                        "RELEVANCE_FILTER", "TRUNCATION_FIX"):
                if rec.original_text and len(rec.original_text) > 10:
                    bad_texts.append(rec.original_text)

            # 修正后的文本（如果不是空） → 可能是GOOD
            if rec.corrected_text and len(rec.corrected_text) > 10:
                good_texts.append(rec.corrected_text)

            # 质量高的原始文本
            if rec.quality_before >= 75 and rec.original_text:
                if rec.correction_type == "OTHER":
                    good_texts.append(rec.original_text)

        if good_texts or bad_texts:
            # 去重
            good_texts = list(set(good_texts))
            bad_texts = list(set(bad_texts))
            train_data[mod_name] = {
                "texts": good_texts + bad_texts,
                "labels": ["GOOD"] * len(good_texts) + ["BAD"] * len(bad_texts),
            }

    return train_data


def augment_training_data(
    train_data: Dict, augmentation_factor: int = 3
) -> Dict:
    """对训练数据进行增强"""
    augmenter = ChineseFinancialTextAugmenter(seed=42)
    template_gen = TemplateGenerator()

    augmented = {}
    for mod_name, data in train_data.items():
        texts = data["texts"]
        labels = data["labels"]

        if not texts:
            augmented[mod_name] = data
            continue

        # 文本增强
        aug_texts, aug_labels = augmenter.augment_batch(
            texts, labels, augmentation_factor=augmentation_factor
        )

        # 为BAD较少的模块添加模板生成的负面样本
        bad_count = aug_labels.count("BAD")
        good_count = aug_labels.count("GOOD")

        if bad_count < good_count * 0.5:
            n_extra_bad = min(good_count - bad_count, 30)
            if mod_name in ("module4", "policy_info"):
                extra_bad = template_gen.generate_accounting_boilerplate(n_extra_bad)
            else:
                extra_bad = template_gen.generate_negative_samples(n_extra_bad // 2)
                extra_bad += template_gen.generate_accounting_boilerplate(n_extra_bad // 2)

            aug_texts.extend(extra_bad[:n_extra_bad])
            aug_labels.extend(["BAD"] * len(extra_bad[:n_extra_bad]))

        augmented[mod_name] = {
            "texts": aug_texts,
            "labels": aug_labels,
        }

    return augmented


def cross_validate(
    embedder, train_data: Dict, n_folds: int = 3
) -> Dict:
    """
    简单交叉验证（留出法）

    对于每个模块，随机分出20%作为验证集
    """
    import random
    random.seed(42)

    results = {}

    for mod_name, data in train_data.items():
        texts = data["texts"]
        labels = data["labels"]

        if len(texts) < 5:
            results[mod_name] = {
                "accuracy": None,
                "n_samples": len(texts),
                "error": "样本太少，无法评估",
            }
            continue

        # 分层采样
        good_indices = [i for i, l in enumerate(labels) if l == "GOOD"]
        bad_indices = [i for i, l in enumerate(labels) if l == "BAD"]

        if not good_indices or not bad_indices:
            results[mod_name] = {
                "accuracy": None,
                "n_samples": len(texts),
                "error": "缺少正类或负类样本",
            }
            continue

        # 留出20%
        n_good_val = max(1, int(len(good_indices) * 0.2))
        n_bad_val = max(1, int(len(bad_indices) * 0.2))

        random.shuffle(good_indices)
        random.shuffle(bad_indices)

        val_good_idx = set(good_indices[:n_good_val])
        val_bad_idx = set(bad_indices[:n_bad_val])
        val_idx = val_good_idx | val_bad_idx
        train_idx = [i for i in range(len(texts)) if i not in val_idx]

        # 训练
        clf = PrototypeClassifier(embedder, class_names=["GOOD", "BAD"])
        clf.add_examples(
            [texts[i] for i in train_idx],
            [labels[i] for i in train_idx],
        )
        clf.fit()

        # 验证
        correct = 0
        total = 0
        predictions = []

        for i in val_idx:
            result = clf.predict(texts[i])
            if result.predicted_class == labels[i]:
                correct += 1
            total += 1
            predictions.append(result.predicted_class)

        accuracy = correct / max(1, total)

        results[mod_name] = {
            "accuracy": round(accuracy, 3),
            "n_train": len(train_idx),
            "n_val": total,
            "good_train": sum(1 for i in train_idx if labels[i] == "GOOD"),
            "bad_train": sum(1 for i in train_idx if labels[i] == "BAD"),
        }

    return results


def _build_corpus_from_cache(max_files: int = 20) -> list:
    """从缓存年报文本构建训练语料"""
    corpus = []
    cache_dir = os.path.join(os.path.dirname(OUTPUT_DIR), "cache")
    if os.path.exists(cache_dir):
        for f in sorted(os.listdir(cache_dir))[:max_files]:
            if f.endswith('.txt'):
                try:
                    with open(os.path.join(cache_dir, f), 'r', encoding='utf-8') as fp:
                        raw = fp.read()
                    lines = raw.split('\n')
                    chunk = []
                    chunk_len = 0
                    for line in lines:
                        s = line.strip()
                        if not s:
                            continue
                        chunk.append(s)
                        chunk_len += len(s)
                        if chunk_len >= 300:
                            corpus.append(''.join(chunk))
                            chunk = []
                            chunk_len = 0
                except Exception:
                    pass
    return corpus


def _load_embedder(args):
    """加载嵌入模型 — 优先轻量级方案（零网络依赖）"""
    # 方案1：轻量级TF-IDF+SVD（纯本地，无需GPU/网络）
    try:
        from lightweight_embedder import LightweightChineseEmbedder
        print("  [ML] 使用轻量级TF-IDF+SVD嵌入器")
        embedder = LightweightChineseEmbedder(
            svd_components=256,
            max_features=8000,
        )
        corpus = _build_corpus_from_cache()
        if corpus:
            embedder.fit_on_corpus(corpus)
            print(f"  [ML] 在 {len(corpus)} 个文本块上训练完成")
        return embedder
    except Exception as e:
        print(f"  [INFO] 轻量级嵌入器不可用: {e}")

    # 方案2：ChineseTextEmbedder（需要sentence-transformers）
    try:
        embedder = ChineseTextEmbedder(
            model_name=args.model_name,
            cache_embeddings=True,
        )
        if embedder.is_ready():
            return embedder
    except Exception:
        pass

    return None


def train(args):
    """主训练流程"""
    print("=" * 60)
    print("ML模型训练: 小样本语义质量分类器")
    print("=" * 60)

    # 1. 加载嵌入模型
    print("\n[1/5] 加载中文嵌入模型...")
    embedder = _load_embedder(args)
    if embedder is None or not embedder.is_ready():
        print("  [ERROR] 所有嵌入模型方案均加载失败")
        print("  提示: 轻量级方案需要cache目录下有年报txt文件")
        return False

    # 2. 构建训练数据
    print("\n[2/5] 构建训练数据...")

    if args.from_feedback:
        print("  来源: 反馈存储")
        store = FeedbackStore(FEEDBACK_STORE_PATH)
        train_data = build_training_data_from_feedback(store)
    else:
        print("  来源: 预定义参考文本")
        train_data = build_training_data_from_references()

    total_samples = sum(len(d["texts"]) for d in train_data.values())
    print(f"  初始训练样本: {total_samples} 条")

    for mod_name, data in train_data.items():
        label_counts = Counter(data["labels"])
        print(f"    {mod_name}: {label_counts}")

    # 3. 数据增强
    if args.with_augment:
        print(f"\n[3/5] 数据增强 (factor={args.augment_factor})...")
        train_data = augment_training_data(train_data, args.augment_factor)
        total_samples = sum(len(d["texts"]) for d in train_data.values())
        print(f"  增强后训练样本: {total_samples} 条")

        for mod_name, data in train_data.items():
            label_counts = Counter(data["labels"])
            print(f"    {mod_name}: {label_counts}")
    else:
        print("\n[3/5] 跳过数据增强 (使用 --with-augment 启用)")

    # 4. 训练分类器
    print("\n[4/5] 训练少样本原型分类器...")
    mclf = MultiModuleClassifier(embedder, confidence_threshold=0.3)

    for mod_name in mclf.module_names:
        data = train_data.get(mod_name, {"texts": [], "labels": []})
        if data["texts"]:
            clf = mclf.classifiers[mod_name]
            clf.add_examples(data["texts"], data["labels"])
            clf.fit()
            print(f"  {mod_name}: {clf.total_examples} 个样本, "
                  f"已训练={clf.is_trained}")
        else:
            print(f"  {mod_name}: 无训练数据")

    # 5. 交叉验证评估
    print("\n[5/5] 交叉验证评估...")
    eval_results = cross_validate(embedder, train_data)

    print("\n评估结果:")
    print(f"{'模块':<15} {'准确率':<10} {'训练样本':<10} {'验证样本':<10} {'状态'}")
    print("-" * 55)

    overall_accuracies = []
    for mod_name in mclf.module_names:
        r = eval_results.get(mod_name, {})
        acc = r.get("accuracy")
        n_train = r.get("n_train", 0)
        n_val = r.get("n_val", 0)
        error = r.get("error", "")

        if acc is not None:
            print(f"{mod_name:<15} {acc:.1%}       {n_train:<10} {n_val:<10} ✓")
            overall_accuracies.append(acc)
        else:
            print(f"{mod_name:<15} {'N/A':<10} {n_train:<10} {n_val:<10} {error}")

    if overall_accuracies:
        avg_acc = sum(overall_accuracies) / len(overall_accuracies)
        print(f"\n{'平均准确率':<15} {avg_acc:.1%}")

    # 6. 保存模型
    model_dir = os.path.join(OUTPUT_DIR, "ml_models")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(model_dir, f"model_{timestamp}")

    mclf.save(save_path)

    # 保存训练元信息
    meta = {
        "timestamp": timestamp,
        "model_name": args.model_name,
        "training_samples": total_samples,
        "augmented": args.with_augment,
        "augment_factor": args.augment_factor,
        "evaluation": eval_results,
        "module_stats": mclf.get_training_stats(),
    }
    with open(os.path.join(save_path, "training_meta.json"), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 保存最新的模型软链接路径
    latest_link = os.path.join(model_dir, "latest_model_path.txt")
    with open(latest_link, 'w', encoding='utf-8') as f:
        f.write(save_path)

    print(f"\n模型已保存到: {save_path}")
    print(f"训练完成!")

    return True


def evaluate(args):
    """评估已有模型"""
    print("=" * 60)
    print("模型评估")
    print("=" * 60)

    model_dir = os.path.join(OUTPUT_DIR, "ml_models")
    latest_link = os.path.join(model_dir, "latest_model_path.txt")

    if os.path.exists(latest_link):
        with open(latest_link, 'r', encoding='utf-8') as f:
            model_path = f.read().strip()
    else:
        # 找最新的模型目录
        subdirs = sorted([
            d for d in os.listdir(model_dir)
            if os.path.isdir(os.path.join(model_dir, d)) and d.startswith("model_")
        ])
        if not subdirs:
            print("未找到已训练的模型")
            return
        model_path = os.path.join(model_dir, subdirs[-1])

    print(f"加载模型: {model_path}")

    # 加载元信息
    meta_path = os.path.join(model_path, "training_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        print(f"训练时间: {meta.get('timestamp')}")
        print(f"训练样本: {meta.get('training_samples')}")
        print(f"数据增强: {meta.get('augmented')}")
        print(f"\n训练时评估结果:")
        for mod_name, r in meta.get("evaluation", {}).items():
            acc = r.get("accuracy")
            if acc:
                print(f"  {mod_name}: {acc:.1%}")

    # 加载并测试模型
    try:
        # 使用与训练相同的嵌入器加载逻辑
        import argparse
        eval_args = argparse.Namespace(model_name="shibing624/text2vec-base-chinese")
        embedder = _load_embedder(eval_args)
        mclf = MultiModuleClassifier.load(model_path, embedder)
        print(f"\n当前模型状态:")
        stats = mclf.get_training_stats()
        for mod_name, s in stats.items():
            print(f"  {mod_name}: 总样本={s['total_examples']}, "
                  f"GOOD={s['good_count']}, BAD={s['bad_count']}, "
                  f"已训练={s['is_trained']}")
    except Exception as e:
        print(f"加载失败: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="训练小样本语义质量分类器"
    )
    parser.add_argument(
        "--model-name", default="shibing624/text2vec-base-chinese",
        help="Sentence-Transformer模型名称",
    )
    parser.add_argument(
        "--with-augment", action="store_true",
        help="启用数据增强",
    )
    parser.add_argument(
        "--augment-factor", type=int, default=3,
        help="数据增强倍数（每条文本生成的变体数）",
    )
    parser.add_argument(
        "--from-feedback", action="store_true",
        help="从反馈存储中构建训练数据",
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="评估已有模型（不重新训练）",
    )

    args = parser.parse_args()

    if args.evaluate:
        evaluate(args)
    else:
        success = train(args)
        if not success:
            print("\n训练失败。可能的解决方案:")
            print("  1. 检查网络连接（需要下载模型）")
            print("  2. 尝试不同的模型: --model-name paraphrase-multilingual-MiniLM-L12-v2")
            print("  3. 仅使用TF-IDF回退方案（无需训练）")


if __name__ == "__main__":
    main()
