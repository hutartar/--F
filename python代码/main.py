"""
主流程入口（v5-ML）：批量处理50家企业年报，深度学习增强版
新增（v5）：
- 语义嵌入质量评分（Chinese Sentence-BERT替代TF-IDF）
- 少样本原型分类器（每类5-20个样本即可训练）
- 主动学习循环（持续从反馈中改进）
- 数据增强训练（解决小样本欠拟合）
- 保留原有TF-IDF回退方案

原有功能：
- 质量评分（多维度：完整性/相关性/清洁度/锅炉板/信息量）
- 跨模块一致性验证
- LLM文本精炼（LOW/MEDIUM置信度结果）
- 自学习反馈存储
- 自适应阈值调整
"""
import os
import sys
from datetime import datetime

from config import (
    OUTPUT_DIR, REPORT_YEAR, OUTPUT_COLUMNS, RESULT_TO_COLUMN, TARGET_EXCEL,
    FEEDBACK_STORE_PATH,
    ML_USE_EMBEDDINGS, ML_ACTIVE_LEARNING, ML_MODEL_DIR,
    ML_DATA_AUGMENTATION, ML_AUGMENT_FACTOR,
)
from mapper import extract_company_list, build_pdf_map
from pdf_reader import batch_extract

# 可选：巨潮资讯网下载器
try:
    from cninfo_downloader import CninfoDownloader
    HAS_CNINFO = True
except ImportError:
    HAS_CNINFO = False
from text_cleaner import clean_full_text
from section_locator import locate_target_subsections
from module_extractors import extract_all_modules
from ml_scorer import score_all_modules, load_ml_models, save_ml_models
from cross_validator import validate_all
from excel_writer import create_output_excel, fill_target_excel

# 可选ML组件
try:
    from llm_refiner import LLMRefiner
    HAS_LLM = True
except ImportError:
    HAS_LLM = False

try:
    from feedback_store import FeedbackStore
    from adaptive_config import AdaptiveThresholdManager
    HAS_FEEDBACK = True
except ImportError:
    HAS_FEEDBACK = False

# 新ML组件（可选）
try:
    from ml_embeddings import ChineseTextEmbedder
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False

try:
    from few_shot_classifier import MultiModuleClassifier
    HAS_FEWSHOT = True
except ImportError:
    HAS_FEWSHOT = False

try:
    from active_learner import ActiveLearner
    HAS_ACTIVE_LEARNER = True
except ImportError:
    HAS_ACTIVE_LEARNER = False

try:
    from data_augmentation import ChineseFinancialTextAugmenter
    HAS_AUGMENTER = True
except ImportError:
    HAS_AUGMENTER = False


def process_all(use_llm=True, refine_all=False, review_suggestions=False,
                train_ml=False, active_learning=False, download=False,
                report_year=None, output_dir=None):
    # 支持动态切换年份和输出目录
    year = report_year or REPORT_YEAR
    out_dir = output_dir or OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    # 动态更新配置中的路径
    import config
    config.OUTPUT_DIR = out_dir
    config.REPORT_YEAR = year
    config.FEEDBACK_STORE_PATH = os.path.join(out_dir, "feedback.jsonl")
    config.ML_MODEL_DIR = os.path.join(out_dir, "ml_models")

    print("=" * 60)
    print(f"年报外汇风险管理文本数据自动化提取系统 v5-ML ({year}年)")
    print("=" * 60)
    print(f"  输出目录: {out_dir}")

    # ── 步骤0：ML模型准备 ──
    if ML_USE_EMBEDDINGS or train_ml:
        print("\n[0/9] 初始化ML语义模型...")
        if HAS_EMBEDDINGS:
            try:
                # 尝试加载已有模型
                model_loaded = False
                if os.path.exists(ML_MODEL_DIR):
                    model_loaded = load_ml_models(ML_MODEL_DIR)
                    if model_loaded:
                        print("  已加载预训练ML模型")
            except Exception:
                model_loaded = False

            if train_ml and not model_loaded and HAS_FEWSHOT:
                print("  开始ML模型训练...")
                try:
                    from train_ml_model import train as train_ml_models
                    import argparse
                    train_args = argparse.Namespace(
                        model_name="shibing624/text2vec-base-chinese",
                        with_augment=ML_DATA_AUGMENTATION,
                        augment_factor=ML_AUGMENT_FACTOR,
                        from_feedback=os.path.exists(FEEDBACK_STORE_PATH),
                        evaluate=False,
                    )
                    train_ml_models(train_args)
                except Exception as e:
                    print(f"  [WARN] ML训练失败: {e}，将使用启发式评分")
        else:
            print("  [INFO] ML嵌入模型未安装，使用TF-IDF回退方案")
            print("  安装: pip install sentence-transformers")

    # ── 步骤1：企业清单 + PDF映射 ──
    print("\n[1/9] 提取企业清单并映射PDF...")
    companies = extract_company_list()
    mapping = build_pdf_map(companies)
    print(f"  企业总数: {len(companies)}, PDF映射成功: {len(mapping)}")

    # ── 步骤1.5：（可选）从巨潮资讯网下载缺失的年报 ──
    if download and HAS_CNINFO:
        print("\n[1.5/9] 检查并下载缺失的年报PDF...")
        missing = [m for m in mapping if not os.path.exists(m["pdf_path"])]
        if missing:
            print(f"  缺失 {len(missing)} 份年报，从巨潮资讯网下载...")
            downloader = CninfoDownloader()
            dl_results = downloader.batch_download(missing, year)
            # 更新mapping中的PDF路径
            for dl in dl_results:
                if dl.pdf_path:
                    for m in mapping:
                        if m["code"] == dl.company_code:
                            m["pdf_path"] = dl.pdf_path
                            break
            # 重新统计
            still_missing = sum(1 for m in mapping if not os.path.exists(m["pdf_path"]))
            print(f"  下载后仍缺失: {still_missing}/{len(missing)}")
        else:
            print(f"  所有年报PDF已就绪")

    # ── 步骤2：批量提取PDF文本 ──
    print("\n[2/9] 批量提取PDF文本...")
    pdf_paths = [m["pdf_path"] for m in mapping]
    pdf_texts = batch_extract(pdf_paths, use_cache=True)
    success_count = sum(1 for v in pdf_texts.values() if v)
    print(f"  PDF文本提取成功: {success_count}/{len(pdf_texts)}")

    # ── 步骤3：清洗 + 章节定位 + 5模块提取 ──
    print("\n[3/9] 清洗文本、定位章节并提取5模块信息...")
    results = []
    empty_stats = {
        "module1": 0, "module2": 0, "module3": 0,
        "module4": 0, "module5": 0, "policy_info": 0,
    }

    for i, comp in enumerate(mapping):
        code = comp["code"]
        name = comp["name"]
        pdf_path = comp["pdf_path"]
        raw_text = pdf_texts.get(pdf_path, "")

        if not raw_text:
            results.append(_empty_record(code, name, "PDF文本为空", year))
            for k in empty_stats:
                empty_stats[k] += 1
            continue

        try:
            text = clean_full_text(raw_text)
            targets = locate_target_subsections(text)
            extracted = extract_all_modules(text, targets)

            record = {
                "code": code,
                "name": name,
                "year": year,
                "module1": extracted.get("module1", ""),
                "module2": extracted.get("module2", ""),
                "module3": extracted.get("module3", ""),
                "module4": extracted.get("module4", ""),
                "module5": extracted.get("module5", ""),
                "policy_info": extracted.get("policy_info", ""),
            }
            results.append(record)

            for mod_key in empty_stats:
                if not extracted.get(mod_key, ""):
                    empty_stats[mod_key] += 1

        except Exception as e:
            print(f"  [ERROR] {code} {name}: {e}")
            results.append(_empty_record(code, name, f"处理异常: {e}", year))
            for k in empty_stats:
                empty_stats[k] += 1

        if (i + 1) % 10 == 0:
            print(f"  已处理 {i + 1}/{len(mapping)} ...")

    print(f"  全部 {len(mapping)} 家企业处理完成")

    # ── 步骤4：ML质量评分（语义嵌入 + 少样本分类 + TF-IDF回退）──
    print("\n[4/9] ML质量评分（语义嵌入 + 少样本分类 + 多维度评分）...")
    results = score_all_modules(results)

    total = len(mapping)
    quality_stats = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for record in results:
        for mod in ["module1", "module2", "module3", "module4", "module5", "policy_info"]:
            q = record.get("_quality", {}).get(mod, {})
            conf = q.get("confidence", "MEDIUM")
            if conf in quality_stats:
                quality_stats[conf] += 1

    print(f"  质量分布: HIGH={quality_stats['HIGH']}, MEDIUM={quality_stats['MEDIUM']}, "
          f"LOW={quality_stats['LOW']}")

    # ── 步骤5：跨模块一致性验证 ──
    print("\n[5/9] 跨模块一致性验证...")
    results = validate_all(results)

    # ── 步骤5.5：主动学习（可选）──
    if active_learning and HAS_ACTIVE_LEARNER and HAS_FEWSHOT:
        print("\n[5.5/9] 主动学习循环...")
        try:
            al_store = FeedbackStore(FEEDBACK_STORE_PATH)
            embedder = ChineseTextEmbedder(cache_embeddings=True) if HAS_EMBEDDINGS else None
            from few_shot_classifier import MultiModuleClassifier
            mclf = MultiModuleClassifier(embedder) if embedder else None

            if mclf and embedder and embedder.is_ready():
                learner = ActiveLearner(mclf, embedder, al_store)
                learner.run_iteration(results, auto_apply=True)

                # 显示审核队列
                queue = learner.get_review_queue()
                if queue:
                    print(f"  发现 {len(queue)} 个不确定样本需要人工审核")
                    print(f"  前5个:")
                    for i, sample in enumerate(queue[:5]):
                        print(f"    [{i+1}] {sample.module_name} "
                              f"({sample.company_name}): "
                              f"置信度={sample.confidence:.2f}")

                learner.save_state(os.path.join(ML_MODEL_DIR, "active_learner_state.json"))
            else:
                print("  [SKIP] ML模型不可用，跳过主动学习")
        except Exception as e:
            print(f"  [WARN] 主动学习出错: {e}")
    else:
        print("\n[5.5/9] 主动学习跳过 (使用 --active-learning 启用)")

    # ── 步骤6：LLM文本精炼 ──
    print("\n[6/9] LLM文本精炼...")
    if use_llm and HAS_LLM:
        try:
            refiner = LLMRefiner()
            results = refiner.batch_refine(results, refine_all=refine_all)
        except Exception as e:
            print(f"  [WARN] LLM精炼失败: {e}")
            print(f"  继续使用未精炼的结果")
    elif use_llm and not HAS_LLM:
        print("  [SKIP] anthropic 未安装，跳过LLM精炼。安装: pip install anthropic")
    else:
        print("  [SKIP] 跳过LLM精炼（使用 --no-llm 标志）")

    # ── 步骤7：自学习反馈 ──
    print("\n[7/9] 自学习反馈记录...")
    if HAS_FEEDBACK:
        store = FeedbackStore(FEEDBACK_STORE_PATH)
        store.record_batch(results)

        adaptive = AdaptiveThresholdManager(feedback_store=store)
        if review_suggestions:
            print(adaptive.get_report())
        else:
            adaptive.apply_adjustments()

        stats = store.get_stats_summary()
        print(f"  {stats.split(chr(10))[0]}")  # 第一行摘要
    else:
        print("  [SKIP] 反馈存储模块未加载")

    # ── 步骤8：统计 + Excel输出 ──
    print("\n[8/9] 统计与输出...")

    # 覆盖率统计
    print("  字段覆盖率:")
    for mod_key in ["module1", "module2", "module3", "module4", "module5", "policy_info"]:
        filled = total - empty_stats.get(mod_key, 0)
        rate = filled * 100 / total if total > 0 else 0
        print(f"    {mod_key}: {filled}/{total} ({rate:.0f}%)")

    # 质量统计
    record_qualities = []
    for record in results:
        overall = record.get("_overall_quality", 0)
        record_qualities.append(overall)
    if record_qualities:
        avg_quality = sum(record_qualities) / len(record_qualities)
        print(f"  平均质量分: {avg_quality:.1f}/100")

    # 生成Excel
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_path = os.path.join(out_dir, f"外汇风险管理提取结果_{timestamp}.xlsx")
    create_output_excel(results, output_path)

    if os.path.exists(TARGET_EXCEL):
        target_path = os.path.join(out_dir, f"目标模板填充结果_{timestamp}.xlsx")
        fill_target_excel(results, TARGET_EXCEL, target_path)
        print(f"  目标模板填充: {target_path}")
    else:
        print(f"  [WARN] 目标模板不存在: {TARGET_EXCEL}")

    print("\n" + "=" * 60)
    print(f"完成! 结果文件: {output_path}")
    print("=" * 60)

    # ── 步骤9：保存ML模型 ──
    print("\n[9/9] 保存ML模型状态...")
    if ML_USE_EMBEDDINGS:
        try:
            save_ml_models(ML_MODEL_DIR)
            print(f"  ML模型已保存到: {ML_MODEL_DIR}")
        except Exception as e:
            print(f"  [WARN] 保存ML模型失败: {e}")
    else:
        print("  [SKIP] 未使用ML嵌入模型")


def _empty_record(code, name, reason="", year=""):
    return {
        "code": code,
        "name": name,
        "year": year or REPORT_YEAR,
        "module1": reason if reason else "",
        "module2": "", "module3": "", "module4": "", "module5": "",
        "policy_info": "",
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="年报外汇风险管理文本数据自动化提取系统 v5-ML")
    parser.add_argument("--no-llm", action="store_true",
                       help="跳过LLM文本精炼步骤")
    parser.add_argument("--refine-all", action="store_true",
                       help="精炼所有结果（包括HIGH置信度）")
    parser.add_argument("--review-suggestions", action="store_true",
                       help="查看自适应阈值建议而不自动应用")
    parser.add_argument("--train-ml", action="store_true",
                       help="在运行前训练/重新训练ML模型")
    parser.add_argument("--active-learning", action="store_true",
                       help="启用主动学习循环")
    parser.add_argument("--no-ml", action="store_true",
                       help="禁用ML语义模型，仅使用TF-IDF回退方案")
    parser.add_argument("--download", action="store_true",
                       help="从巨潮资讯网自动下载缺失的年报PDF")
    parser.add_argument("--year", type=str, default=None,
                       help="年报年份（默认2023）")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="输出目录（默认output/）")

    args = parser.parse_args()

    # 如果指定--no-ml，禁用嵌入模型
    if args.no_ml:
        from config import ML_USE_EMBEDDINGS

    process_all(
        use_llm=not args.no_llm,
        refine_all=args.refine_all,
        review_suggestions=args.review_suggestions,
        train_ml=args.train_ml,
        active_learning=args.active_learning,
        download=args.download,
        report_year=args.year,
        output_dir=args.output_dir,
    )
