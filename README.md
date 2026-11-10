# 钢炼-F — 财务报表文本数据结构化提取系统

从A股上市公司年报PDF中自动识别、提取并结构化外汇风险管理相关文本，覆盖5个核心模块，支持多年度数据对比。

> 说明：本系统的字段抽取由**规则/正则**完成，语义嵌入模型仅用于**质量评分**，LLM 仅用于**文本清理**，三者分工明确，详见下文。

## 📊 项目效果

下表为各模块的**字段填充率（非空率）**，衡量工程覆盖情况：

| 提取模块 | 2023年填充率 | 2024年填充率 |
|---|---|---|
| M1 管理层讨论与分析-未来展望 | 36% | 42% |
| M2 衍生品投资/套期保值情况 | 100% | 98% |
| M3 外币货币性项目币种 | 100% | 98% |
| M4 记账本位币/境外经营实体 | 100% | 100% |
| M5 金融工具风险-汇率风险段落 | 98% | 98% |
| 外汇风险管理制度信息 | 8% | 8% |

**核心指标**：50家企业 · 5个提取模块 · 98%核心字段填充率 · 支持2023-2024两年对比

> ⚠️ **填充率 ≠ 准确率**：填充率只表示该字段非空，不代表抽取内容正确。基于人工金标准的精确率/召回率/F1 评估正在建设中，见“已知局限”。

## 🏗️ 系统架构

```
PDF年报文件
    ↓
[1] PDF文本提取（pdfplumber + 本地缓存）
    ↓
[2] 文本清洗（去页眉页脚 + 格式规范化）
    ↓
[3] 章节定位（十大节结构自动识别）
    ↓
[4] 5模块规则提取（关键词/正则/字符窗口，不依赖模型）
    ↓
[5] 质量评分（语义嵌入 + 5维启发式打分；可用 --no-ml 关闭）
    ↓
[6] 跨模块一致性验证（缺失/详略/制度类规则检测）
    ↓
[7] LLM文本清理（仅清理低置信度文本，不做事实判断）
    ↓
[8] 反馈存储（人工修正记录，供后续学习）
    ↓
[9] Excel结构化输出
```

## ✨ 技术亮点

- **规则驱动的可审计抽取**：字段抽取全部由关键词、正则与段落边界规则完成，结果可追溯、可复现，不依赖模型黑盒
- **语义嵌入质量评分**：评分路径实际加载 BAAI/bge-small-zh-v1.5（512维），结合完整性、相关性、清洁度、锅炉板、信息量5维启发式打分
- **跨模块一致性验证**：基于5条确定性规则检测跨模块的缺失、详略与制度问题（规则清单见下表）
- **可关闭的语义路径**：`--no-ml` 真正生效，关闭神经网络后回退到 TF-IDF/启发式，便于离线与对照
- **巨潮资讯自动爬取**：内置cninfo下载器，按股票代码批量搜索下载年报PDF
- **PDF缓存加速**：提取过的文本自动缓存，重复运行秒级完成

### 跨模块验证规则（cross_validator.py）

| 规则 | 检测内容 | 级别 | 类型 |
|---|---|---|---|
| CURRENCIES_WITHOUT_RISK | M3 有币种但 M5 无实质汇率风险披露 | WARNING/INFO | 缺失型 |
| DERIVATIVES_WITHOUT_POLICY | M2 有衍生品但未找到对应风险管理制度 | WARNING | 缺失型 |
| OUTLOOK_FX_WITHOUT_RISK | M1 展望有汇率讨论但 M5 无汇率风险 | INFO | 缺失型 |
| CURRENCY_MISMATCH | M3 与 M5 币种集合不一致 | INFO | 详略型 |
| POLICY_CSRC_ONLY | 制度信息仅含 CSRC 通用文档、无公司特定制度 | INFO | 制度有效性 |

> 注：当前规则检测的是“**此处有、彼处无**”的缺失与详略差异，**尚不能自动判定两处陈述直接冲突（如一处明确“无衍生品”、另一处描述具体工具）**。冲突型检测是后续重点工作。

## 📁 项目结构

```
├── python代码/
│   ├── main.py                # 主流程入口
│   ├── config.py              # 配置（路径、关键词、正则、阈值）
│   ├── pdf_reader.py          # PDF文本提取 + 缓存
│   ├── text_cleaner.py        # 文本清洗
│   ├── section_locator.py     # 章节定位
│   ├── module_extractors.py   # 5模块规则提取
│   ├── ml_scorer.py           # 质量评分（语义+启发式）
│   ├── ml_embeddings.py       # 语义嵌入与参考文本
│   ├── few_shot_classifier.py # 少样本原型分类器
│   ├── cross_validator.py     # 跨模块一致性验证
│   ├── active_learner.py      # 主动学习
│   ├── feedback_store.py      # 反馈存储
│   ├── adaptive_config.py     # 自适应阈值
│   ├── data_augmentation.py   # 文本数据增强
│   ├── lightweight_embedder.py# TF-IDF+SVD 回退嵌入
│   ├── train_ml_model.py      # 模型训练脚本
│   ├── llm_refiner.py         # LLM 文本清理
│   ├── mapper.py              # 企业清单/PDF映射
│   ├── excel_writer.py        # Excel输出
│   └── cninfo_downloader.py   # 巨潮资讯下载器
├── index.html                 # 可视化看板
└── README.md
```

## 🚀 快速开始

### 环境要求
- Python 3.12+
- 依赖：pdfplumber, openpyxl, jieba, torch, transformers, sentence-transformers, scikit-learn

### 安装依赖
```bash
pip install pdfplumber openpyxl jieba scikit-learn
pip install torch transformers sentence-transformers
```

### 路径配置
默认基于 `config.py` 所在位置自动推导路径；也可通过环境变量覆盖：

| 环境变量 | 作用 |
|---|---|
| `FX_BASE_DIR` | 工作根目录（含企业清单、cache、output） |
| `FX_PDF_DIR` | 年报 PDF 所在目录 |
| `FX_VIS_DIR` | 可视化目录（output 位于其下） |
| `FX_TARGET_EXCEL` | 目标模板 Excel 路径 |

### 运行提取
```bash
cd python代码
python main.py --year 2024                 # 完整流程（语义评分+LLM清理）
python main.py --year 2024 --no-llm        # 不调用 LLM
python main.py --year 2024 --no-ml         # 关闭语义模型，仅 TF-IDF/启发式
```

### 下载年报
```python
from cninfo_downloader import CninfoDownloader
downloader = CninfoDownloader()
result = downloader.download("300258", "精锻科技", "2024")
```

## 🔧 已处理的工程问题

1. **路径硬编码**：改为基于文件位置自动推导，并支持环境变量覆盖，换机器可直接运行
2. **`--no-ml` 失效**：参数现在真正贯穿流程，关闭语义模型后回退 TF-IDF/启发式
3. **M3 空值规范**：未检索到内容时留空（不再误填“无”）；原文“不适用”忠实返回；二者均正确识别为有效披露
4. **M1 覆盖范围**：提取窗口扩大以覆盖完整展望章节
5. **制度信息过滤**：收紧关键词并使用 CSRC 通用文档黑名单
6. **PDF 摘要版问题**：增加文件大小校验，过小的摘要版重新下载完整版

## 🔎 已知局限与后续工作

- [ ] 建立人工金标准，报告各模块精确率/召回率/F1 与标注者一致性（Cohen's κ）
- [ ] 增加**冲突型**跨模块检测（事实槽位 + 语义 NLI + 带证据约束的 LLM 裁决）
- [ ] 主动学习闭环：当前反馈/主动学习框架已就绪，待积累足够标注后真正闭环
- [ ] 扩展至更早年度（2022 年及以前），扩大样本规模
- [ ] 扩展更多风险维度（利率风险、信用风险等）

## 📄 License

MIT License
