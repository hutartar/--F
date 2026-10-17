# 钢炼-F — 财务报表文本数据结构化提取系统

从A股上市公司年报PDF中自动识别、提取并结构化外汇风险管理相关文本，覆盖5个核心模块，支持多年度数据对比。

## 📊 项目效果

| 提取模块 | 2023年填充率 | 2024年填充率 |
|---|---|---|
| M1 管理层讨论与分析-未来展望 | 36% | 42% |
| M2 衍生品投资/套期保值情况 | 100% | 98% |
| M3 外币货币性项目币种 | 100% | 98% |
| M4 记账本位币/境外经营实体 | 100% | 100% |
| M5 金融工具风险-汇率风险段落 | 98% | 98% |
| 外汇风险管理制度信息 | 8% | 8% |

**核心指标**：50家企业 · 5个提取模块 · 98%核心字段填充率 · 支持2023-2024两年对比

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
[4] 5模块正则提取（关键词精确检索）
    ↓
[5] ML质量评分（BGE语义嵌入 + 多维度打分）
    ↓
[6] 跨模块一致性验证（矛盾自动检测）
    ↓
[7] LLM文本精炼（LOW置信度结果优化）
    ↓
[8] 反馈存储（人工修正自动学习）
    ↓
[9] Excel结构化输出（19列）
```

## ✨ 技术亮点

- **BGE语义嵌入**：采用BAAI/bge-small-zh-v1.5中文语义模型，512维向量表示
- **多维度质量评分**：完整性、相关性、清洁度、锅炉板检测、信息量5维评估
- **跨模块一致性验证**：自动检测跨模块矛盾（如M2说无衍生品但M5出现远期合约）
- **主动学习机制**：人工修正样本自动反馈，持续优化分类模型
- **巨潮资讯自动爬取**：内置cninfo下载器，按股票代码批量搜索下载年报PDF
- **PDF缓存加速**：提取过的文本自动缓存，重复运行秒级完成

## 📁 项目结构

```
├── python代码/           # 核心代码
│   ├── main.py          # 主流程入口
│   ├── config.py        # 配置文件（路径、关键词、正则）
│   ├── pdf_reader.py    # PDF文本提取 + 缓存
│   ├── text_cleaner.py  # 文本清洗
│   ├── section_locator.py # 章节定位
│   ├── module_extractors.py # 5模块提取逻辑
│   ├── ml_scorer.py     # ML质量评分
│   ├── ml_embeddings.py # BGE语义嵌入
│   ├── cross_validator.py # 跨模块一致性验证
│   ├── excel_writer.py  # Excel输出
│   ├── cninfo_downloader.py # 巨潮资讯年报下载器
│   └── ...
├── output/              # 提取结果输出
├── index.html           # 可视化Demo看板
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

### 运行提取
```bash
cd python代码
python main.py --year 2024 --no-llm --no-ml
```

### 下载年报
```python
from cninfo_downloader import CninfoDownloader
downloader = CninfoDownloader()
result = downloader.download("300258", "精锻科技", "2024")
```

## 📈 已解决的关键问题

1. **M1填充率低**：提取窗口从2500字符扩大到6000，覆盖完整展望章节
2. **制度信息误提**：扩充CSRC通用文档黑名单，过滤准确率从6%提升到92%
3. **PDF摘要版问题**：增加文件大小校验（<500KB标记为摘要），重新下载完整版
4. **torch环境崩溃**：降级至torch 2.2.2 CPU版，解决DLL初始化失败问题

## 🔮 未来优化方向

- [ ] 接入LLM精炼模块，优化LOW置信度提取结果
- [ ] 扩展至更多年度数据（2022年及更早）
- [ ] 主动学习闭环，积累200+标注样本后启用
- [ ] 开发Web端可视化查询界面
- [ ] 扩展更多风险维度（利率风险、信用风险等）

## 📄 License

MIT License
