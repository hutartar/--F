"""
关键词、正则模式、输出字段定义
"""

import os

# 路径配置
BASE_DIR = r"C:\Users\MECHREV\Desktop\文本数据提取方法\文本\文本数据提取"
PDF_DIR = r"C:\Users\MECHREV\Desktop\文本数据提取方法\文本\测试集\annual_reports"
VIS_DIR = r"C:\Users\MECHREV\Desktop\文本数据提取方法\文本\可视化"
OUTPUT_DIR = os.path.join(VIS_DIR, "output")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
COMPANY_LIST_DOCX = os.path.join(BASE_DIR, "负责的50家企业.docx")
TARGET_EXCEL = r"C:\Users\MECHREV\Desktop\文本数据提取方法\负责的企业名单（50家）上市公司年报外汇风险管理数据查找、搜集、整理、输入 - 正式信息输入文本（劲仔食品-中颖电子）.xlsx"

REPORT_YEAR = "2023"

# 输出列名（匹配范例格式）
OUTPUT_COLUMNS = [
    "代码",
    "上市公司简称",
    "",
    '谷任检索年报文字信息-"衍生品投资情况"或"套期"',
    '谷任检索年报文字信息-"管理层讨论与分析"中的"公司未来发展的展望"',
    '谷任检索年报文字信息-"外币货币性项目"',
    '谷任检索年报文字"境外经营实体"',
    '谷任检索年报文字信息-"市场风险"或者"与金融工具相关的风险"',
    "谷任调整版本",
    "制度信息",
    "风险管理模式",
    "E1", "E2", "E3",
    "F1", "F2", "F3",
    "G1", "G2",
]

# 从extract_all_modules结果到输出列的映射
RESULT_TO_COLUMN = {
    "module2": 4,     # 衍生品投资情况
    "module1": 5,     # 管理层讨论与分析-未来展望
    "module3": 6,     # 外币货币性项目
    "module4": 7,     # 境外经营实体
    "module5": 8,     # 市场风险
    "policy_info": 10,  # 制度信息
}

# 模块配置（精简后更精准的关键词）
MODULES = {
    "module1": {
        "name": "管理层讨论与分析-未来展望",
        "parent_section": "第三节",
        "subsection_patterns": [
            r'十一[、，]\s*公司未来.*发展.*展望',
            r'公司未来.*发展.*展望',
            r'十一[、，].*发展.*展望',
        ],
        "keywords": ["汇率", "外汇"],
        "context_window": 6000,
    },
    "module2": {
        "name": "衍生品投资情况",
        "parent_section": "第三节",
        "subsection_patterns": [
            r'衍生品投资情况',
            r'套期保值',
        ],
        "search_passes": [
            ["汇率", "外汇"],
            ["远期", "期权"],
            ["掉期", "互换"],
        ],
        "keywords": ["衍生品", "套期保值", "远期", "期权", "掉期", "互换", "汇率", "外汇"],
        "context_window": 5000,
    },
    "module3": {
        "name": "外币货币性项目",
        "parent_section": "第十节",
        "subsection_patterns": [
            r'\d+[、，]\s*外币货币性项目',
            r'外币货币性项目',
        ],
        "keywords": ["外币", "币种", "美元", "欧元", "港币", "日元"],
        "context_window": 1500,
    },
    "module4": {
        "name": "记账本位币",
        "parent_section": "第十节",
        "subsection_patterns": [
            r'记账本位币',
        ],
        "keywords": ["记账本位币", "境外经营实体", "本位币"],
        "context_window": 1200,
    },
    "module5": {
        "name": "与金融工具相关的风险",
        "parent_section": "第十节",
        "subsection_patterns": [
            r'与金融工具相关的风险',
        ],
        "keywords": ["汇率风险", "外汇风险", "汇率波动", "其他价格风险"],
        "context_window": 6000,
    },
}

# === ML质量评分配置 ===

# 已知的CSRC模板/锅炉板n-gram
TEMPLATE_NGRAMS = [
    "包括对于重要的境外经营实体",
    "应披露其境外主要经营地",
    "发生变化的还应披露原因",
    "除有特别说明外",
    "均以人民币元为单位表示",
    "本公司的主要经营位于中国境内",
    "利率风险利率风险",
    "信用风险信用风险",
    "流动性风险流动性风险",
    "企业内部控制基本规范",
    "企业内部控制应用指引",
    "企业内部控制评价指引",
    "上市公司内部控制指引",
    "证券交易所股票上市规则",
    "报告期内以套期保值为目的的衍生品投资",
    "独立董事对公司衍生品投资及风险控制情况的专项意见",
    "遵循企业会计准则的声明",
    "财务报表符合企业会计准则的要求",
    "会计期间会计年度自公历",
    "营业周期正常营业周期是指",
    "重要性标准确定方法和选择依据",
]

# 质量阈值
QUALITY_THRESHOLDS = {
    "table_residue_digit_ratio": 0.45,
    "boilerplate_min_score": 0.6,
    "min_text_length": {
        "module1": 20, "module2": 15, "module3": 2,
        "module4": 10, "module5": 30, "policy_info": 5,
    },
    "truncation_patterns": [
        r'[，,、]\s*$',
        r'(?:单位[：:]|币\s*种)\s*$',
        r'列示如下[：:]?\s*$',
    ],
}

# 各模块质量维度权重
MODULE_QUALITY_WEIGHTS = {
    "module1": {"completeness": 0.25, "relevance": 0.25, "cleanliness": 0.20,
                "boilerplate": 0.15, "informativeness": 0.15},
    "module2": {"completeness": 0.20, "relevance": 0.30, "cleanliness": 0.15,
                "boilerplate": 0.20, "informativeness": 0.15},
    "module3": {"completeness": 0.30, "relevance": 0.30, "cleanliness": 0.15,
                "boilerplate": 0.10, "informativeness": 0.15},
    "module4": {"completeness": 0.15, "relevance": 0.35, "cleanliness": 0.15,
                "boilerplate": 0.25, "informativeness": 0.10},
    "module5": {"completeness": 0.20, "relevance": 0.30, "cleanliness": 0.20,
                "boilerplate": 0.15, "informativeness": 0.15},
    "policy_info": {"completeness": 0.10, "relevance": 0.40, "cleanliness": 0.15,
                    "boilerplate": 0.25, "informativeness": 0.10},
}

# Policy提取黑名单（CSRC通用文档，非公司特定外汇制度）
POLICY_BLACKLIST_PHRASES = [
    "企业内部控制基本规范",
    "企业内部控制应用指引",
    "企业内部控制评价指引",
    "上市公司内部控制指引",
    "上市公司治理准则",
    "公开发行证券的公司信息披露",
    "内部控制审计报告",
    "内部控制评价报告",
    "内部控制自我评价报告",
    "内部控制规范",
    "年度内部控制",
    "内部控制审计",
    "审计报告",
    "自我评价报告",
    "信息披露编报规则",
    "上市公司章程指引",
    "股东大会规则",
    "董事会议事规则",
    "监事会议事规则",
    "独立董事",
    "关联交易",
    "对外担保",
    "募集资金",
]

# 反馈存储路径
FEEDBACK_STORE_PATH = os.path.join(OUTPUT_DIR, "feedback.jsonl")
AUTO_APPLY_THRESHOLD_CONFIDENCE = 0.7

# === ML模型配置（v2 语义嵌入方案）===

# 嵌入模型选择
# 推荐：shibing624/text2vec-base-chinese (轻量中文专用)
# 备选：BAAI/bge-small-zh-v1.5 (更高精度，稍慢)
# 备选：paraphrase-multilingual-MiniLM-L12-v2 (多语言，最轻量)
ML_EMBEDDING_MODEL = "shibing624/text2vec-base-chinese"

# 是否启用语义ML评分（True=使用嵌入模型，False=仅TF-IDF回退）
ML_USE_EMBEDDINGS = True

# 少样本分类器配置
ML_FEWSHOT_CONFIDENCE_THRESHOLD = 0.3   # 分类置信度阈值
ML_FEWSHOT_MIN_EXAMPLES = 2             # 每类最少样本数

# 主动学习配置
ML_ACTIVE_LEARNING = True               # 是否启用主动学习
ML_UNCERTAINTY_THRESHOLD = 0.4          # 不确定性阈值
ML_MAX_REVIEW_SAMPLES = 20              # 每批最大审核样本数

# ML模型存储路径
ML_MODEL_DIR = os.path.join(OUTPUT_DIR, "ml_models")

# 数据增强配置
ML_DATA_AUGMENTATION = True             # 是否启用数据增强
ML_AUGMENT_FACTOR = 3                   # 增强倍数

# ML与启发式融合权重
ML_FUSION_WEIGHT = 0.6                  # ML语义分权重（剩余为启发式分权重）
ML_FUSION_MIN_CONFIDENCE = 0.5          # ML最低置信度才参与融合

# 年报10节标准结构
SECTION_PATTERNS = [
    (r'第一节\s*重要提示[、，].*目录.*释义', '第一节'),
    (r'第二节\s*公司简介.*主要财务指标', '第二节'),
    (r'第三节\s*管理层讨论与分析', '第三节'),
    (r'第四节\s*公司治理', '第四节'),
    (r'第五节\s*环境和社会责任', '第五节'),
    (r'第六节\s*重要事项', '第六节'),
    (r'第七节\s*股份变动及股东情况', '第七节'),
    (r'第八节\s*优先股相关情况', '第八节'),
    (r'第九节\s*债券相关情况', '第九节'),
    (r'第十节\s*(?:财务报告|监事会报告)', '第十节'),
]

