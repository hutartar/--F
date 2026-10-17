"""
中文年报文本数据增强（v1）：解决小样本学习中的样本不足问题

方法：
1. 同义词替换 - 使用领域同义词词典
2. 回译增强 - 中文→英文→中文（需要网络）
3. 随机插入/删除 - 轻微扰动
4. 段落重组 - 对多段落文本进行重新排列组合
5. 模板生成 - 基于已知模式生成变体
"""
import re
import random
import jieba
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict


class ChineseFinancialTextAugmenter:
    """
    中文金融文本增强器

    为年报提取结果生成变体，扩充训练样本
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)

        # 金融领域同义词词典
        self.synonym_dict = {
            "汇率": ["外汇汇率", "兑换率", "汇价"],
            "风险": ["不确定性", "敞口", "隐患"],
            "波动": ["变动", "震荡", "起伏"],
            "衍生品": ["金融衍生工具", "衍生金融产品"],
            "套期保值": ["对冲", "避险", "风险对冲"],
            "远期": ["远期合约", "远期协议"],
            "期权": ["选择权", "期权合约"],
            "互换": ["掉期", "互换协议"],
            "美元": ["美金", "USD"],
            "欧元": ["EUR"],
            "港币": ["港元", "HKD"],
            "人民币": ["CNY", "本币"],
            "记账本位币": ["功能货币", "本位币"],
            "境外经营": ["海外经营", "境外业务"],
            "子公司": ["附属公司", "下属公司"],
            "管理层": ["经营层", "管理团队"],
            "展望": ["前景", "趋势", "预判"],
            "投资": ["出资", "投入"],
            "损失": ["亏损", "亏蚀"],
            "收益": ["盈利", "回报", "利润"],
        }

        # 停用词（不要替换的词）
        self.stop_words = {
            "的", "了", "在", "是", "我", "有", "和", "就",
            "不", "人", "都", "一", "一个", "上", "也", "很",
            "到", "说", "要", "去", "你", "会", "着", "没有",
            "看", "好", "自己", "这", "他", "她", "它", "们",
        }

        # 连接词
        self.connectors = [
            "。", "；", "；此外，", "。同时，", "。另外，",
            "。值得注意的是，", "。需要关注的是，", "。",
        ]

    def synonym_replace(self, text: str, replace_ratio: float = 0.15) -> str:
        """
        同义词替换增强

        Args:
            text: 原始文本
            replace_ratio: 替换比例（0.0-1.0），建议0.1-0.2

        Returns:
            增强后的文本
        """
        if not text:
            return text

        words = list(jieba.cut(text))
        replaceable_indices = [
            i for i, w in enumerate(words)
            if w in self.synonym_dict and w not in self.stop_words
        ]

        if not replaceable_indices:
            return text

        n_replace = max(1, int(len(replaceable_indices) * replace_ratio))
        to_replace = random.sample(
            replaceable_indices,
            min(n_replace, len(replaceable_indices))
        )

        new_words = list(words)
        for idx in to_replace:
            word = words[idx]
            synonyms = self.synonym_dict[word]
            new_word = random.choice(synonyms)
            new_words[idx] = new_word

        return ''.join(new_words)

    def random_insert(self, text: str, n_insert: int = 1) -> str:
        """随机插入领域相关的修饰词"""
        if not text:
            return text

        insert_words = [
            "一定的", "较大的", "潜在的", "相关的",
            "明显的", "较高的", "显著的", "持续的",
            "合理的", "必要的", "有效的",
        ]

        sentences = re.split(r'[。；\n]', text)
        if not sentences:
            return text

        for _ in range(n_insert):
            idx = random.randint(0, len(sentences) - 1)
            if len(sentences[idx]) > 10:
                word = random.choice(insert_words)
                pos = random.randint(0, min(len(sentences[idx]), 20))
                sentences[idx] = (
                    sentences[idx][:pos] + word + sentences[idx][pos:]
                )

        return '；'.join(s for s in sentences if s)

    def random_delete(self, text: str, delete_ratio: float = 0.05) -> str:
        """随机删除少量字符（模拟OCR/PDF提取噪声）"""
        if not text or len(text) < 20:
            return text

        chars = list(text)
        n_delete = max(1, int(len(chars) * delete_ratio))
        for _ in range(n_delete):
            idx = random.randint(0, len(chars) - 1)
            if chars[idx] not in '，。；：""''（）《》':
                chars[idx] = ''

        result = ''.join(chars)
        # 清理多余空格
        result = re.sub(r'\s+', ' ', result)
        return result.strip()

    def paragraph_shuffle(self, text: str) -> str:
        """
        段落顺序随机重排（适用于多段落提取结果）

        注意：这会改变原文顺序，适合作为负样本增强或多样性增强
        """
        paragraphs = re.split(r'\n\s*\n', text)
        if len(paragraphs) < 2:
            return text

        random.shuffle(paragraphs)
        return '\n\n'.join(paragraphs)

    def sentence_variation(self, text: str) -> str:
        """
        生成句子的轻微变体（改变表达方式）

        用于生成与原始文本语义相同但表达不同的训练样本
        """
        if not text:
            return text

        # 策略1：移动"公司"、"本集团"等主语
        variations = []
        text_v = text

        # 将"公司XXX"变为"XXX，公司"
        text_v = re.sub(
            r'(?:公司|本集团|本公司)(.{5,30}?)(?:。|；)',
            lambda m: m.group(1).strip('，,') + '，' + m.group(0)[:len(m.group(0))-len(m.group(1))],
            text_v
        )

        # 将"包括X、Y、Z"变为"如X、Y、Z等"
        text_v = re.sub(r'包括([^。；]{5,40})', r'如\1等', text_v)

        return text_v

    def back_translate(self, text: str) -> Optional[str]:
        """
        回译增强：中文→英文→中文

        注意：需要安装 deep-translator 或类似库
        当前为简化实现，使用基本的翻译思路
        """
        # 回译需要额外的翻译API，这里提供接口但不强制依赖
        # 用户可自行安装: pip install deep-translator
        try:
            from deep_translator import GoogleTranslator
            en = GoogleTranslator(source='zh-CN', target='en').translate(text)
            zh = GoogleTranslator(source='en', target='zh-CN').translate(en)
            return zh
        except ImportError:
            return None
        except Exception:
            return None

    def augment(
        self,
        text: str,
        n_variants: int = 5,
        include_back_translate: bool = False,
    ) -> List[str]:
        """
        为一条文本生成多个增强变体

        Args:
            text: 原始文本
            n_variants: 生成变体数量
            include_back_translate: 是否包含回译（慢，需要网络）

        Returns:
            变体文本列表（包含原始文本）
        """
        if not text or len(text) < 10:
            return [text] if text else []

        variants = [text]
        methods_used = 0

        # 1. 同义词替换（不同替换比例）
        for ratio in [0.1, 0.2, 0.3]:
            if len(variants) >= n_variants:
                break
            var = self.synonym_replace(text, replace_ratio=ratio)
            if var != text and var not in variants:
                variants.append(var)
                methods_used += 1

        # 2. 随机插入
        if len(variants) < n_variants:
            var = self.random_insert(text, n_insert=random.randint(1, 2))
            if var != text and var not in variants:
                variants.append(var)
                methods_used += 1

        # 3. 句子变体
        if len(variants) < n_variants:
            var = self.sentence_variation(text)
            if var != text and var not in variants:
                variants.append(var)
                methods_used += 1

        # 4. 随机删除（模拟噪声，生成变体）
        if len(variants) < n_variants:
            var = self.random_delete(text, delete_ratio=0.03)
            if var != text and var not in variants:
                variants.append(var)
                methods_used += 1

        # 5. 回译（如果需要）
        if include_back_translate and len(variants) < n_variants:
            var = self.back_translate(text)
            if var and var != text and var not in variants:
                variants.append(var)
                methods_used += 1

        return variants

    def augment_batch(
        self,
        texts: List[str],
        labels: List[str],
        augmentation_factor: int = 3,
    ) -> Tuple[List[str], List[str]]:
        """
        批量增强训练数据

        Args:
            texts: 原始文本列表
            labels: 对应的标签列表
            augmentation_factor: 每条文本生成的变体数量

        Returns:
            (augmented_texts, augmented_labels)
        """
        all_texts = []
        all_labels = []

        for text, label in zip(texts, labels):
            variants = self.augment(text, n_variants=augmentation_factor + 1)
            all_texts.extend(variants)
            all_labels.extend([label] * len(variants))

        print(f"  [ML] 数据增强: {len(texts)} → {len(all_texts)} 条 "
              f"(factor={augmentation_factor})")

        return all_texts, all_labels


class TemplateGenerator:
    """
    基于模板生成领域特定的合成训练样本

    特别适用于生成"负面"样本（如各种锅炉板变体）
    """

    def __init__(self):
        # 会计政策模板
        self.accounting_templates = [
            "{num}、{topic}{company}遵循企业会计准则的要求{action}。",
            "{num}、{topic}财政部颁布的{standard}要求{action}。",
            "{num}、{topic}本财务报表{action}。",
        ]

        self.accounting_topics = [
            "遵循企业会计准则的声明",
            "会计期间",
            "营业周期",
            "重要性标准确定方法和选择依据",
            "记账本位币",
            "外币业务核算方法",
        ]

        self.companies = [
            "本公司", "本集团", "公司",
        ]

        self.standard_actions = [
            "真实、完整地反映了公司的财务状况、经营成果和现金流量",
            "符合企业会计准则的规定",
            "在所有重大方面公允反映了公司的财务状况",
            "按照企业会计准则的规定编制",
        ]

        # 风险披露模板
        self.risk_templates = [
            "金融工具的市场风险包括{risks}。",
            "{company}面临的主要金融风险包括{risks}。",
            "本集团承受的{risk_type}主要来源于{source}。",
        ]

        self.risk_types = [
            "汇率风险", "外汇风险", "利率风险", "信用风险", "流动性风险",
        ]

        self.risk_sources = [
            "以美元计价的金融资产和金融负债",
            "出口销售产生的外币应收账款",
            "境外经营产生的外币净资产敞口",
            "以外币计价的银行借款",
        ]

        # 衍生品模板
        self.derivative_templates = [
            "{company}报告期{status}衍生品投资。",
            "衍生品投资类型：{types}。",
            "套期保值效果：{effect}。",
        ]

        self.derivative_status = [
            "不存在", "存在", "开展了", "未开展",
        ]

        self.derivative_types = [
            "远期外汇合约", "远期结售汇", "外汇期权",
            "货币互换", "利率互换", "商品期货",
        ]

    def generate_accounting_boilerplate(self, n: int = 10) -> List[str]:
        """生成会计政策锅炉板变体"""
        samples = []
        for _ in range(n):
            template = random.choice(self.accounting_templates)
            topic = random.choice(self.accounting_topics)
            company = random.choice(self.companies)
            action = random.choice(self.standard_actions)
            num = str(random.randint(1, 5))

            sample = template.format(
                num=num, topic=topic, company=company, action=action,
                standard="企业会计准则",
            )
            samples.append(sample)
        return samples

    def generate_risk_disclosures(self, n: int = 10) -> List[str]:
        """生成风险披露变体"""
        samples = []
        for _ in range(n):
            template = random.choice(self.risk_templates)
            n_risks = random.randint(1, 3)
            risks = '、'.join(random.sample(self.risk_types, n_risks))
            company = random.choice(self.companies)
            source = random.choice(self.risk_sources)

            sample = template.format(
                risks=risks, company=company, risk_type=random.choice(self.risk_types),
                source=source,
            )
            samples.append(sample)
        return samples

    def generate_negative_samples(self, n_per_type: int = 10) -> List[str]:
        """生成各类负面（锅炉板）样本"""
        samples = []
        samples.extend(self.generate_accounting_boilerplate(n_per_type))

        # 通用前言样本
        general_templates = [
            "以上内容详见本公司{year}年年度报告全文。",
            "本报告摘要来自年度报告全文。",
            "本公司董事会及全体董事保证本公告内容不存在任何虚假记载。",
            "投资者应当到证券交易所网站仔细阅读年度报告全文。",
        ]
        for _ in range(n_per_type):
            template = random.choice(general_templates)
            sample = template.format(year="2023")
            samples.append(sample)

        return samples


if __name__ == "__main__":
    print("=== 中文金融文本增强器测试 ===\n")

    augmenter = ChineseFinancialTextAugmenter(seed=42)

    test_texts = [
        "汇率波动风险：公司出口业务主要以美元结算，面临较大的汇率波动风险。为应对汇率风险，公司采用远期结售汇锁定汇率。",
        "公司报告期不存在衍生品投资。",
        "4、记账本位币本公司及境内子公司以人民币为记账本位币。境外子公司以美元为记账本位币。",
    ]

    for text in test_texts:
        print(f"原文: {text[:80]}...")
        variants = augmenter.augment(text, n_variants=3)
        for i, v in enumerate(variants):
            print(f"  变体{i}: {v[:80]}...")
        print()

    # 批量增强测试
    labels = ["GOOD", "GOOD", "BAD"]
    aug_texts, aug_labels = augmenter.augment_batch(
        test_texts, labels, augmentation_factor=2
    )
    print(f"增强后: {len(aug_texts)} 条样本, 标签分布: "
          f"GOOD={aug_labels.count('GOOD')}, BAD={aug_labels.count('BAD')}")

    # 模板生成测试
    print("\n=== 模板生成器测试 ===")
    gen = TemplateGenerator()
    neg_samples = gen.generate_negative_samples(3)
    for s in neg_samples:
        print(f"  负面样本: {s[:80]}...")
