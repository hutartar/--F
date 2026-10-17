"""
LLM文本精炼器（v1）：使用Claude API清理和优化提取的中文财务文本
"""
import os
import re
from typing import List, Dict, Optional

import anthropic

# 模块特定的精炼提示词
REFINEMENT_PROMPTS = {
    "module1": """你是一个中国上市公司年报文本的清理助手。请清理以下从"管理层讨论与分析→公司未来发展的展望"部分提取的文字。

清理规则：
1. 删除表格残片（数字列、对齐的数值行）
2. 删除页眉页脚残留（"XXX股份有限公司"、"年度报告全文"等）
3. 删除不相关的董事会会议记录或投票记录
4. 保留所有关于汇率、外汇风险的实质性讨论
5. 保留原文的段落结构和标题层级
6. 不要添加任何解释或总结

返回清理后的纯文本，不要加任何说明。""",

    "module2": """你是一个中国上市公司年报文本的清理助手。请清理以下从"衍生品投资情况"部分提取的文字。

清理规则：
1. 删除表格残片（投资额度表、损益数据表等纯数据行）
2. 删除董事会会议记录、投票表决等无关内容
3. 删除商品期货（铜、铝、PTA等）相关内容，只保留外汇衍生品
4. 删除股票期权、股权激励相关内容
5. 保留衍生品投资类型、套期保值效果、风险分析、独立董事意见等实质信息
6. 如果公司明确声明"报告期不存在衍生品投资"，保留该声明
7. 保留原文的段落结构
8. 不要添加任何解释

返回清理后的纯文本，不要加任何说明。""",

    "module3": """你是一个中国上市公司年报文本的清理助手。请清理以下从"外币货币性项目"部分提取的币种名称。

清理规则：
1. 只保留币种名称（如美元、欧元、港币等），用顿号或逗号分隔
2. 删除表格标题、列名、数字等
3. 删除重复的币种
4. 如果原始文字显示"无"或没有外币，返回"无"
5. 保持简洁：只返回币种列表或"无"
6. 不要添加任何解释

返回清理后的纯文本，不要加任何说明。""",

    "module4": """你是一个中国上市公司年报文本的清理助手。请清理以下从"记账本位币/境外经营实体"部分提取的文字。

清理规则：
1. 删除会计政策锅炉板文字（"遵循企业会计准则"、"会计期间"、"营业周期"等）
2. 删除"重要性标准确定方法和选择依据"及其后续表格
3. 删除模板/指南文字（"包括对于重要的境外经营实体，应披露其境外主要经营地..."）
4. 只保留：公司及子公司的记账本位币是什么，境外经营实体的记账本位币选择依据
5. 如果公司仅以人民币为记账本位币且无境外经营，保留该简短声明
6. 保留原文的实质内容
7. 不要添加任何解释

返回清理后的纯文本，不要加任何说明。""",

    "module5": """你是一个中国上市公司年报文本的清理助手。请清理以下从"与金融工具相关的风险→汇率风险"部分提取的文字。

清理规则：
1. 删除利率风险、信用风险、流动性风险等其他风险子节的内容
2. 删除表格残片（敏感性分析表、货币资金明细表等纯数据行列）
3. 删除"本公司的主要经营位于中国境内..."等通用性前言（如果后面有公司具体内容则保留前言）
4. 只保留汇率风险/外汇风险相关讨论
5. 保留"市场风险"父标题（如有）
6. 保留具体的汇率敏感性数值、外币资产负债金额等定量信息
7. 保留公司采取的外汇风险管理措施
8. 不要添加任何解释

返回清理后的纯文本，不要加任何说明。""",

    "policy_info": """你是一个中国上市公司年报文本的清理助手。请清理以下提取的制度信息。

清理规则：
1. 删除CSRC/交易所强制要求的基本规范文档名（如《企业内部控制基本规范》、《企业内部控制应用指引》等）
2. 删除内部控制自我评价报告、审计报告等定期披露文档
3. 只保留公司自己制定的与外汇、衍生品、套期保值相关的具体管理制度
4. 如果有多个相关制度，用分号分隔
5. 如果一个外汇/衍生品相关制度都没有，返回空字符串
6. 不要添加任何解释

返回清理后的纯文本，不要加任何说明。""",
}


class LLMRefiner:
    """使用Claude API精炼提取的财务文本"""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-5-20250514"):
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.refinement_count = 0
        self.cost_estimate = 0  # 估算token消耗

    def refine_module(self, module_name: str, text: str,
                      company_name: str = "") -> str:
        """精炼单个模块的提取文本"""
        if not text or len(text.strip()) < 5:
            return text

        prompt_template = REFINEMENT_PROMPTS.get(module_name, REFINEMENT_PROMPTS["module5"])

        full_prompt = f"""{prompt_template}

公司名称：{company_name}

原始文本：
\`\`\`
{text[:4000]}
\`\`\`"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=3000,
                temperature=0.0,
                system="你是一个专业的中文年报文本清理助手。只返回清理后的文本，不加任何解释、标记或总结。",
                messages=[{"role": "user", "content": full_prompt}],
            )

            refined = response.content[0].text.strip()

            # 计算估算成本（input: ~$3/M tokens, output: ~$15/M tokens）
            input_chars = len(full_prompt)
            output_chars = len(refined)
            input_tokens_est = input_chars / 1.5
            output_tokens_est = output_chars / 1.5

            self.refinement_count += 1
            self.cost_estimate += (input_tokens_est / 1_000_000 * 3 +
                                  output_tokens_est / 1_000_000 * 15)

            # 如果返回结果与原文本差异太大（可能LLM出错），保留原文本
            if len(refined) < len(text) * 0.05 and len(text) > 30:
                return text

            return refined

        except Exception as e:
            print(f"  [LLM ERROR] {module_name} {company_name}: {e}")
            return text

    def batch_refine(self, results: List[Dict],
                     refine_all: bool = False) -> List[Dict]:
        """批量精炼所有LOW/MEDIUM置信度的结果"""
        refined_count = 0

        for record in results:
            quality = record.get("_quality", {})
            record["_refined_modules"] = []

            for mod in ["module1", "module2", "module3", "module4", "module5", "policy_info"]:
                text = record.get(mod, "")
                if not text:
                    continue

                mod_quality = quality.get(mod, {})
                confidence = mod_quality.get("confidence", "MEDIUM")

                should_refine = refine_all or confidence in ("LOW", "MEDIUM")

                if should_refine:
                    original = text
                    refined = self.refine_module(
                        mod, text,
                        company_name=str(record.get("name", ""))
                    )

                    # 检查是否实际发生了变化
                    if refined != original and len(refined.strip()) > 0:
                        record[mod] = refined
                        record["_refined_modules"].append(mod)
                        refined_count += 1

        print(f"  LLM精炼: {refined_count} 个模块已优化，估算成本 ${self.cost_estimate:.4f}")
        return results


if __name__ == "__main__":
    # 简单测试（需要ANTHROPIC_API_KEY）
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("请设置 ANTHROPIC_API_KEY 环境变量")
    else:
        refiner = LLMRefiner(api_key=api_key)
        test = "4、记账本位币本公司及境内子公司采用人民币为记账本位币。1、遵循企业会计准则的声明本财务报表符合企业会计准则的要求，真实、完整地反映了公司的财务状况。"
        result = refiner.refine_module("module4", test, "测试公司")
        print(f"原文: {test[:100]}...")
        print(f"精炼: {result[:200]}...")
