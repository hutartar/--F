"""
跨模块一致性验证器（v1）：检测5个模块之间的逻辑矛盾
"""
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable


@dataclass
class CrossValidationResult:
    passed: bool
    rule_name: str
    company_code: str
    company_name: str
    modules_involved: List[str]
    description: str
    severity: str  # ERROR / WARNING / INFO


class CrossValidator:
    """跨模块逻辑一致性验证"""

    def __init__(self):
        self.rules: List[Callable] = [
            self._rule_currencies_without_risk,
            self._rule_derivatives_without_policy,
            self._rule_forex_outlook_without_risk,
            self._rule_m3_m5_currency_consistency,
            self._rule_policy_only_csrc,
        ]

    def validate(self, record: Dict) -> List[CrossValidationResult]:
        """对一条公司记录执行所有验证规则"""
        results = []
        for rule in self.rules:
            result = rule(record)
            if result is not None:
                results.append(result)
        return results

    # ── 规则1：有外币货币性项目但无汇率风险披露 ──
    def _rule_currencies_without_risk(self, record: Dict) -> Optional[CrossValidationResult]:
        m3 = record.get("module3", "")
        m5 = record.get("module5", "")

        # M3有实质币种
        has_currencies = bool(m3 and m3 != "无" and len(m3) >= 2)
        # M5为空或无实质内容
        no_risk = not m5 or len(m5.strip()) < 30

        if has_currencies and no_risk:
            return CrossValidationResult(
                passed=False,
                rule_name="CURRENCIES_WITHOUT_RISK",
                company_code=str(record.get("code", "")),
                company_name=str(record.get("name", "")),
                modules_involved=["module3", "module5"],
                description=f"M3有外币({m3[:40]})但M5无汇率风险披露",
                severity="WARNING",
            )

        # 有币种但M5只有通用前言
        if has_currencies and m5 and "市场风险" in m5 and len(m5) < 150:
            if not re.search(r'(?:汇率风险|外汇风险).{20,}', m5):
                return CrossValidationResult(
                    passed=False,
                    rule_name="CURRENCIES_WITHOUT_RISK",
                    company_code=str(record.get("code", "")),
                    company_name=str(record.get("name", "")),
                    modules_involved=["module3", "module5"],
                    description="M3有外币但M5仅有通用市场风险前言，无实质汇率风险讨论",
                    severity="INFO",
                )
        return None

    # ── 规则2：有衍生品投资但无相关制度 ──
    def _rule_derivatives_without_policy(self, record: Dict) -> Optional[CrossValidationResult]:
        m2 = record.get("module2", "")
        policy = record.get("policy_info", "")

        has_derivatives = bool(m2 and "不存在衍生品投资" not in m2 and len(m2) > 30)
        no_policy = not policy or len(policy.strip()) < 5

        if has_derivatives and no_policy:
            return CrossValidationResult(
                passed=False,
                rule_name="DERIVATIVES_WITHOUT_POLICY",
                company_code=str(record.get("code", "")),
                company_name=str(record.get("name", "")),
                modules_involved=["module2", "policy_info"],
                description="M2表明有衍生品投资但未找到相关风险管理制度",
                severity="WARNING",
            )
        return None

    # ── 规则3：展望中有汇率讨论但无汇率风险披露 ──
    def _rule_forex_outlook_without_risk(self, record: Dict) -> Optional[CrossValidationResult]:
        m1 = record.get("module1", "")
        m5 = record.get("module5", "")

        has_outlook_fx = bool(m1 and len(m1) > 30 and
                             re.search(r'(?:汇率|外汇)', m1))
        no_risk = not m5 or len(m5.strip()) < 30

        if has_outlook_fx and no_risk:
            return CrossValidationResult(
                passed=False,
                rule_name="OUTLOOK_FX_WITHOUT_RISK",
                company_code=str(record.get("code", "")),
                company_name=str(record.get("name", "")),
                modules_involved=["module1", "module5"],
                description="展望中有汇率讨论但金融工具风险部分无汇率风险",
                severity="INFO",
            )
        return None

    # ── 规则4：M3和M5币种一致性 ──
    def _rule_m3_m5_currency_consistency(self, record: Dict) -> Optional[CrossValidationResult]:
        m3 = record.get("module3", "")
        m5 = record.get("module5", "")

        if not m3 or m3 == "无" or not m5:
            return None

        # 提取M3中的币种
        m3_currencies = set(re.findall(r'(?:美元|欧元|港币|日元|韩元|英镑|澳元|越南盾'
                                       r'|新加坡元|新台币|菲律宾比索|马来西亚林吉特'
                                       r'|印尼盾|泰铢|俄罗斯卢布|巴西雷亚尔|墨西哥比索'
                                       r'|南非兰特|迪拉姆|坚戈|印度卢比|新加坡币'
                                       r'|瑞典克朗|瑞士法郎|澳门元|新西兰元|里拉|福林|先令)',
                                       m3))

        if not m3_currencies:
            return None

        # M5中提到的币种
        m5_currencies = set(re.findall(r'(?:美元|欧元|港币|日元|韩元|英镑|澳元|越南盾'
                                       r'|新加坡元|新台币|菲律宾比索|马来西亚林吉特'
                                       r'|印尼盾|泰铢|俄罗斯卢布|巴西雷亚尔|墨西哥比索'
                                       r'|南非兰特|迪拉姆|坚戈|印度卢比|新加坡币'
                                       r'|瑞典克朗|瑞士法郎|澳门元|新西兰元|里拉|福林|先令)',
                                       m5))

        if m5_currencies and m3_currencies != m5_currencies:
            only_m3 = m3_currencies - m5_currencies
            only_m5 = m5_currencies - m3_currencies
            details = []
            if only_m3:
                details.append(f"M3独有: {', '.join(sorted(only_m3))}")
            if only_m5:
                details.append(f"M5独有: {', '.join(sorted(only_m5))}")
            return CrossValidationResult(
                passed=False,
                rule_name="CURRENCY_MISMATCH",
                company_code=str(record.get("code", "")),
                company_name=str(record.get("name", "")),
                modules_involved=["module3", "module5"],
                description="M3与M5币种不一致: " + "; ".join(details),
                severity="INFO",
            )
        return None

    # ── 规则5：Policy只有CSRC通用文档 ──
    def _rule_policy_only_csrc(self, record: Dict) -> Optional[CrossValidationResult]:
        policy = record.get("policy_info", "")
        if not policy:
            return None

        from config import POLICY_BLACKLIST_PHRASES
        entries = [e.strip() for e in policy.split(';') if e.strip()]
        if not entries:
            return None

        blacklisted = sum(1 for e in entries
                        if any(bp in e for bp in POLICY_BLACKLIST_PHRASES))

        if blacklisted >= len(entries):
            return CrossValidationResult(
                passed=False,
                rule_name="POLICY_CSRC_ONLY",
                company_code=str(record.get("code", "")),
                company_name=str(record.get("name", "")),
                modules_involved=["policy_info"],
                description=f"制度信息仅包含CSRC通用文档({len(entries)}条)，无公司特定外汇/衍生品制度",
                severity="INFO",
            )
        return None


def validate_all(results: List[Dict]) -> List[Dict]:
    """对所有结果运行跨模块验证，返回增强后的结果列表"""
    validator = CrossValidator()
    total_warnings = 0
    total_infos = 0

    for record in results:
        validation_results = validator.validate(record)
        record["_cross_validation"] = [
            {
                "rule": r.rule_name,
                "severity": r.severity,
                "description": r.description,
            }
            for r in validation_results
        ]

        for r in validation_results:
            if r.severity == "WARNING":
                total_warnings += 1
            elif r.severity == "INFO":
                total_infos += 1

    print(f"  跨模块验证: {total_warnings} WARNING, {total_infos} INFO")
    return results


if __name__ == "__main__":
    # 测试
    test_record = {
        "code": "3000",
        "name": "测试公司",
        "module3": "美元、欧元、港币",
        "module5": "",
        "module2": "衍生品投资类型：远期结售汇...",
        "policy_info": "",
        "module1": "",
    }
    validator = CrossValidator()
    results = validator.validate(test_record)
    for r in results:
        print(f"  [{r.severity}] {r.rule_name}: {r.description}")
