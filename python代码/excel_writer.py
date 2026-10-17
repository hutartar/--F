"""
生成Excel输出文件（v3-ML：支持质量评分、条件格式、跨模块验证）
"""
import os
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter
from config import OUTPUT_COLUMNS, RESULT_TO_COLUMN, OUTPUT_DIR


# 条件格式化颜色
QUALITY_FILLS = {
    "HIGH": PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid"),    # 绿色
    "MEDIUM": PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid"),  # 黄色
    "LOW": PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid"),     # 红色
}


def create_output_excel(results, output_path=None):
    """将提取结果写入独立Excel（含质量评分和跨模块验证）"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if output_path is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUT_DIR, f"外汇风险管理提取结果_{timestamp}.xlsx")

    wb = openpyxl.Workbook()

    # ── Sheet 1: 数据表格 ──
    ws = wb.active
    ws.title = "外汇风险管理提取结果"

    # 样式
    header_font = Font(name="微软雅黑", size=10, bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    header_alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    cell_alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
    thin_border = Border(
        left=Side(style="thin"), right=Side(style="thin"),
        top=Side(style="thin"), bottom=Side(style="thin"),
    )

    # 动态列定义：原始列 + 质量列 + 验证列
    quality_columns = {
        "M1_Quality": 20, "M2_Quality": 21, "M3_Quality": 22,
        "M4_Quality": 23, "M5_Quality": 24, "Policy_Quality": 25,
    }
    cross_val_column = 26

    base_headers = list(OUTPUT_COLUMNS)  # 原始19列
    extra_headers = list(quality_columns.keys()) + ["Cross_Validation"]
    all_headers = base_headers + extra_headers

    # 质量列号 → 模块映射
    quality_col_to_module = {
        20: "module1", 21: "module2", 22: "module3",
        23: "module4", 24: "module5", 25: "policy_info",
    }

    # 写表头
    for col_idx, col_name in enumerate(all_headers, 1):
        cell = ws.cell(row=1, column=col_idx, value=col_name)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_alignment
        cell.border = thin_border

    # 写数据行
    for row_idx, record in enumerate(results, 2):
        # 原始数据列
        col_values = {
            1: record.get("code", ""),
            2: record.get("name", ""),
            3: record.get("year", ""),
        }
        for key, col in RESULT_TO_COLUMN.items():
            col_values[col] = record.get(key, "")

        for col_idx in range(1, len(base_headers) + 1):
            val = col_values.get(col_idx, "")
            cell = ws.cell(row=row_idx, column=col_idx, value=val)
            cell.alignment = cell_alignment
            cell.border = thin_border
            cell.font = Font(name="微软雅黑", size=9)

        # 质量评分列
        quality_meta = record.get("_quality", {})
        for col_idx, mod_name in quality_col_to_module.items():
            mod_quality = quality_meta.get(mod_name, {})
            score = mod_quality.get("overall_score", 0)
            conf = mod_quality.get("confidence", "N/A")
            flags = mod_quality.get("flags", [])

            if score > 0:
                display = f"{score:.0f}/{conf}"
                if flags:
                    display += f" [{', '.join(flags)}]"
            else:
                display = conf if record.get(mod_name, "") else "-"

            cell = ws.cell(row=row_idx, column=col_idx, value=display)
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = thin_border
            cell.font = Font(name="微软雅黑", size=8)

            # 条件格式
            if conf in QUALITY_FILLS:
                cell.fill = QUALITY_FILLS[conf]

        # 跨模块验证列
        cross_val_results = record.get("_cross_validation", [])
        if cross_val_results:
            val_text = "; ".join(
                f"[{r['severity']}] {r['rule']}: {r['description'][:80]}"
                for r in cross_val_results
            )
        else:
            val_text = ""
        cell = ws.cell(row=row_idx, column=cross_val_column, value=val_text)
        cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
        cell.border = thin_border
        cell.font = Font(name="微软雅黑", size=8)

    # 列宽
    base_widths = {
        1: 10, 2: 14, 3: 8,
        4: 55, 5: 55, 6: 50, 7: 50, 8: 55,
        9: 16, 10: 45, 11: 20,
        12: 10, 13: 10, 14: 10,
        15: 10, 16: 10, 17: 10,
        18: 10, 19: 10,
    }
    for col_idx, width in base_widths.items():
        ws.column_dimensions[get_column_letter(col_idx)].width = width
    # 质量列宽
    for col_idx in quality_columns.values():
        ws.column_dimensions[get_column_letter(col_idx)].width = 20
    # 验证列宽
    ws.column_dimensions[get_column_letter(cross_val_column)].width = 45

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(all_headers))}{len(results) + 1}"

    # ── Sheet 2: 质量摘要 ──
    ws2 = wb.create_sheet("Quality_Summary")

    # 标题
    ws2.cell(row=1, column=1, value="质量评分摘要").font = Font(name="微软雅黑", size=14, bold=True)
    ws2.merge_cells("A1:F1")

    # 表头
    summary_headers = ["模块", "HIGH数量", "MEDIUM数量", "LOW数量", "平均分", "常见标志"]
    for i, h in enumerate(summary_headers, 1):
        cell = ws2.cell(row=3, column=i, value=h)
        cell.font = Font(name="微软雅黑", size=10, bold=True, color="FFFFFF")
        cell.fill = header_fill
        cell.alignment = header_alignment
        cell.border = thin_border

    # 统计每个模块的质量
    module_names = ["module1", "module2", "module3", "module4", "module5", "policy_info"]
    module_labels = ["M1-展望", "M2-衍生品", "M3-外币项目", "M4-记账本位币", "M5-汇率风险", "Policy-制度"]

    for row_offset, (mod_name, mod_label) in enumerate(zip(module_names, module_labels)):
        scores = []
        conf_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        all_flags = []

        for record in results:
            q = record.get("_quality", {}).get(mod_name, {})
            s = q.get("overall_score", 0)
            if s > 0:
                scores.append(s)
            conf = q.get("confidence", "LOW")
            if conf in conf_counts:
                conf_counts[conf] += 1
            all_flags.extend(q.get("flags", []))

        avg_score = sum(scores) / len(scores) if scores else 0
        # 最常见的3个标志
        from collections import Counter
        top_flags = [f"{flag}({cnt})" for flag, cnt in Counter(all_flags).most_common(3)]

        row = 4 + row_offset
        ws2.cell(row=row, column=1, value=mod_label).font = Font(name="微软雅黑", size=9)
        ws2.cell(row=row, column=2, value=conf_counts["HIGH"])
        ws2.cell(row=row, column=3, value=conf_counts["MEDIUM"])
        ws2.cell(row=row, column=4, value=conf_counts["LOW"])
        ws2.cell(row=row, column=5, value=f"{avg_score:.1f}")
        ws2.cell(row=row, column=6, value=", ".join(top_flags))

        for col in range(1, 7):
            ws2.cell(row=row, column=col).border = thin_border
            ws2.cell(row=row, column=col).alignment = Alignment(horizontal="center", vertical="center")

    # 列宽
    ws2.column_dimensions['A'].width = 18
    ws2.column_dimensions['B'].width = 12
    ws2.column_dimensions['C'].width = 14
    ws2.column_dimensions['D'].width = 12
    ws2.column_dimensions['E'].width = 10
    ws2.column_dimensions['F'].width = 50

    # ── Sheet 3: 跨模块验证摘要 ──
    ws3 = wb.create_sheet("Cross_Validation")
    ws3.cell(row=1, column=1, value="跨模块验证结果").font = Font(name="微软雅黑", size=14, bold=True)
    ws3.merge_cells("A1:E1")

    val_headers = ["公司代码", "公司名称", "严重度", "规则", "说明"]
    for i, h in enumerate(val_headers, 1):
        cell = ws3.cell(row=3, column=i, value=h)
        cell.font = Font(name="微软雅黑", size=10, bold=True, color="FFFFFF")
        cell.fill = header_fill
        cell.alignment = header_alignment
        cell.border = thin_border

    val_row = 4
    for record in results:
        cross_results = record.get("_cross_validation", [])
        if not cross_results:
            continue
        for cr in cross_results:
            ws3.cell(row=val_row, column=1, value=record.get("code", ""))
            ws3.cell(row=val_row, column=2, value=record.get("name", ""))
            ws3.cell(row=val_row, column=3, value=cr["severity"])
            ws3.cell(row=val_row, column=4, value=cr["rule"])
            ws3.cell(row=val_row, column=5, value=cr["description"])
            for col in range(1, 6):
                ws3.cell(row=val_row, column=col).border = thin_border
                ws3.cell(row=val_row, column=col).font = Font(name="微软雅黑", size=9)
                ws3.cell(row=val_row, column=col).alignment = Alignment(horizontal="left", vertical="center")
            # 严重度颜色
            sev = cr["severity"]
            if sev == "WARNING":
                ws3.cell(row=val_row, column=3).fill = QUALITY_FILLS["MEDIUM"]
            elif sev == "ERROR":
                ws3.cell(row=val_row, column=3).fill = QUALITY_FILLS["LOW"]
            val_row += 1

    if val_row == 4:
        ws3.cell(row=4, column=1, value="无跨模块矛盾").font = Font(name="微软雅黑", size=9, color="666666")

    ws3.column_dimensions['A'].width = 12
    ws3.column_dimensions['B'].width = 14
    ws3.column_dimensions['C'].width = 10
    ws3.column_dimensions['D'].width = 30
    ws3.column_dimensions['E'].width = 60

    wb.save(output_path)
    print(f"  独立输出已保存: {output_path}")
    return output_path


def fill_target_excel(results, template_path, output_path):
    """
    直接将提取结果填充到目标Excel模板中（含质量标注）。
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    wb = openpyxl.load_workbook(template_path)
    ws = wb['Sheet1']

    result_map = {}
    for r in results:
        code = str(r.get("code", ""))
        result_map[code] = r

    col_mapping = {
        "module1": 3, "module2": 4, "module3": 5,
        "module4": 6, "module5": 7, "policy_info": 9,
    }

    filled_count = 0
    for row_idx in range(2, ws.max_row + 1):
        code_cell = ws.cell(row=row_idx, column=1)
        code = str(code_cell.value).strip() if code_cell.value else ""

        if code in result_map:
            r = result_map[code]
            for field, col in col_mapping.items():
                val = r.get(field, "")
                if val:
                    ws.cell(row=row_idx, column=col).value = val
                    filled_count += 1

    wb.save(output_path)
    print(f"  目标模板填充完成: {output_path} (填充了 {filled_count} 个字段)")
    return output_path


if __name__ == "__main__":
    test_results = [
        {
            "code": "3000", "name": "劲仔食品", "year": "2023",
            "module1": "汇率波动风险：公司出口业务面临汇率波动...",
            "module2": "公司报告期不存在衍生品投资。",
            "module3": "美元、先令",
            "module4": "境外子公司以先令为记账本位币。",
            "module5": "汇率风险：本集团承受外汇风险主要与美元、先令有关...",
            "policy_info": "《企业内部控制基本规范》",
            "_quality": {
                "module1": {"overall_score": 72, "confidence": "HIGH", "flags": []},
                "module2": {"overall_score": 95, "confidence": "HIGH", "flags": []},
                "module3": {"overall_score": 88, "confidence": "HIGH", "flags": []},
                "module4": {"overall_score": 45, "confidence": "MEDIUM", "flags": ["ACCOUNTING_POLICY_MIXED"]},
                "module5": {"overall_score": 91, "confidence": "HIGH", "flags": []},
                "policy_info": {"overall_score": 25, "confidence": "LOW", "flags": ["CSRC_GENERIC_ONLY"]},
            },
            "_overall_quality": 69.3,
            "_cross_validation": [
                {"severity": "INFO", "rule": "POLICY_CSRC_ONLY", "description": "制度信息仅包含CSRC通用文档"},
            ],
        }
    ]
    create_output_excel(test_results, os.path.join(OUTPUT_DIR, "test_output_ml.xlsx"))
