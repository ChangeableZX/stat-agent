import math
import os
import subprocess
import sys
from datetime import datetime

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from state import CategoricalPlan, CorrelationPlan, MultiGroupPlan, set_current_plan
from tools import (
    check_bivariate_normality,
    check_expected_frequencies,
    check_linearity,
    check_normality,
    check_variance_equality,
    load_data,
    run_chi_square,
    run_fisher_exact,
    run_one_way_anova,
    run_pearson,
    run_posthoc_tukey,
    select_method,
)


REPORT_PATH = os.path.join("docs", "test-logs", "textbook-validation.md")


def rel_close(actual, expected, tol=0.005):
    return abs(actual - expected) / abs(expected) < tol


def pearson_ci(r_value, n, alpha=0.05):
    z = np.arctanh(r_value)
    se = 1 / math.sqrt(n - 3)
    zcrit = 1.959963984540054
    return [float(np.tanh(z - zcrit * se)), float(np.tanh(z + zcrit * se))]


def commit_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def add_check(rows, case, dataset, expected_method, actual_method, expected, actual, passed, details):
    rows.append(
        {
            "case": case,
            "dataset": dataset,
            "expected_method": expected_method,
            "actual_method": actual_method,
            "expected": expected,
            "actual": actual,
            "passed": passed,
            "details": details,
        }
    )


def validate_plantgrowth(rows):
    plan = MultiGroupPlan(
        research_question="PlantGrowth ANOVA",
        intent="compare_multi_groups",
        target_variable="weight",
        grouping_variable="group",
        group_levels=["ctrl", "trt1", "trt2"],
        design="independent",
    )
    set_current_plan(plan)
    load_data("data/textbook_anova_plantgrowth.csv")
    for group in ["ctrl", "trt1", "trt2"]:
        check_normality("weight", "group", group)
    check_variance_equality("weight", "group")
    selected = select_method()
    result = run_one_way_anova("weight", "group")
    tukey = run_posthoc_tukey("weight", "group")

    group_stats = result["group_stats"]
    df_between = len(group_stats) - 1
    df_within = sum(item["n"] for item in group_stats.values()) - len(group_stats)
    method_ok = selected["selected_method"] == "one_way_anova"
    f_ok = rel_close(result["statistic"], 4.846)
    p_ok = rel_close(result["p_value"], 0.01591)
    df_ok = df_between == 2 and df_within == 27
    add_check(
        rows,
        "1.1",
        "PlantGrowth",
        "one_way_anova",
        selected["selected_method"],
        "F=4.846, p=0.01591, df=(2,27)",
        f"F={result['statistic']:.6g}, p={result['p_value']:.6g}, df=({df_between},{df_within})",
        method_ok and f_ok and p_ok and df_ok,
        {"anova": result},
    )

    comparisons = tukey.get("comparisons", [])
    lookup = {}
    for item in comparisons:
        a = item.get("A")
        b = item.get("B")
        if a and b:
            lookup[(a, b)] = item
            lookup[(b, a)] = item
    trt2_trt1 = lookup.get(("trt1", "trt2"), {})
    diff = trt2_trt1.get("diff")
    p_adj = trt2_trt1.get("p_tukey", trt2_trt1.get("p-tukey"))
    if diff is not None and trt2_trt1.get("A") == "trt1":
        diff = -diff
    tukey_ok = diff is not None and p_adj is not None and rel_close(diff, 0.865) and rel_close(p_adj, 0.012)
    add_check(
        rows,
        "1.2",
        "PlantGrowth Tukey",
        "tukey_hsd",
        tukey.get("method", "tukey_hsd"),
        "trt2-trt1 diff=0.865, p_adj=0.012",
        f"trt2-trt1 diff={diff}, p_adj={p_adj}",
        tukey_ok,
        {"tukey": tukey},
    )


def validate_anscombe(rows):
    plan = CorrelationPlan(
        research_question="Anscombe quartet I Pearson",
        intent="correlation",
        x_variable="x",
        y_variable="y",
    )
    set_current_plan(plan)
    load_data("data/textbook_corr_anscombe.csv")
    check_bivariate_normality("x", "y")
    check_linearity("x", "y")
    selected = select_method()
    result = run_pearson("x", "y")
    ci = pearson_ci(result["statistic"], result["group_stats"]["n"])
    method_ok = selected["selected_method"] == "pearson"
    r_ok = rel_close(result["statistic"], 0.8164)
    p_ok = rel_close(result["p_value"], 0.00217)
    ci_ok = abs(ci[0] - 0.424) < 0.02 and abs(ci[1] - 0.951) < 0.02
    add_check(
        rows,
        "2",
        "Anscombe quartet I",
        "pearson",
        selected["selected_method"],
        "r=0.8164, p=0.00217, CI≈[0.424,0.951]",
        f"r={result['statistic']:.6g}, p={result['p_value']:.6g}, CI=[{ci[0]:.3f},{ci[1]:.3f}]",
        method_ok and r_ok and p_ok and ci_ok,
        {"pearson": result, "ci_95": ci},
    )


def validate_titanic(rows):
    plan = CategoricalPlan(
        research_question="Titanic class survival chi-square",
        intent="categorical_test",
        row_variable="class",
        col_variable="survived",
    )
    set_current_plan(plan)
    load_data("data/textbook_chi2_titanic.csv")
    expected_freq = check_expected_frequencies("class", "survived")
    selected = select_method()
    result = run_chi_square("class", "survived", yates=True)
    chi_ok = rel_close(result["statistic"], 130.94)
    p_ok = rel_close(result["p_value"], 2.55e-30)
    df_ok = expected_freq.get("dof") == 1
    add_check(
        rows,
        "3",
        "Titanic 1st vs 3rd",
        "chi_square_yates",
        selected["selected_method"],
        "χ²=130.94, p≈2.55e-30, df=1",
        f"χ²={result['statistic']:.6g}, p={result['p_value']:.6g}, df={expected_freq.get('dof')}",
        selected["selected_method"] == "chi_square_yates" and chi_ok and p_ok and df_ok,
        {"expected_frequencies": expected_freq, "chi_square_yates": result},
    )


def validate_tea(rows):
    plan = CategoricalPlan(
        research_question="Lady tasting tea Fisher exact",
        intent="categorical_test",
        row_variable="actual",
        col_variable="guess",
    )
    set_current_plan(plan)
    load_data("data/textbook_fisher_tea.csv")
    check_expected_frequencies("actual", "guess")
    selected = select_method()
    result = run_fisher_exact("actual", "guess")
    or_ok = rel_close(result["statistic"], 9.0)
    p_ok = rel_close(result["p_value"], 0.4857)
    add_check(
        rows,
        "4",
        "Fisher tea",
        "fisher_exact",
        selected["selected_method"],
        "OR=9.0, p=0.4857, CI≈[0.21,626.2]",
        f"OR={result['statistic']:.6g}, p={result['p_value']:.6g}",
        selected["selected_method"] == "fisher_exact" and or_ok and p_ok,
        {"fisher_exact": result},
    )


def write_report(rows):
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    lines = [
        "# 教科书数据集统计正确性验证",
        "",
        f"**日期**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Git commit**: {commit_hash()}",
        "**验证依据**: R 内置数据集与标准统计教科书",
        "",
        "## 验证结果汇总",
        "",
        "| Case | 数据集 | 期望方法 | 实际方法 | 期望值 | 实际值 | 一致? |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        mark = "✓" if row["passed"] else "✗"
        lines.append(
            f"| {row['case']} | {row['dataset']} | {row['expected_method']} | {row['actual_method']} | "
            f"{row['expected']} | {row['actual']} | {mark} |"
        )
    lines.extend(["", "## 详细输出", ""])
    for row in rows:
        lines.extend(
            [
                f"### Case {row['case']} - {row['dataset']}",
                "",
                f"- 通过: {'是' if row['passed'] else '否'}",
                f"- 期望方法: `{row['expected_method']}`",
                f"- 实际方法: `{row['actual_method']}`",
                f"- 期望值: {row['expected']}",
                f"- 实际值: {row['actual']}",
                "",
                "```text",
                repr(row["details"]),
                "```",
                "",
            ]
        )
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    rows = []
    validate_plantgrowth(rows)
    validate_anscombe(rows)
    validate_titanic(rows)
    validate_tea(rows)
    write_report(rows)
    print("\nTextbook validation comparison:")
    for row in rows:
        print(f"{row['case']} {row['dataset']}: expected {row['expected']} | actual {row['actual']} | {'PASS' if row['passed'] else 'FAIL'}")
    print(f"\nReport written to {REPORT_PATH}")
    failed = [row for row in rows if not row["passed"]]
    if failed:
        raise AssertionError(f"{len(failed)} textbook validation check(s) failed. See {REPORT_PATH}")


if __name__ == "__main__":
    main()
