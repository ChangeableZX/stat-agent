import os
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import state
from tools import (
    check_bivariate_normality,
    check_expected_frequencies,
    check_linearity,
    check_normality,
    check_variance_equality,
    load_data,
    make_analysis_plan,
    run_chi_square,
    run_fisher_exact,
    run_independent_ttest,
    run_kendall,
    run_kruskal_wallis,
    run_mannwhitney,
    run_one_way_anova,
    run_paired_ttest,
    run_pearson,
    run_spearman,
    run_welch_anova,
    run_welch_ttest,
    select_method,
)


DATA_DIR = os.path.dirname(__file__)
DAY1_T = -4.292991096003232
DAY1_P = 4.147346687277438e-05
DAY1_D = -0.8585982192006464


def generate_cases():
    rng = np.random.default_rng(42)
    _write_two_group("case1_unequal_variance.csv", rng.normal(100, 5, 50), rng.normal(108, 20, 50))
    _write_two_group("case2_non_normal_small.csv", rng.exponential(1.0, 15), rng.exponential(1.2, 15) + 0.8)
    _write_two_group("case3_non_normal_large.csv", rng.exponential(1.0, 50), rng.exponential(1.2, 50) + 0.8)
    before = rng.normal(100, 12, 40)
    after = before + rng.normal(5, 8, 40)
    pd.DataFrame({"before": before, "after": after}).to_csv(os.path.join(DATA_DIR, "case4_paired.csv"), index=False)

    _write_multi("case5_multi_anova.csv", [rng.normal(100 + i * 4, 10, 50) for i in range(4)])
    _write_multi("case6_multi_welch.csv", [rng.normal(100 + i * 4, sd, 50) for i, sd in enumerate([5, 20, 8, 25])])
    _write_multi("case7_multi_kruskal.csv", [rng.exponential(1.0 + i * 0.25, 15) + i * 0.5 for i in range(4)])

    x = rng.normal(0, 1, 80)
    y = 0.75 * x + rng.normal(0, 0.35, 80)
    pd.DataFrame({"x": x, "y": y}).to_csv(os.path.join(DATA_DIR, "case8_corr_pearson.csv"), index=False)
    x = rng.exponential(1.0, 80)
    y = np.log1p(x) + rng.normal(0, 0.08, 80)
    pd.DataFrame({"x": x, "y": y}).to_csv(os.path.join(DATA_DIR, "case9_corr_spearman.csv"), index=False)

    _write_categorical("case10_cat_chi_yates.csv", [[30, 20], [18, 32]])
    _write_categorical("case11_cat_fisher.csv", [[1, 9], [8, 2]])
    _write_categorical("case12_cat_chi_square.csv", [[20, 15, 18], [14, 22, 16], [17, 13, 24]])


def run_all_cases():
    generate_cases()
    cases = [
        ("Case 0 regression", "demo.csv", "student_t", "双正态+方差齐", "two"),
        ("Case 1 unequal variance", "case1_unequal_variance.csv", "welch_t", "方差不齐", "two"),
        ("Case 2 non-normal small", "case2_non_normal_small.csv", "mannwhitney", "非正态+小样本", "two"),
        ("Case 3 non-normal large", "case3_non_normal_large.csv", "welch_t", "CLT 兜底", "two"),
        ("Case 4 paired normal", "case4_paired.csv", "paired_t", "配对设计", "paired"),
        ("Case 5 multi anova", "case5_multi_anova.csv", "one_way_anova", "各组正态+方差齐", "multi"),
        ("Case 6 multi welch", "case6_multi_welch.csv", "welch_anova", "方差不齐", "multi"),
        ("Case 7 multi kruskal", "case7_multi_kruskal.csv", "kruskal_wallis", "至少一组非正态", "multi"),
        ("Case 8 corr pearson", "case8_corr_pearson.csv", "pearson", "双正态+线性", "corr"),
        ("Case 9 corr spearman", "case9_corr_spearman.csv", "spearman", "非正态或非线性", "corr"),
        ("Case 10 cat chi yates", "case10_cat_chi_yates.csv", "chi_square_yates", "Yates", "cat"),
        ("Case 11 cat fisher", "case11_cat_fisher.csv", "fisher_exact", "Fisher", "cat"),
        ("Case 12 cat chi square", "case12_cat_chi_square.csv", "chi_square", "R×C", "cat"),
    ]
    for case in cases:
        _run_case(*case)


def _write_two_group(filename, a_values, b_values):
    df = pd.DataFrame({"group": ["A"] * len(a_values) + ["B"] * len(b_values), "value": np.concatenate([a_values, b_values])})
    df.to_csv(os.path.join(DATA_DIR, filename), index=False)


def _write_multi(filename, arrays):
    rows = []
    for idx, values in enumerate(arrays):
        rows.extend({"group": f"G{idx + 1}", "value": v} for v in values)
    pd.DataFrame(rows).to_csv(os.path.join(DATA_DIR, filename), index=False)


def _write_categorical(filename, table):
    rows = []
    for i, row in enumerate(table):
        for j, count in enumerate(row):
            rows.extend({"row": f"R{i + 1}", "col": f"C{j + 1}"} for _ in range(count))
    pd.DataFrame(rows).to_csv(os.path.join(DATA_DIR, filename), index=False)


def _run_case(name, filename, expected_method, expected_keyword, kind):
    print(f"\n{'=' * 80}\n{name}")
    path = os.path.join("data", filename)
    if kind == "paired":
        _must_plan(make_analysis_plan(name, "compare_two_groups", target_variable="after-before", paired_columns=["before", "after"], design="paired"))
        load_data(path)
        normality = check_normality("diff")
        selection = select_method()
        result = run_paired_ttest("before", "after")
        print(f"前提检验: normality_diff={normality}")
    elif kind == "two":
        _must_plan(make_analysis_plan(name, "compare_two_groups", target_variable="value", grouping_variable="group", design="independent"))
        loaded = load_data(path)
        groups = list(loaded["value_counts"]["group"].keys())
        normality = [check_normality("value", "group", group) for group in groups]
        variance = check_variance_equality("value", "group")
        selection = select_method()
        result = _run_two_selected(selection["selected_method"])
        print(f"前提检验: normality={normality}, variance={variance}")
    elif kind == "multi":
        _must_plan(make_analysis_plan(name, "compare_multi_groups", target_variable="value", grouping_variable="group", design="independent"))
        loaded = load_data(path)
        groups = list(loaded["value_counts"]["group"].keys())
        normality = [check_normality("value", "group", group) for group in groups]
        variance = check_variance_equality("value", "group")
        selection = select_method()
        result = _run_multi_selected(selection["selected_method"])
        print(f"前提检验: normality={normality}, variance={variance}")
    elif kind == "corr":
        _must_plan(make_analysis_plan(name, "correlation", x_variable="x", y_variable="y"))
        load_data(path)
        normality = check_bivariate_normality("x", "y")
        linearity = check_linearity("x", "y")
        selection = select_method()
        result = _run_corr_selected(selection["selected_method"])
        print(f"前提检验: bivariate_normality={normality}, linearity={linearity}")
    elif kind == "cat":
        _must_plan(make_analysis_plan(name, "categorical_test", row_variable="row", col_variable="col", paired=False))
        load_data(path)
        expected = check_expected_frequencies("row", "col")
        selection = select_method()
        result = _run_cat_selected(selection["selected_method"])
        print(f"前提检验: expected_frequencies={expected}")
    else:
        raise ValueError(kind)

    plan = state.get_current_plan()
    print(f"📋 StatPlan 最终状态:\n{plan.summary()}")
    print(f"选中方法: {plan.selected_method} (expected {expected_method})")
    print(f"决策路径: {plan.method_rationale}")
    print(f"统计结果: {result}")
    if plan.selected_method != expected_method or expected_keyword not in " ".join(plan.method_rationale):
        raise AssertionError(f"{name} expected {expected_method}/{expected_keyword}, got {plan.selected_method}/{plan.method_rationale}")
    if name.startswith("Case 0"):
        _assert_day1_regression(result)


def _run_two_selected(method):
    if method == "student_t":
        return run_independent_ttest("value", "group", equal_var=True)
    if method == "welch_t":
        return run_welch_ttest("value", "group")
    if method == "mannwhitney":
        return run_mannwhitney("value", "group")
    raise ValueError(f"Unsupported method: {method}")


def _must_plan(result):
    if isinstance(result, dict) and "error" in result:
        raise AssertionError(f"make_analysis_plan failed: {result['error']}")
    return result


def _run_multi_selected(method):
    if method == "one_way_anova":
        return run_one_way_anova("value", "group")
    if method == "welch_anova":
        return run_welch_anova("value", "group")
    if method == "kruskal_wallis":
        return run_kruskal_wallis("value", "group")
    raise ValueError(f"Unsupported method: {method}")


def _run_corr_selected(method):
    if method == "pearson":
        return run_pearson("x", "y")
    if method == "spearman":
        return run_spearman("x", "y")
    if method == "kendall":
        return run_kendall("x", "y")
    raise ValueError(f"Unsupported method: {method}")


def _run_cat_selected(method):
    if method == "chi_square_yates":
        return run_chi_square("row", "col", yates=True)
    if method == "chi_square":
        return run_chi_square("row", "col")
    if method == "fisher_exact":
        return run_fisher_exact("row", "col")
    raise ValueError(f"Unsupported method: {method}")


def _assert_day1_regression(result):
    for label, actual, expected in [
        ("t_statistic", result["t_statistic"], DAY1_T),
        ("p_value", result["p_value"], DAY1_P),
        ("cohens_d", result["cohens_d"], DAY1_D),
    ]:
        if not np.isclose(actual, expected, rtol=0, atol=1e-12):
            raise AssertionError(f"Case 0 regression changed {label}: {actual} != {expected}")
    print("Case 0 regression PASS: Day 1 t, p, Cohen's d unchanged.")


if __name__ == "__main__":
    run_all_cases()
