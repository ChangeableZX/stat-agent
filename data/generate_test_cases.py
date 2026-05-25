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
    check_normality,
    check_variance_equality,
    load_data,
    make_analysis_plan,
    run_independent_ttest,
    run_mannwhitney,
    run_paired_ttest,
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


def run_all_cases():
    generate_cases()
    cases = [
        ("Case 0 regression", "demo.csv", "student_t", "双正态+方差齐", "independent"),
        ("Case 1 unequal variance", "case1_unequal_variance.csv", "welch_t", "方差不齐", "independent"),
        ("Case 2 non-normal small", "case2_non_normal_small.csv", "mannwhitney", "非正态+小样本", "independent"),
        ("Case 3 non-normal large", "case3_non_normal_large.csv", "welch_t", "CLT 兜底", "independent"),
        ("Case 4 paired normal", "case4_paired.csv", "paired_t", "配对设计", "paired"),
    ]
    for case in cases:
        _run_case(*case)


def _write_two_group(filename, a_values, b_values):
    df = pd.DataFrame({
        "group": ["A"] * len(a_values) + ["B"] * len(b_values),
        "value": np.concatenate([a_values, b_values]),
    })
    df.to_csv(os.path.join(DATA_DIR, filename), index=False)


def _run_case(name, filename, expected_method, expected_keyword, design):
    print(f"\n{'=' * 80}\n{name}")
    path = os.path.join("data", filename)
    if design == "paired":
        make_analysis_plan(
            research_question=name,
            intent="compare_two_groups",
            target_variable="after-before",
            grouping_variable=None,
            paired_columns=["before", "after"],
            design="paired",
        )
        load_data(path)
        normality = check_normality("diff")
        selection = select_method()
        result = run_paired_ttest("before", "after") if selection["selected_method"] == "paired_t" else None
    else:
        make_analysis_plan(
            research_question=name,
            intent="compare_two_groups",
            target_variable="value",
            grouping_variable="group",
            paired_columns=None,
            design="independent",
        )
        loaded = load_data(path)
        groups = list(loaded["value_counts"]["group"].keys())
        normality = [check_normality("value", "group", group) for group in groups]
        variance = check_variance_equality("value", "group")
        selection = select_method()
        result = _run_selected(selection["selected_method"])
        print(f"前提检验: normality={normality}, variance={variance}")

    plan = state.get_current_plan()
    print(f"📋 StatPlan 最终状态:\n{plan.summary()}")
    print(f"选中方法: {plan.selected_method} (expected {expected_method})")
    print(f"决策路径: {plan.method_rationale}")
    if design == "paired":
        print(f"前提检验: normality_diff={normality}")
    print(f"统计结果: {result}")

    if plan.selected_method != expected_method or expected_keyword not in " ".join(plan.method_rationale):
        raise AssertionError(f"{name} expected {expected_method}/{expected_keyword}, got {plan.selected_method}/{plan.method_rationale}")
    if name.startswith("Case 0"):
        _assert_day1_regression(result)


def _run_selected(method):
    if method == "student_t":
        return run_independent_ttest("value", "group", equal_var=True)
    if method == "welch_t":
        return run_welch_ttest("value", "group")
    if method == "mannwhitney":
        return run_mannwhitney("value", "group")
    raise ValueError(f"Unsupported method in test runner: {method}")


def _assert_day1_regression(result):
    checks = [
        ("t_statistic", result["t_statistic"], DAY1_T),
        ("p_value", result["p_value"], DAY1_P),
        ("cohens_d", result["cohens_d"], DAY1_D),
    ]
    for label, actual, expected in checks:
        if not np.isclose(actual, expected, rtol=0, atol=1e-12):
            raise AssertionError(f"Case 0 regression changed {label}: {actual} != {expected}")
    print("Case 0 regression PASS: Day 1 t, p, Cohen's d unchanged.")


if __name__ == "__main__":
    run_all_cases()
