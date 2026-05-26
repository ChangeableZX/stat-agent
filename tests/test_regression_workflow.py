import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import state
from decision import select_method as direct_select_method
from state import RegressionPlan
from tools import (
    check_homoscedasticity,
    check_independence,
    check_multicollinearity,
    check_outliers,
    check_residual_normality,
    load_data,
    make_analysis_plan,
    plot_qq_residuals,
    plot_regression_fit,
    plot_residuals,
    run_multiple_linear_regression,
    run_ols_robust_se,
    run_ols_with_log_y,
    run_simple_linear_regression,
    select_final_model,
    select_method,
)


def test_case15_diagnostics_selects_log_y_and_reruns_model():
    make_analysis_plan("case15", "regression", y_variable="y", x_variables=["x"])
    load_data("data/case15_regression_log_y.csv")
    assert select_method()["selected_method"] == "simple_linear_regression"
    assert run_simple_linear_regression("y", "x")["method"] == "simple_linear_regression"

    normality = check_residual_normality("y", ["x"])
    homoscedasticity = check_homoscedasticity("y", ["x"])
    check_independence("y", ["x"])
    check_outliers("y", ["x"])
    final = select_final_model()

    assert normality["passed"] is False
    assert homoscedasticity["passed"] is False
    assert final["final_model"] == "ols_log_y"
    assert "残差非正态+异方差" in " ".join(final["method_rationale"])
    assert final["transformations"] == [{"type": "log_y", "reason": "残差非正态+异方差"}]
    assert run_ols_with_log_y("y", ["x"])["method"] == "ols_log_y"


def test_case14_diagnostics_selects_robust_se_and_reruns_model():
    make_analysis_plan("case14", "regression", y_variable="y", x_variables=["x"])
    load_data("data/case14_regression_robust.csv")
    normality = check_residual_normality("y", ["x"])
    homoscedasticity = check_homoscedasticity("y", ["x"])
    check_independence("y", ["x"])
    check_outliers("y", ["x"])
    final = select_final_model()

    assert normality["passed"] is True
    assert normality["p_value"] > 0.1
    assert homoscedasticity["passed"] is False
    assert final["final_model"] == "ols_robust_se"
    assert "残差正态 + 异方差" in " ".join(final["method_rationale"])
    assert "稳健标准误" in " ".join(final["method_rationale"])
    assert run_ols_robust_se("y", ["x"])["method"] == "ols_robust_se"


def test_regression_tools_reject_non_regression_plan():
    make_analysis_plan(
        "two group",
        "compare_two_groups",
        target_variable="value",
        grouping_variable="group",
        design="independent",
    )
    load_data("data/demo.csv")

    calls = [
        lambda: run_simple_linear_regression("value", "value"),
        lambda: run_multiple_linear_regression("value", ["value"]),
        lambda: run_ols_with_log_y("value", ["value"]),
        lambda: run_ols_robust_se("value", ["value"]),
        lambda: select_final_model(),
        lambda: check_residual_normality("value", ["value"]),
        lambda: check_homoscedasticity("value", ["value"]),
        lambda: check_independence("value", ["value"]),
        lambda: check_multicollinearity(["value"]),
        lambda: check_outliers("value", ["value"]),
        lambda: plot_residuals("value", ["value"]),
        lambda: plot_qq_residuals("value", ["value"]),
        lambda: plot_regression_fit("value", "value"),
    ]
    for call in calls:
        result = call()
        assert "error" in result
        assert "RegressionPlan" in result["error"]


def test_regression_summary_injects_diagnostics_for_followup():
    make_analysis_plan("case14", "regression", y_variable="y", x_variables=["x"])
    load_data("data/case14_regression_robust.csv")
    check_homoscedasticity("y", ["x"])
    summary = state.get_current_plan().summary()

    assert "回归诊断" in summary
    assert "homoscedasticity" in summary
    assert "breusch_pagan" in summary
    assert "1.4052212848372664e-05" in summary


def test_regression_router_and_x_variables_string_coercion():
    plan = RegressionPlan(research_question="test", intent="regression", y_variable="y", x_variables=["x"])
    selected = direct_select_method(plan)
    assert selected.selected_method == "simple_linear_regression"

    result = make_analysis_plan("coerce", "regression", y_variable="y", x_variables="x")
    assert result["plan_type"] == "RegressionPlan"
    assert result["x_variables"] == ["x"]
    assert "warning" in result


def test_case16_detects_multicollinearity_and_flags_drop_collinear():
    make_analysis_plan("case16", "regression", y_variable="y", x_variables=["x1", "x2", "x3"])
    load_data("data/case16_regression_collinear.csv")
    run_multiple_linear_regression("y", ["x1", "x2", "x3"])
    check_residual_normality("y", ["x1", "x2", "x3"])
    check_homoscedasticity("y", ["x1", "x2", "x3"])
    check_independence("y", ["x1", "x2", "x3"])
    check_outliers("y", ["x1", "x2", "x3"])
    vif = check_multicollinearity(["x1", "x2", "x3"])
    final = select_final_model()

    assert vif["max_vif"] > 10
    assert vif["passed"] is False
    assert final["final_model"] == "ols_drop_collinear"
    assert "多重共线性" in " ".join(final["method_rationale"])


def main():
    test_case15_diagnostics_selects_log_y_and_reruns_model()
    test_case14_diagnostics_selects_robust_se_and_reruns_model()
    test_regression_tools_reject_non_regression_plan()
    test_regression_summary_injects_diagnostics_for_followup()
    test_regression_router_and_x_variables_string_coercion()
    test_case16_detects_multicollinearity_and_flags_drop_collinear()
    print("PASS")


if __name__ == "__main__":
    main()
