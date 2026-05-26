import math
import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor
from statsmodels.stats.power import FTestAnovaPower, TTestIndPower
from statsmodels.stats.stattools import durbin_watson

import state
from decision import select_final_model as route_select_final_model
from decision import select_method as route_select_method
from state import CategoricalPlan, CorrelationPlan, MultiGroupPlan, RegressionPlan, TwoGroupPlan


_DATA_CACHE = None
_MISSING_MARKERS = {"", "na", "n/a", "nan", "null", "none"}


def make_analysis_plan(research_question: str, intent: str, **kwargs):
    """Create the correct StatPlan subclass after validating intent-specific fields."""
    try:
        valid_intents = ["compare_two_groups", "compare_multi_groups", "correlation", "categorical_test", "regression"]
        if intent not in valid_intents:
            return {"error": f"intent 必须是 {valid_intents} 之一,得到 '{intent}'"}
        missing = []
        if intent == "compare_two_groups":
            missing += _missing(kwargs, ["target_variable"])
            design = kwargs.get("design", "independent")
            if design == "paired":
                missing += _missing(kwargs, ["paired_columns"])
            else:
                missing += _missing(kwargs, ["grouping_variable"])
        elif intent == "compare_multi_groups":
            missing += _missing(kwargs, ["target_variable", "grouping_variable"])
        elif intent == "correlation":
            missing += _missing(kwargs, ["x_variable", "y_variable"])
        elif intent == "categorical_test":
            missing += _missing(kwargs, ["row_variable", "col_variable"])
        elif intent == "regression":
            missing += _missing(kwargs, ["y_variable", "x_variables"])
        if missing:
            return {"error": f"make_analysis_plan 缺少字段: {missing}. 请补全后重试。"}

        if intent == "compare_two_groups":
            paired_columns = kwargs.get("paired_columns")
            if paired_columns:
                paired_columns = tuple(paired_columns)
            plan = TwoGroupPlan(
                research_question=research_question,
                intent=intent,
                target_variable=kwargs["target_variable"],
                grouping_variable=kwargs.get("grouping_variable"),
                paired_columns=paired_columns,
                design=kwargs.get("design", "independent"),
            )
        elif intent == "compare_multi_groups":
            plan = MultiGroupPlan(
                research_question=research_question,
                intent=intent,
                target_variable=kwargs["target_variable"],
                grouping_variable=kwargs["grouping_variable"],
                design=kwargs.get("design", "independent"),
            )
        elif intent == "correlation":
            plan = CorrelationPlan(
                research_question=research_question,
                intent=intent,
                x_variable=kwargs["x_variable"],
                y_variable=kwargs["y_variable"],
                x_type=kwargs.get("x_type", "continuous"),
                y_type=kwargs.get("y_type", "continuous"),
            )
        elif intent == "categorical_test":
            plan = CategoricalPlan(
                research_question=research_question,
                intent=intent,
                row_variable=kwargs["row_variable"],
                col_variable=kwargs["col_variable"],
                paired=bool(kwargs.get("paired", False)),
            )
        else:
            x_variables = kwargs["x_variables"]
            warning = None
            if isinstance(x_variables, str):
                x_variables = [x_variables]
                warning = "x_variables received as string and was converted to a one-item list."
            if not isinstance(x_variables, list) or not all(isinstance(item, str) and item for item in x_variables):
                return {"error": "x_variables 必须是非空字符串列表。"}
            plan = RegressionPlan(
                research_question=research_question,
                intent=intent,
                y_variable=kwargs["y_variable"],
                x_variables=x_variables,
                regression_type=kwargs.get("regression_type", "linear"),
            )
        state.set_current_plan(plan)
        result = plan.to_dict()
        if intent == "regression" and warning:
            result["warning"] = warning
        return result
    except Exception as exc:
        return {"error": str(exc)}


def load_data(file_path: str):
    """Load a CSV file, cache its DataFrame globally, and return a JSON-serializable summary."""
    global _DATA_CACHE
    try:
        raw_df = pd.read_csv(file_path, keep_default_na=False)
        df = pd.read_csv(file_path)
        _DATA_CACHE = df
        preview = df.head().replace({np.nan: None}).to_dict(orient="records")
        value_counts = {}
        for col in df.columns:
            unique_count = int(df[col].dropna().nunique())
            if unique_count <= 20 and (df[col].dtype == "object" or unique_count <= 10):
                counts = df[col].dropna().astype(str).value_counts().to_dict()
                value_counts[col] = {str(k): int(v) for k, v in counts.items()}
        data_quality = {}
        for col in raw_df.columns:
            raw = raw_df[col].astype(str).str.strip()
            lowered = raw.str.lower()
            missing_like = int(lowered.isin(_MISSING_MARKERS).sum())
            parsed = pd.to_numeric(raw, errors="coerce")
            has_numeric_values = int(parsed.notna().sum()) > 0
            non_numeric = int((~lowered.isin(_MISSING_MARKERS) & parsed.isna()).sum()) if has_numeric_values else 0
            if missing_like or non_numeric:
                data_quality[col] = {"missing_like_values": missing_like, "non_numeric_values": non_numeric}
        result = {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "preview": preview,
            "value_counts": value_counts,
            "data_quality": data_quality,
        }
        plan = state.get_current_plan()
        if plan:
            plan.data_quality = data_quality
            if isinstance(plan, (TwoGroupPlan, MultiGroupPlan)) and plan.grouping_variable in value_counts:
                plan.sample_sizes = value_counts[plan.grouping_variable]
                if isinstance(plan, MultiGroupPlan):
                    plan.group_levels = list(value_counts[plan.grouping_variable].keys())
            if isinstance(plan, CategoricalPlan) and plan.row_variable in df.columns and plan.col_variable in df.columns:
                table = pd.crosstab(df[plan.row_variable], df[plan.col_variable])
                plan.table_shape = tuple(table.shape)
                plan.sample_sizes = {"rows": int(table.shape[0]), "cols": int(table.shape[1]), "n": int(table.values.sum())}

            # 提前验证计划变量是否存在于数据中
            missing_vars = []
            if isinstance(plan, TwoGroupPlan):
                for col in [plan.target_variable, plan.grouping_variable]:
                    if col and col not in df.columns:
                        missing_vars.append(col)
            elif isinstance(plan, MultiGroupPlan):
                for col in [plan.target_variable, plan.grouping_variable]:
                    if col and col not in df.columns:
                        missing_vars.append(col)
            elif isinstance(plan, CorrelationPlan):
                for col in [plan.x_variable, plan.y_variable]:
                    if col and col not in df.columns:
                        missing_vars.append(col)
            elif isinstance(plan, RegressionPlan):
                for col in [plan.y_variable] + list(plan.x_variables or []):
                    if col and col not in df.columns:
                        missing_vars.append(col)
            elif isinstance(plan, CategoricalPlan):
                for col in [plan.row_variable, plan.col_variable]:
                    if col and col not in df.columns:
                        missing_vars.append(col)
            if missing_vars:
                result["plan_variable_warning"] = (
                    f"[警告] 计划中的以下变量在数据中不存在: {missing_vars}。"
                    f"请停止分析并告知用户列名有误。数据实际列名为: {list(df.columns)}"
                )
        return result
    except Exception as exc:
        return {"error": str(exc)}


def check_normality(column: str, group_column: str = None, group_value: str = None):
    """Run a Shapiro-Wilk normality test on a numeric column, optionally filtered to one group."""
    try:
        df = _require_data()
        if column not in df.columns:
            values = _paired_diff_values(column)
            if values is None:
                return {"error": f"Column not found: {column}"}
            group_rows = int(len(values))
            check_key = "normality_diff"
        else:
            if group_column is not None:
                if group_column not in df.columns:
                    return {"error": f"Group column not found: {group_column}"}
                df = df[df[group_column] == group_value]
            group_rows = int(len(df))
            values = pd.to_numeric(df[column], errors="coerce").dropna()
            check_key = f"normality_{group_value}" if group_value is not None else f"normality_{column}"
        n = int(len(values))
        if n < 3:
            return {
                "error": "Shapiro-Wilk test requires at least 3 valid numeric observations.",
                "group_rows": group_rows,
                "valid_n": n,
                "dropped_missing_or_invalid": group_rows - n,
            }
        stat, p_value = stats.shapiro(values)
        result = {
            "column": column,
            "group_column": group_column,
            "group_value": group_value,
            "n": n,
            "group_rows": group_rows,
            "dropped_missing_or_invalid": group_rows - n,
            "statistic": float(stat),
            "p_value": float(p_value),
            "reject_normality_alpha_0_05": bool(p_value < 0.05),
        }
        plan = state.get_current_plan()
        if plan:
            plan.assumption_checks[check_key] = {"passed": bool(p_value >= 0.05), "p": float(p_value), "n": n}
        return result
    except Exception as exc:
        return {"error": str(exc)}


def check_variance_equality(value_col: str, group_col: str, group_values: list = None):
    """Run Levene's test for equal variance across two or more groups. Use group_values to restrict to specific groups."""
    try:
        groups, arrays = _group_arrays(value_col, group_col, group_filter=group_values)
        stat, p_value = stats.levene(*arrays)
        result = {
            "statistic": float(stat),
            "p_value": float(p_value),
            "equal_variance": bool(p_value >= 0.05),
            "variances": {str(g): float(a.var(ddof=1)) for g, a in zip(groups, arrays)},
        }
        plan = state.get_current_plan()
        if plan:
            plan.assumption_checks["variance_equality"] = result
        return result
    except Exception as exc:
        return {"error": str(exc)}


def check_sphericity(subject_col: str, within_col: str, value_col: str):
    """Run Mauchly's sphericity test for repeated-measures data."""
    try:
        plan = _guard(MultiGroupPlan, "check_sphericity")
        if isinstance(plan, dict):
            return plan
        import pingouin as pg

        df = _require_data().dropna(subset=[subject_col, within_col, value_col])
        wide = df.pivot(index=subject_col, columns=within_col, values=value_col)
        sph, w_value, chi2, dof, p_value = pg.sphericity(wide)
        result = {"passed": bool(sph), "w": float(w_value), "chi2": float(chi2), "dof": int(dof), "p_value": float(p_value)}
        plan.assumption_checks["sphericity"] = result
        return result
    except Exception as exc:
        return {"error": str(exc)}


def check_bivariate_normality(x_col: str, y_col: str):
    """Simplified bivariate normality check: Shapiro-Wilk on both variables."""
    try:
        plan = _guard(CorrelationPlan, "check_bivariate_normality")
        if isinstance(plan, dict):
            return plan
        df = _numeric_pair(x_col, y_col)
        sx, px = stats.shapiro(df[x_col])
        sy, py = stats.shapiro(df[y_col])
        result = {
            "passed": bool(px >= 0.05 and py >= 0.05),
            "x": {"statistic": float(sx), "p_value": float(px)},
            "y": {"statistic": float(sy), "p_value": float(py)},
            "n": int(len(df)),
        }
        plan.assumption_checks["bivariate_normality"] = result
        return result
    except Exception as exc:
        return {"error": str(exc)}


def check_linearity(x_col: str, y_col: str):
    """Simplified linearity check using Pearson vs Spearman similarity."""
    try:
        plan = _guard(CorrelationPlan, "check_linearity")
        if isinstance(plan, dict):
            return plan
        df = _numeric_pair(x_col, y_col)
        pearson_r, _ = stats.pearsonr(df[x_col], df[y_col])
        spearman_r, _ = stats.spearmanr(df[x_col], df[y_col])
        result = {
            "passed": bool(abs(pearson_r - spearman_r) < 0.15),
            "pearson_r": float(pearson_r),
            "spearman_r": float(spearman_r),
        }
        plan.assumption_checks["linearity"] = result
        return result
    except Exception as exc:
        return {"error": str(exc)}


def check_expected_frequencies(row_col: str, col_col: str):
    """Compute chi-square expected frequencies for a contingency table."""
    try:
        plan = _guard(CategoricalPlan, "check_expected_frequencies")
        if isinstance(plan, dict):
            return plan
        table = pd.crosstab(_require_data()[row_col], _require_data()[col_col])
        chi2, p_value, dof, expected = stats.chi2_contingency(table, correction=False)
        low = expected < 5
        result = {
            "table_shape": tuple(table.shape),
            "expected": expected.tolist(),
            "all_expected_ge_5": bool(np.all(~low)),
            "low_frequency_ratio": float(low.sum() / expected.size),
            "chi2_preview": float(chi2),
            "p_value_preview": float(p_value),
            "dof": int(dof),
        }
        plan.table_shape = tuple(table.shape)
        plan.assumption_checks["expected_frequencies"] = result
        return _jsonable(result)
    except Exception as exc:
        return {"error": str(exc)}


def build_contingency_table(row_col: str, col_col: str):
    """Build and return a contingency table."""
    try:
        plan = _guard(CategoricalPlan, "build_contingency_table")
        if isinstance(plan, dict):
            return plan
        table = pd.crosstab(_require_data()[row_col], _require_data()[col_col])
        plan.table_shape = tuple(table.shape)
        return {"table": table.to_dict(), "table_shape": list(table.shape)}
    except Exception as exc:
        return {"error": str(exc)}


def select_method():
    """Select a statistical method from the current StatPlan using deterministic Python rules."""
    try:
        plan = state.get_current_plan()
        if plan is None:
            return {"error": "No StatPlan exists. Call make_analysis_plan first."}
        plan = route_select_method(plan)
        state.set_current_plan(plan)
        return {"selected_method": plan.selected_method, "method_rationale": plan.method_rationale, "plan_summary": plan.summary()}
    except Exception as exc:
        return {"error": str(exc)}


def run_simple_linear_regression(y_col: str, x_col: str):
    """Run simple OLS regression with one predictor."""
    try:
        plan = _guard(RegressionPlan, "run_simple_linear_regression")
        if isinstance(plan, dict):
            return plan
        result = _run_regression("simple_linear_regression", y_col, [x_col])
        _store_result(result)
        return result
    except Exception as exc:
        return {"error": str(exc)}


def run_multiple_linear_regression(y_col: str, x_cols: list):
    """Run multiple OLS regression with two or more predictors."""
    try:
        plan = _guard(RegressionPlan, "run_multiple_linear_regression")
        if isinstance(plan, dict):
            return plan
        result = _run_regression("multiple_linear_regression", y_col, _as_list(x_cols))
        _store_result(result)
        return result
    except Exception as exc:
        return {"error": str(exc)}


def run_ols_with_log_y(y_col: str, x_cols: list):
    """Fit OLS after log-transforming Y."""
    try:
        plan = _guard(RegressionPlan, "run_ols_with_log_y")
        if isinstance(plan, dict):
            return plan
        result = _run_regression("ols_log_y", y_col, _as_list(x_cols), log_y=True)
        _store_result(result)
        return result
    except Exception as exc:
        return {"error": str(exc)}


def run_ols_robust_se(y_col: str, x_cols: list):
    """Fit OLS and report HC3 robust standard errors and p-values."""
    try:
        plan = _guard(RegressionPlan, "run_ols_robust_se")
        if isinstance(plan, dict):
            return plan
        result = _run_regression("ols_robust_se", y_col, _as_list(x_cols), robust_cov="HC3")
        _store_result(result)
        return result
    except Exception as exc:
        return {"error": str(exc)}


def select_final_model():
    """Select the final regression model after diagnostics have been written to RegressionPlan."""
    try:
        plan = _guard(RegressionPlan, "select_final_model")
        if isinstance(plan, dict):
            return plan
        plan = route_select_final_model(plan)
        state.set_current_plan(plan)
        return {
            "final_model": plan.final_model,
            "transformations": plan.transformations,
            "method_rationale": plan.method_rationale,
            "plan_summary": plan.summary(),
        }
    except Exception as exc:
        return {"error": str(exc)}


def check_residual_normality(y_col: str, x_cols: list):
    """Run Shapiro-Wilk normality test on OLS residuals."""
    try:
        plan = _guard(RegressionPlan, "check_residual_normality")
        if isinstance(plan, dict):
            return plan
        model = _fit_ols(y_col, _as_list(x_cols))
        stat, p_value = stats.shapiro(model.resid)
        result = {
            "test": "shapiro_residuals",
            "statistic": float(stat),
            "p_value": float(p_value),
            "passed": bool(p_value >= 0.05),
            "interpretation_hints": _generic_hints(float(p_value), "残差偏离正态"),
        }
        plan.diagnostics["residual_normality"] = {"passed": result["passed"], "p": result["p_value"]}
        return result
    except Exception as exc:
        return {"error": str(exc)}


def check_homoscedasticity(y_col: str, x_cols: list):
    """Run Breusch-Pagan test for homoscedasticity."""
    try:
        plan = _guard(RegressionPlan, "check_homoscedasticity")
        if isinstance(plan, dict):
            return plan
        model = _fit_ols(y_col, _as_list(x_cols))
        lm_stat, lm_p, f_stat, f_p = het_breuschpagan(model.resid, model.model.exog)
        result = {
            "test": "breusch_pagan",
            "statistic": float(lm_stat),
            "p_value": float(lm_p),
            "f_statistic": float(f_stat),
            "f_p_value": float(f_p),
            "passed": bool(lm_p >= 0.05),
            "interpretation_hints": _generic_hints(float(lm_p), "存在异方差"),
        }
        plan.diagnostics["homoscedasticity"] = {"passed": result["passed"], "p": result["p_value"], "test": "breusch_pagan"}
        return result
    except Exception as exc:
        return {"error": str(exc)}


def check_independence(y_col: str, x_cols: list):
    """Compute Durbin-Watson statistic for residual independence."""
    try:
        plan = _guard(RegressionPlan, "check_independence")
        if isinstance(plan, dict):
            return plan
        model = _fit_ols(y_col, _as_list(x_cols))
        dw = float(durbin_watson(model.resid))
        result = {
            "test": "durbin_watson",
            "statistic": dw,
            "dw_statistic": dw,
            "passed": bool(1.5 <= dw <= 2.5),
            "interpretation_hints": {"practical_caveat": "Durbin-Watson 接近 2 表示残差独立性较好。"},
        }
        plan.diagnostics["independence"] = {"passed": result["passed"], "dw_statistic": dw}
        return result
    except Exception as exc:
        return {"error": str(exc)}


def check_multicollinearity(x_cols: list):
    """Compute VIF values for multiple regression predictors."""
    try:
        plan = _guard(RegressionPlan, "check_multicollinearity")
        if isinstance(plan, dict):
            return plan
        x_cols = _as_list(x_cols)
        if len(x_cols) <= 1:
            result = {"test": "vif", "vif": {}, "max_vif": 0.0, "passed": True, "interpretation_hints": {"practical_caveat": "单自变量无需 VIF 诊断。"}}
            plan.diagnostics["multicollinearity"] = {"passed": True, "max_vif": 0.0}
            return result
        x = _regression_frame(None, x_cols, require_y=False)[x_cols]
        exog = sm.add_constant(x, has_constant="add")
        vif = {col: float(variance_inflation_factor(exog.values, idx + 1)) for idx, col in enumerate(x_cols)}
        max_vif = max(vif.values()) if vif else 0.0
        result = {
            "test": "vif",
            "vif": vif,
            "max_vif": float(max_vif),
            "passed": bool(max_vif <= 10),
            "interpretation_hints": {"practical_caveat": "VIF > 10 通常提示严重多重共线性。"},
        }
        plan.diagnostics["multicollinearity"] = {"passed": result["passed"], "max_vif": float(max_vif)}
        return result
    except Exception as exc:
        return {"error": str(exc)}


def check_outliers(y_col: str, x_cols: list):
    """Count influential observations using Cook's distance > 4/n."""
    try:
        plan = _guard(RegressionPlan, "check_outliers")
        if isinstance(plan, dict):
            return plan
        model = _fit_ols(y_col, _as_list(x_cols))
        cooks_d = OLSInfluence(model).cooks_distance[0]
        threshold = 4 / len(cooks_d)
        n_influential = int(np.sum(cooks_d > threshold))
        allowed = max(1, int(0.05 * len(cooks_d)))
        result = {
            "test": "cooks_distance",
            "statistic": float(np.max(cooks_d)),
            "threshold": float(threshold),
            "n_influential": n_influential,
            "max_cooks_d": float(np.max(cooks_d)),
            "passed": bool(n_influential <= allowed),
            "interpretation_hints": {"practical_caveat": "Cook's D > 4/n 的点需要人工复核。"},
        }
        plan.diagnostics["outliers"] = {"passed": result["passed"], "n_influential": n_influential, "max_cooks_d": result["max_cooks_d"]}
        return result
    except Exception as exc:
        return {"error": str(exc)}


def plot_residuals(y_col: str, x_cols: list):
    """Save residuals vs fitted values plot for regression."""
    try:
        plan = _guard(RegressionPlan, "plot_residuals")
        if isinstance(plan, dict):
            return plan
        import matplotlib.pyplot as plt

        model = _fit_ols(y_col, _as_list(x_cols))
        path = f"output/residuals_{_method_label()}_{datetime.now().strftime('%H%M%S')}.png"
        os.makedirs("output", exist_ok=True)
        plt.figure(figsize=(6, 4))
        plt.scatter(model.fittedvalues, model.resid, alpha=0.75)
        plt.axhline(0, color="black", linewidth=1)
        plt.xlabel("Fitted values")
        plt.ylabel("Residuals")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        _store_plot(path)
        return {"plot_path": path, "saved_to": path}
    except Exception as exc:
        return {"error": str(exc)}


def plot_qq_residuals(y_col: str, x_cols: list):
    """Save residual Q-Q plot for regression."""
    try:
        plan = _guard(RegressionPlan, "plot_qq_residuals")
        if isinstance(plan, dict):
            return plan
        import matplotlib.pyplot as plt

        model = _fit_ols(y_col, _as_list(x_cols))
        path = f"output/qq_residuals_{_method_label()}_{datetime.now().strftime('%H%M%S')}.png"
        os.makedirs("output", exist_ok=True)
        plt.figure(figsize=(5, 5))
        stats.probplot(model.resid, dist="norm", plot=plt)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        _store_plot(path)
        return {"plot_path": path, "saved_to": path}
    except Exception as exc:
        return {"error": str(exc)}


def plot_regression_fit(y_col: str, x_col: str):
    """Save simple linear regression scatter plot with fitted line."""
    try:
        plan = _guard(RegressionPlan, "plot_regression_fit")
        if isinstance(plan, dict):
            return plan
        import matplotlib.pyplot as plt

        df = _regression_frame(y_col, [x_col])
        model = _fit_ols(y_col, [x_col])
        xs = np.linspace(df[x_col].min(), df[x_col].max(), 100)
        pred = model.params["const"] + model.params[x_col] * xs
        path = f"output/regression_fit_{_method_label()}_{datetime.now().strftime('%H%M%S')}.png"
        os.makedirs("output", exist_ok=True)
        plt.figure(figsize=(6, 4))
        plt.scatter(df[x_col], df[y_col], alpha=0.75)
        plt.plot(xs, pred, color="black", linewidth=1.5)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        _store_plot(path)
        return {"plot_path": path, "saved_to": path}
    except Exception as exc:
        return {"error": str(exc)}


def power_analysis_ttest(effect_size=None, alpha=0.05, power=0.8, n=None):
    """Solve two-sample t-test power, sample size per group, or effect size."""
    try:
        computed = _missing_power_target(effect_size, alpha, power, n)
        analysis = TTestIndPower()
        value = analysis.solve_power(effect_size=effect_size, nobs1=n, alpha=alpha, power=power, ratio=1.0, alternative="two-sided")
        return _power_result(computed, value, {"effect_size": effect_size, "alpha": alpha, "power": power, "n": n}, "two-sample t-test")
    except Exception as exc:
        return {"error": str(exc)}


def power_analysis_anova(effect_size=None, n_groups=None, alpha=0.05, power=0.8, n=None):
    """Solve one-way ANOVA power. n is interpreted as per-group sample size."""
    try:
        if n_groups is None:
            return {"error": "n_groups is required for ANOVA power analysis."}
        computed = _missing_power_target(effect_size, alpha, power, n)
        analysis = FTestAnovaPower()
        nobs = None if n is None else n * n_groups
        value = analysis.solve_power(effect_size=effect_size, nobs=nobs, alpha=alpha, power=power, k_groups=n_groups)
        if computed == "n":
            value = value / n_groups
        return _power_result(computed, value, {"effect_size": effect_size, "n_groups": n_groups, "alpha": alpha, "power": power, "n": n}, "one-way ANOVA")
    except Exception as exc:
        return {"error": str(exc)}


def power_analysis_correlation(r=None, alpha=0.05, power=0.8, n=None):
    """Approximate Pearson correlation power using Fisher z transformation."""
    try:
        computed = _missing_power_target(r, alpha, power, n, effect_name="r")
        if computed == "power":
            value = _correlation_power(r, alpha, n)
        elif computed == "n":
            value = _solve_correlation_n(r, alpha, power)
        elif computed == "r":
            value = _solve_correlation_r(alpha, power, n)
        else:
            return {"error": "Solving alpha for correlation power is not supported."}
        return _power_result(computed, value, {"r": r, "alpha": alpha, "power": power, "n": n}, "Pearson correlation")
    except Exception as exc:
        return {"error": str(exc)}


def run_independent_ttest(value_column: str, group_column: str, equal_var: bool = True, group_values: list = None):
    """Run an independent-samples t-test for exactly two groups. Use group_values to filter when column has more than 2 groups."""
    try:
        plan = state.get_current_plan()
        if plan is not None and not isinstance(plan, TwoGroupPlan):
            return {"error": f"run_independent_ttest 要求 TwoGroupPlan,当前是 {type(plan).__name__}"}
        groups, x1, x2, raw1, raw2 = _two_group_values(value_column, group_column, include_raw=True, group_filter=group_values)
        result = _independent_ttest_result(groups, x1, x2, raw1, raw2, equal_var)
        _store_result(result)
        return result
    except Exception as exc:
        return {"error": str(exc)}


def run_welch_ttest(value_col: str, group_col: str, group_values: list = None):
    """Run Welch's independent-samples t-test. Use group_values to filter when column has more than 2 groups."""
    return run_independent_ttest(value_col, group_col, equal_var=False, group_values=group_values)


def run_mannwhitney(value_col: str, group_col: str, group_values: list = None):
    """Run Mann-Whitney U test for two independent groups. Use group_values to filter when column has more than 2 groups."""
    try:
        plan = _guard(TwoGroupPlan, "run_mannwhitney")
        if isinstance(plan, dict):
            return plan
        groups, x1, x2, raw1, raw2 = _two_group_values(value_col, group_col, include_raw=True, group_filter=group_values)
        u_stat, p_value = stats.mannwhitneyu(x1, x2, alternative="two-sided")
        n1, n2 = len(x1), len(x2)
        mean_u = n1 * n2 / 2
        sd_u = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        z_value = (u_stat - mean_u) / sd_u if sd_u else 0.0
        r_value = abs(z_value) / math.sqrt(n1 + n2)
        result = _base_result("mannwhitney", float(u_stat), float(p_value), groups, x1, x2)
        result.update({
            "effect_size": {"name": "r", "value": float(r_value), "magnitude": _r_magnitude(r_value)},
            "ci_95": [None, None],
            "dropped_missing_or_invalid": {str(groups[0]): int(len(raw1) - n1), str(groups[1]): int(len(raw2) - n2)},
        })
        _store_result(result)
        return result
    except Exception as exc:
        return {"error": str(exc)}


def run_paired_ttest(col1: str, col2: str):
    """Run a paired-samples t-test for two numeric columns."""
    try:
        plan = _guard(TwoGroupPlan, "run_paired_ttest")
        if isinstance(plan, dict):
            return plan
        x, y = _paired_values(col1, col2)
        diff = x - y
        stat, p_value = stats.ttest_rel(x, y)
        n = len(diff)
        mean_diff = float(diff.mean())
        se = float(diff.std(ddof=1) / math.sqrt(n))
        t_crit = stats.t.ppf(0.975, n - 1)
        dz = mean_diff / float(diff.std(ddof=1)) if float(diff.std(ddof=1)) else float("nan")
        result = {
            "method": "paired_t",
            "statistic": float(stat),
            "p_value": float(p_value),
            "effect_size": {"name": "cohens_dz", "value": float(dz), "magnitude": _cohens_magnitude(dz)},
            "ci_95": [float(mean_diff - t_crit * se), float(mean_diff + t_crit * se)],
            "group_stats": {col1: _one_group_stats(x), col2: _one_group_stats(y)},
            "interpretation_hints": _hints(float(p_value), float(y.mean() - x.mean()), n, n),
        }
        _store_result(result)
        return result
    except Exception as exc:
        return {"error": str(exc)}


def run_wilcoxon(col1: str, col2: str):
    """Run Wilcoxon signed-rank test for paired columns."""
    try:
        plan = _guard(TwoGroupPlan, "run_wilcoxon")
        if isinstance(plan, dict):
            return plan
        x, y = _paired_values(col1, col2)
        stat, p_value = stats.wilcoxon(x, y)
        z_approx = stats.norm.isf(float(p_value) / 2)
        r_value = abs(z_approx) / math.sqrt(len(x))
        result = {
            "method": "wilcoxon",
            "statistic": float(stat),
            "p_value": float(p_value),
            "effect_size": {"name": "r", "value": float(r_value), "magnitude": _r_magnitude(r_value)},
            "ci_95": [None, None],
            "group_stats": {col1: _one_group_stats(x), col2: _one_group_stats(y)},
            "interpretation_hints": _hints(float(p_value), float(y.median() - x.median()), len(x), len(y)),
        }
        _store_result(result)
        return result
    except Exception as exc:
        return {"error": str(exc)}


def run_one_way_anova(value_col: str, group_col: str):
    """Run one-way ANOVA for independent multi-group data."""
    try:
        plan = _guard(MultiGroupPlan, "run_one_way_anova")
        if isinstance(plan, dict):
            return plan
        groups, arrays = _group_arrays(value_col, group_col)
        stat, p_value = stats.f_oneway(*arrays)
        eta = _eta_squared_anova(arrays)
        result = _multi_result("one_way_anova", stat, p_value, groups, arrays, "eta_squared", eta)
        _store_result(result)
        return result
    except Exception as exc:
        return {"error": str(exc)}


def run_welch_anova(value_col: str, group_col: str):
    """Run Welch ANOVA for independent multi-group data."""
    try:
        plan = _guard(MultiGroupPlan, "run_welch_anova")
        if isinstance(plan, dict):
            return plan
        import pingouin as pg

        df = _require_data().dropna(subset=[value_col, group_col]).copy()
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=[value_col])
        out = pg.welch_anova(data=df, dv=value_col, between=group_col).iloc[0]
        groups, arrays = _group_arrays(value_col, group_col)
        p_col = "p-unc" if "p-unc" in out else "p_unc"
        result = _multi_result("welch_anova", out["F"], out[p_col], groups, arrays, "np2", out.get("np2", np.nan))
        _store_result(result)
        return result
    except Exception as exc:
        return {"error": str(exc)}


def run_kruskal_wallis(value_col: str, group_col: str):
    """Run Kruskal-Wallis H test."""
    try:
        plan = _guard(MultiGroupPlan, "run_kruskal_wallis")
        if isinstance(plan, dict):
            return plan
        groups, arrays = _group_arrays(value_col, group_col)
        stat, p_value = stats.kruskal(*arrays)
        epsilon_sq = max((stat - len(groups) + 1) / (sum(len(a) for a in arrays) - len(groups)), 0)
        result = _multi_result("kruskal_wallis", stat, p_value, groups, arrays, "epsilon_squared", epsilon_sq)
        _store_result(result)
        return result
    except Exception as exc:
        return {"error": str(exc)}


def run_repeated_anova(subject_col: str, within_col: str, value_col: str):
    """Run repeated-measures ANOVA using pingouin."""
    try:
        plan = _guard(MultiGroupPlan, "run_repeated_anova")
        if isinstance(plan, dict):
            return plan
        import pingouin as pg

        df = _require_data().dropna(subset=[subject_col, within_col, value_col]).copy()
        out = pg.rm_anova(data=df, dv=value_col, within=within_col, subject=subject_col, detailed=True).iloc[0]
        result = {
            "method": "repeated_anova",
            "statistic": float(out["F"]),
            "p_value": float(out["p-unc"]),
            "effect_size": {"name": "ng2", "value": float(out.get("ng2", np.nan)), "magnitude": _eta_magnitude(out.get("ng2", 0))},
            "group_stats": _group_stats(df, value_col, within_col),
            "interpretation_hints": _generic_hints(float(out["p-unc"]), "某些条件之间有差异,需事后检验"),
        }
        _store_result(result)
        return result
    except Exception as exc:
        return {"error": str(exc)}


def run_friedman(subject_col: str, within_col: str, value_col: str):
    """Run Friedman test for repeated-measures data."""
    try:
        plan = _guard(MultiGroupPlan, "run_friedman")
        if isinstance(plan, dict):
            return plan
        df = _require_data().dropna(subset=[subject_col, within_col, value_col])
        wide = df.pivot(index=subject_col, columns=within_col, values=value_col).dropna()
        stat, p_value = stats.friedmanchisquare(*[wide[col] for col in wide.columns])
        result = {
            "method": "friedman",
            "statistic": float(stat),
            "p_value": float(p_value),
            "effect_size": {"name": "kendall_w", "value": float(stat / (len(wide) * (wide.shape[1] - 1))), "magnitude": "medium"},
            "group_stats": _group_stats(df, value_col, within_col),
            "interpretation_hints": _generic_hints(float(p_value), "某些条件之间有差异,需事后检验"),
        }
        _store_result(result)
        return result
    except Exception as exc:
        return {"error": str(exc)}


def run_posthoc_tukey(value_col: str, group_col: str):
    """Run Tukey HSD posthoc test."""
    try:
        plan = _guard(MultiGroupPlan, "run_posthoc_tukey")
        if isinstance(plan, dict):
            return plan
        import pingouin as pg

        out = pg.pairwise_tukey(data=_require_data(), dv=value_col, between=group_col)
        return {"method": "tukey_hsd", "comparisons": _jsonable(out.to_dict(orient="records"))}
    except Exception as exc:
        return {"error": str(exc)}


def run_posthoc_games_howell(value_col: str, group_col: str):
    """Run Games-Howell posthoc test."""
    try:
        plan = _guard(MultiGroupPlan, "run_posthoc_games_howell")
        if isinstance(plan, dict):
            return plan
        import pingouin as pg

        out = pg.pairwise_gameshowell(data=_require_data(), dv=value_col, between=group_col)
        return {"method": "games_howell", "comparisons": _jsonable(out.to_dict(orient="records"))}
    except Exception as exc:
        return {"error": str(exc)}


def run_posthoc_dunn(value_col: str, group_col: str, p_adjust: str = "bh"):
    """Run Dunn posthoc test."""
    try:
        plan = _guard(MultiGroupPlan, "run_posthoc_dunn")
        if isinstance(plan, dict):
            return plan
        import scikit_posthocs as sp

        adjust = "fdr_bh" if p_adjust == "bh" else p_adjust
        out = sp.posthoc_dunn(_require_data(), val_col=value_col, group_col=group_col, p_adjust=adjust)
        return {"method": "dunn_bh", "p_values": _jsonable(out.to_dict())}
    except Exception as exc:
        return {"error": str(exc)}


def run_pearson(x_col: str, y_col: str):
    """Run Pearson correlation."""
    return _correlation_result("pearson", x_col, y_col, stats.pearsonr)


def run_spearman(x_col: str, y_col: str):
    """Run Spearman correlation."""
    return _correlation_result("spearman", x_col, y_col, stats.spearmanr)


def run_kendall(x_col: str, y_col: str):
    """Run Kendall tau correlation."""
    return _correlation_result("kendall", x_col, y_col, stats.kendalltau)


def run_chi_square(row_col: str, col_col: str, yates: bool = False):
    """Run chi-square test of independence."""
    try:
        plan = _guard(CategoricalPlan, "run_chi_square")
        if isinstance(plan, dict):
            return plan
        table = pd.crosstab(_require_data()[row_col], _require_data()[col_col])
        chi2, p_value, dof, expected = stats.chi2_contingency(table, correction=yates)
        method = "chi_square_yates" if yates else "chi_square"
        result = {
            "method": method,
            "statistic": float(chi2),
            "p_value": float(p_value),
            "effect_size": {"name": "cramers_v", "value": _cramers_v(chi2, table), "magnitude": "medium"},
            "group_stats": {"observed": table.to_dict(), "expected": expected.tolist()},
            "interpretation_hints": _generic_hints(float(p_value), "分类变量之间有关联"),
        }
        _store_result(result)
        return _jsonable(result)
    except Exception as exc:
        return {"error": str(exc)}


def run_fisher_exact(row_col: str, col_col: str):
    """Run Fisher exact test for a 2x2 table."""
    try:
        plan = _guard(CategoricalPlan, "run_fisher_exact")
        if isinstance(plan, dict):
            return plan
        table = pd.crosstab(_require_data()[row_col], _require_data()[col_col])
        if table.shape != (2, 2):
            return {"error": f"run_fisher_exact 要求 2x2 表,当前是 {table.shape}"}
        odds, p_value = stats.fisher_exact(table.values)
        result = {
            "method": "fisher_exact",
            "statistic": float(odds),
            "p_value": float(p_value),
            "effect_size": {"name": "odds_ratio", "value": float(odds), "magnitude": "not_classified"},
            "group_stats": {"observed": table.to_dict()},
            "interpretation_hints": _generic_hints(float(p_value), "分类变量之间有关联"),
        }
        _store_result(result)
        return _jsonable(result)
    except Exception as exc:
        return {"error": str(exc)}


def run_fisher_freeman_halton(row_col: str, col_col: str):
    """Approximate Fisher-Freeman-Halton fallback using chi-square for R x C tables."""
    try:
        plan = _guard(CategoricalPlan, "run_fisher_freeman_halton")
        if isinstance(plan, dict):
            return plan
        result = run_chi_square(row_col, col_col, yates=False)
        if "error" not in result:
            result["method"] = "fisher_freeman_halton"
            result["interpretation_hints"]["practical_caveat"] = "Python 环境中使用 chi-square 近似替代 Fisher-Freeman-Halton 精确检验"
            _store_result(result)
        return result
    except Exception as exc:
        return {"error": str(exc)}


def run_mcnemar(row_col: str, col_col: str):
    """Run McNemar test for paired 2x2 categorical data."""
    try:
        plan = _guard(CategoricalPlan, "run_mcnemar")
        if isinstance(plan, dict):
            return plan
        from statsmodels.stats.contingency_tables import mcnemar

        table = pd.crosstab(_require_data()[row_col], _require_data()[col_col])
        if table.shape != (2, 2):
            return {"error": f"run_mcnemar 要求 2x2 表,当前是 {table.shape}"}
        out = mcnemar(table.values, exact=False, correction=True)
        result = _categorical_simple("mcnemar", out.statistic, out.pvalue, table)
        _store_result(result)
        return _jsonable(result)
    except Exception as exc:
        return {"error": str(exc)}


def run_mcnemar_bowker(row_col: str, col_col: str):
    """Run Bowker symmetry test for paired multi-category data."""
    try:
        plan = _guard(CategoricalPlan, "run_mcnemar_bowker")
        if isinstance(plan, dict):
            return plan
        table = pd.crosstab(_require_data()[row_col], _require_data()[col_col])
        values = table.values
        stat = 0.0
        dof = 0
        for i in range(values.shape[0]):
            for j in range(i + 1, values.shape[1]):
                denom = values[i, j] + values[j, i]
                if denom:
                    stat += (values[i, j] - values[j, i]) ** 2 / denom
                    dof += 1
        p_value = stats.chi2.sf(stat, dof) if dof else 1.0
        result = _categorical_simple("mcnemar_bowker", stat, p_value, table)
        _store_result(result)
        return _jsonable(result)
    except Exception as exc:
        return {"error": str(exc)}


def plot_boxplot(value_col: str, group_col: str, save_path: str = None):
    """Save a boxplot under output/ and record the path in the current plan."""
    try:
        import matplotlib.pyplot as plt

        df = _require_data()
        groups = [g for g in df[group_col].dropna().unique().tolist()]
        data = [pd.to_numeric(df[df[group_col] == g][value_col], errors="coerce").dropna() for g in groups]
        os.makedirs("output", exist_ok=True)
        path = _output_path(save_path) if save_path else _default_boxplot_path()
        plt.figure(figsize=(6, 4))
        plt.boxplot(data, labels=[str(g) for g in groups])
        plt.ylabel(value_col)
        plt.xlabel(group_col)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        _store_plot(path)
        return {"plot_path": path, "saved_to": path}
    except Exception as exc:
        return {"error": str(exc)}


def plot_qq(column: str, group_col: str = None, group_value: str = None, save_path: str = "qq.png"):
    """Save a Q-Q plot under output/ and record the path in the current plan."""
    try:
        import matplotlib.pyplot as plt

        df = _require_data()
        if column not in df.columns:
            values = _paired_diff_values(column)
            title = "paired difference"
        else:
            if group_col is not None:
                df = df[df[group_col] == group_value]
            values = pd.to_numeric(df[column], errors="coerce").dropna()
            title = f"{column}" if group_value is None else f"{column} ({group_value})"
        os.makedirs("output", exist_ok=True)
        path = _output_path(save_path) if save_path and save_path != "qq.png" else _default_qq_path(group_value)
        plt.figure(figsize=(5, 5))
        stats.probplot(values, dist="norm", plot=plt)
        plt.title(f"Q-Q plot: {title}")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        _store_plot(path)
        return {"plot_path": path, "saved_to": path}
    except Exception as exc:
        return {"error": str(exc)}


def plot_grouped_boxplot(value_col: str, group_col: str):
    """Save a grouped boxplot for multi-group analysis."""
    return plot_boxplot(value_col, group_col)


def plot_scatter(x_col: str, y_col: str, fit_line: bool = True):
    """Save a scatter plot for correlation analysis."""
    try:
        plan = _guard(CorrelationPlan, "plot_scatter")
        if isinstance(plan, dict):
            return plan
        import matplotlib.pyplot as plt

        df = _numeric_pair(x_col, y_col)
        path = f"output/scatter_{_method_label()}_{datetime.now().strftime('%H%M%S')}.png"
        os.makedirs("output", exist_ok=True)
        plt.figure(figsize=(6, 4))
        plt.scatter(df[x_col], df[y_col], alpha=0.75)
        if fit_line:
            coeff = np.polyfit(df[x_col], df[y_col], 1)
            xs = np.linspace(df[x_col].min(), df[x_col].max(), 100)
            plt.plot(xs, coeff[0] * xs + coeff[1])
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        _store_plot(path)
        return {"plot_path": path, "saved_to": path}
    except Exception as exc:
        return {"error": str(exc)}


def plot_mosaic(row_col: str, col_col: str):
    """Save a simple stacked bar chart for a contingency table."""
    try:
        plan = _guard(CategoricalPlan, "plot_mosaic")
        if isinstance(plan, dict):
            return plan
        import matplotlib.pyplot as plt

        table = pd.crosstab(_require_data()[row_col], _require_data()[col_col])
        path = f"output/mosaic_{_method_label()}_{datetime.now().strftime('%H%M%S')}.png"
        os.makedirs("output", exist_ok=True)
        table.plot(kind="bar", stacked=True, figsize=(6, 4))
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        _store_plot(path)
        return {"plot_path": path, "saved_to": path}
    except Exception as exc:
        return {"error": str(exc)}


def _as_list(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, tuple):
        return list(value)
    return value


def _regression_frame(y_col, x_cols, require_y=True, log_y=False):
    df = _require_data()
    x_cols = _as_list(x_cols)
    if not x_cols:
        raise ValueError("Regression requires at least one x column.")
    missing = [col for col in x_cols if col not in df.columns]
    if require_y and y_col not in df.columns:
        missing.append(y_col)
    if missing:
        raise ValueError(f"Column(s) not found: {missing}")
    cols = ([y_col] if require_y else []) + x_cols
    data = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if log_y:
        data = data[data[y_col] > 0].copy()
        data[y_col] = np.log(data[y_col])
    min_rows = len(x_cols) + 2 if require_y else len(x_cols) + 1
    if len(data) < min_rows:
        raise ValueError("Regression requires more complete numeric rows than predictors.")
    return data


def _fit_ols(y_col, x_cols, log_y=False, robust_cov=None):
    data = _regression_frame(y_col, x_cols, log_y=log_y)
    x = sm.add_constant(data[x_cols], has_constant="add")
    model = sm.OLS(data[y_col], x).fit()
    if robust_cov:
        model = model.get_robustcov_results(cov_type=robust_cov)
        model.model.data.xnames = ["const"] + list(x_cols)
    return model


def _run_regression(method, y_col, x_cols, log_y=False, robust_cov=None):
    model = _fit_ols(y_col, x_cols, log_y=log_y, robust_cov=robust_cov)
    return _regression_result(method, model, x_cols, log_y=log_y, robust_cov=robust_cov)


def _regression_result(method, model, x_cols, log_y=False, robust_cov=None):
    names = list(getattr(model.model.data, "xnames", ["const"] + list(x_cols)))
    label_map = {"const": "intercept"}
    params = dict(zip(names, model.params))
    pvalues = dict(zip(names, model.pvalues))
    conf = np.asarray(model.conf_int())
    coefficients = {label_map.get(name, name): float(value) for name, value in params.items()}
    p_values = {label_map.get(name, name): float(value) for name, value in pvalues.items()}
    ci_95 = {label_map.get(name, name): [float(conf[idx][0]), float(conf[idx][1])] for idx, name in enumerate(names)}
    return {
        "method": method,
        "coefficients": coefficients,
        "p_values": p_values,
        "ci_95": ci_95,
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "f_statistic": float(model.fvalue) if model.fvalue is not None else None,
        "f_p_value": float(model.f_pvalue) if model.f_pvalue is not None else None,
        "n_obs": int(model.nobs),
        "interpretation_hints": {
            "significant_model": bool(model.f_pvalue < 0.05) if model.f_pvalue is not None else None,
            "log_y": bool(log_y),
            "robust_cov": robust_cov,
        },
    }


def _missing_power_target(effect_size, alpha, power, n, effect_name="effect_size"):
    values = {effect_name: effect_size, "alpha": alpha, "power": power, "n": n}
    missing = [name for name, value in values.items() if value is None]
    if len(missing) != 1:
        raise ValueError("Power analysis requires exactly one of effect_size/r, alpha, power, or n to be None.")
    return missing[0]


def _power_result(computed, value, inputs, label):
    return {
        "computed": computed,
        "value": float(value),
        "inputs": inputs,
        "interpretation": f"{label}: computed {computed} = {float(value):.6g}.",
    }


def _correlation_power(r, alpha, n):
    if n <= 3:
        raise ValueError("Correlation power requires n > 3.")
    z_effect = abs(np.arctanh(r)) * math.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    return float(stats.norm.sf(z_crit - z_effect) + stats.norm.cdf(-z_crit - z_effect))


def _solve_correlation_n(r, alpha, target_power):
    low, high = 4, 8
    while _correlation_power(r, alpha, high) < target_power:
        high *= 2
        if high > 100000:
            raise ValueError("Could not solve correlation sample size.")
    for _ in range(60):
        mid = (low + high) / 2
        if _correlation_power(r, alpha, mid) < target_power:
            low = mid
        else:
            high = mid
    return high


def _solve_correlation_r(alpha, target_power, n):
    low, high = 1e-6, 0.999999
    for _ in range(60):
        mid = (low + high) / 2
        if _correlation_power(mid, alpha, n) < target_power:
            low = mid
        else:
            high = mid
    return high


def _missing(kwargs, names):
    return [name for name in names if kwargs.get(name) in (None, "", [])]


def _guard(expected_type, tool_name):
    plan = state.get_current_plan()
    if not isinstance(plan, expected_type):
        return {"error": f"{tool_name} 要求 {expected_type.__name__},当前是 {type(plan).__name__}"}
    return plan


def _require_data():
    if _DATA_CACHE is None:
        raise ValueError("No data loaded. Call load_data first.")
    return _DATA_CACHE


def _group_arrays(value_col, group_col, group_filter=None):
    df = _require_data()
    if value_col not in df.columns:
        raise ValueError(f"Value column not found: {value_col}")
    if group_col not in df.columns:
        raise ValueError(f"Group column not found: {group_col}")
    if group_filter:
        df = df[df[group_col].isin(group_filter)]
        if df.empty:
            raise ValueError(f"group_values={group_filter} 过滤后数据为空,请检查组名是否与数据匹配。")
    groups = [g for g in df[group_col].dropna().unique().tolist()]
    arrays = [pd.to_numeric(df[df[group_col] == g][value_col], errors="coerce").dropna() for g in groups]
    if any(len(a) < 2 for a in arrays):
        raise ValueError("Each group needs at least 2 valid numeric observations.")
    return groups, arrays


def _two_group_values(value_col, group_col, include_raw=False, group_filter=None):
    df = _require_data()
    if value_col not in df.columns:
        raise ValueError(f"Value column not found: {value_col}")
    if group_col not in df.columns:
        raise ValueError(f"Group column not found: {group_col}")
    if group_filter:
        df = df[df[group_col].isin(group_filter)]
        if df.empty:
            raise ValueError(f"group_values={group_filter} 过滤后数据为空,请检查组名是否与数据匹配。")
    groups = [g for g in df[group_col].dropna().unique().tolist()]
    group_counts = df[group_col].dropna().astype(str).value_counts().to_dict()
    if len(groups) != 2:
        raise ValueError(f"Expected exactly 2 groups in {group_col}, found {len(groups)}: {group_counts}")
    raw1 = df[df[group_col] == groups[0]][value_col]
    raw2 = df[df[group_col] == groups[1]][value_col]
    x1 = pd.to_numeric(raw1, errors="coerce").dropna()
    x2 = pd.to_numeric(raw2, errors="coerce").dropna()
    if len(x1) < 2 or len(x2) < 2:
        raise ValueError("Each group needs at least 2 valid numeric observations.")
    if include_raw:
        return groups, x1, x2, raw1, raw2
    return groups, x1, x2


def _paired_values(col1, col2):
    df = _require_data()
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError(f"Paired columns not found: {col1}, {col2}")
    pair = df[[col1, col2]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(pair) < 3:
        raise ValueError("Paired tests require at least 3 valid pairs.")
    return pair[col1], pair[col2]


def _paired_diff_values(column):
    plan = state.get_current_plan()
    if column not in {"diff", "difference", "paired_diff"} or not isinstance(plan, TwoGroupPlan) or not plan.paired_columns:
        return None
    x, y = _paired_values(plan.paired_columns[0], plan.paired_columns[1])
    return x - y


def _independent_ttest_result(groups, x1, x2, raw1, raw2, equal_var):
    n1, n2 = int(len(x1)), int(len(x2))
    t_stat, p_value = stats.ttest_ind(x1, x2, equal_var=equal_var)
    mean1, mean2 = float(x1.mean()), float(x2.mean())
    var1, var2 = float(x1.var(ddof=1)), float(x2.var(ddof=1))
    if equal_var:
        dfree = n1 + n2 - 2
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / dfree
        se = math.sqrt(pooled_var * (1 / n1 + 1 / n2))
        method = "student_t"
    else:
        se = math.sqrt(var1 / n1 + var2 / n2)
        numerator = (var1 / n1 + var2 / n2) ** 2
        denominator = (var1 ** 2) / ((n1 ** 2) * (n1 - 1)) + (var2 ** 2) / ((n2 ** 2) * (n2 - 1))
        dfree = numerator / denominator
        method = "welch_t"
    mean_diff = mean1 - mean2
    t_crit = stats.t.ppf(0.975, dfree)
    ci_low = mean_diff - t_crit * se
    ci_high = mean_diff + t_crit * se
    pooled_sd = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    cohens_d = mean_diff / pooled_sd if pooled_sd else float("nan")
    warnings = []
    if n1 < 5 or n2 < 5:
        warnings.append("Very small sample size; interpret normality and t-test results cautiously.")
    result = _base_result(method, float(t_stat), float(p_value), groups, x1, x2)
    result.update({
        "test": "Student independent t-test" if equal_var else "Welch independent t-test",
        "groups": [str(groups[0]), str(groups[1])],
        "n": {str(groups[0]): n1, str(groups[1]): n2},
        "dropped_missing_or_invalid": {str(groups[0]): int(len(raw1) - n1), str(groups[1]): int(len(raw2) - n2)},
        "means": {str(groups[0]): mean1, str(groups[1]): mean2},
        "mean_difference_first_minus_second": float(mean_diff),
        "t_statistic": float(t_stat),
        "degrees_of_freedom": float(dfree),
        "confidence_interval_95": [float(ci_low), float(ci_high)],
        "cohens_d": float(cohens_d),
        "effect_size": {"name": "cohens_d", "value": float(cohens_d), "magnitude": _cohens_magnitude(cohens_d)},
        "ci_95": [float(ci_low), float(ci_high)],
        "warnings": warnings,
    })
    return result


def _base_result(method, statistic, p_value, groups, x1, x2):
    return {
        "method": method,
        "statistic": statistic,
        "p_value": p_value,
        "effect_size": None,
        "ci_95": None,
        "group_stats": {str(groups[0]): _one_group_stats(x1), str(groups[1]): _one_group_stats(x2)},
        "interpretation_hints": _hints(p_value, float(x2.mean() - x1.mean()), len(x1), len(x2), groups),
    }


def _multi_result(method, statistic, p_value, groups, arrays, effect_name, effect_value):
    return {
        "method": method,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "effect_size": {"name": effect_name, "value": float(effect_value), "magnitude": _eta_magnitude(effect_value)},
        "group_stats": {str(g): _one_group_stats(a) for g, a in zip(groups, arrays)},
        "interpretation_hints": _generic_hints(float(p_value), "某些组之间有差异,需事后检验"),
    }


def _correlation_result(method, x_col, y_col, func):
    try:
        plan = _guard(CorrelationPlan, f"run_{method}")
        if isinstance(plan, dict):
            return plan
        df = _numeric_pair(x_col, y_col)
        stat, p_value = func(df[x_col], df[y_col])
        result = {
            "method": method,
            "statistic": float(stat),
            "p_value": float(p_value),
            "effect_size": {"name": f"{method}_r", "value": float(stat), "magnitude": _r_magnitude(stat)},
            "group_stats": {"n": int(len(df)), "x": _one_group_stats(df[x_col]), "y": _one_group_stats(df[y_col])},
            "interpretation_hints": _generic_hints(float(p_value), "变量之间存在相关关系"),
        }
        _store_result(result)
        return result
    except Exception as exc:
        return {"error": str(exc)}


def _categorical_simple(method, statistic, p_value, table):
    return {
        "method": method,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "effect_size": {"name": "not_available", "value": None, "magnitude": "not_classified"},
        "group_stats": {"observed": table.to_dict()},
        "interpretation_hints": _generic_hints(float(p_value), "分类变量之间有关联"),
    }


def _numeric_pair(x_col, y_col):
    df = _require_data()
    pair = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(pair) < 3:
        raise ValueError("Correlation requires at least 3 complete numeric pairs.")
    return pair


def _one_group_stats(values):
    return {"n": int(len(values)), "mean": float(values.mean()), "std": float(values.std(ddof=1)), "median": float(values.median())}


def _group_stats(df, value_col, group_col):
    return {str(g): _one_group_stats(pd.to_numeric(sub[value_col], errors="coerce").dropna()) for g, sub in df.groupby(group_col)}


def _hints(p_value, diff_second_minus_first, n1, n2, groups=None):
    if p_value >= 0.05:
        direction = "no_diff"
    elif groups is None:
        direction = "after > before" if diff_second_minus_first > 0 else "before > after"
    else:
        direction = f"{groups[1]} > {groups[0]}" if diff_second_minus_first > 0 else f"{groups[0]} > {groups[1]}"
    return {"significant": bool(p_value < 0.05), "direction": direction, "practical_caveat": "样本量较小,结果需谨慎" if min(n1, n2) < 5 else None}


def _generic_hints(p_value, direction):
    return {"significant": bool(p_value < 0.05), "direction": direction if p_value < 0.05 else "no_diff", "practical_caveat": None}


def _cohens_magnitude(value):
    value = abs(value)
    if value < 0.2:
        return "negligible"
    if value < 0.5:
        return "small"
    if value < 0.8:
        return "medium"
    return "large"


def _r_magnitude(value):
    value = abs(value)
    if value < 0.1:
        return "negligible"
    if value < 0.3:
        return "small"
    if value < 0.5:
        return "medium"
    return "large"


def _eta_magnitude(value):
    value = abs(value)
    if value < 0.01:
        return "negligible"
    if value < 0.06:
        return "small"
    if value < 0.14:
        return "medium"
    return "large"


def _eta_squared_anova(arrays):
    all_values = np.concatenate([np.asarray(a, dtype=float) for a in arrays])
    grand_mean = all_values.mean()
    ss_between = sum(len(a) * (a.mean() - grand_mean) ** 2 for a in arrays)
    ss_total = sum((all_values - grand_mean) ** 2)
    return float(ss_between / ss_total) if ss_total else 0.0


def _cramers_v(chi2, table):
    n = table.values.sum()
    r, k = table.shape
    denom = n * (min(k - 1, r - 1))
    return float(math.sqrt(chi2 / denom)) if denom else 0.0


def _store_result(result):
    plan = state.get_current_plan()
    if plan:
        plan.results = result


def _store_plot(path):
    plan = state.get_current_plan()
    if plan and path not in plan.plots:
        plan.plots.append(path)


def _method_label():
    plan = state.get_current_plan()
    return plan.selected_method if plan and plan.selected_method else "preview"


def _default_boxplot_path():
    return f"output/boxplot_{_method_label()}_{datetime.now().strftime('%H%M%S')}.png"


def _default_qq_path(group_value):
    label = str(group_value) if group_value else "all"
    return f"output/qq_{label}_{datetime.now().strftime('%H%M%S')}.png"


def _output_path(save_path):
    path = save_path.replace("\\", "/")
    if not path.startswith("output/"):
        path = f"output/{os.path.basename(path)}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value) if not isinstance(value, (dict, list, tuple, np.ndarray)) else False:
        return None
    return value
