import math
import os

import numpy as np
import pandas as pd
from scipy import stats

import state
from decision import select_method_for_two_group
from state import StatPlan


_DATA_CACHE = None
_MISSING_MARKERS = {"", "na", "n/a", "nan", "null", "none"}


def make_analysis_plan(
    research_question: str,
    intent: str,
    target_variable: str,
    grouping_variable: str = None,
    paired_columns=None,
    design: str = "independent",
):
    """Initialize the global StatPlan from the LLM's interpretation of the user question."""
    try:
        if paired_columns:
            paired_columns = tuple(paired_columns)
        plan = StatPlan(
            research_question=research_question,
            intent=intent,
            target_variable=target_variable,
            grouping_variable=grouping_variable,
            paired_columns=paired_columns,
            design=design,
        )
        state.set_current_plan(plan)
        return plan.to_dict()
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
            non_numeric = 0
            if has_numeric_values:
                non_numeric = int((~lowered.isin(_MISSING_MARKERS) & parsed.isna()).sum())
            if missing_like or non_numeric:
                data_quality[col] = {
                    "missing_like_values": missing_like,
                    "non_numeric_values": non_numeric,
                }
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
            if plan.grouping_variable in value_counts:
                plan.sample_sizes = value_counts[plan.grouping_variable]
        return result
    except Exception as exc:
        return {"error": str(exc)}


def check_normality(column: str, group_column: str = None, group_value: str = None):
    """Run a Shapiro-Wilk normality test on a numeric column, optionally filtered to one group."""
    try:
        if _DATA_CACHE is None:
            return {"error": "No data loaded. Call load_data first."}
        df = _DATA_CACHE
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


def check_variance_equality(value_col: str, group_col: str):
    """Run Levene's test for equal variance across exactly two groups."""
    try:
        groups, x1, x2 = _two_group_values(value_col, group_col)
        stat, p_value = stats.levene(x1, x2)
        result = {
            "statistic": float(stat),
            "p_value": float(p_value),
            "equal_variance": bool(p_value >= 0.05),
            "variances": {str(groups[0]): float(x1.var(ddof=1)), str(groups[1]): float(x2.var(ddof=1))},
        }
        plan = state.get_current_plan()
        if plan:
            plan.assumption_checks["variance_equality"] = result
        return result
    except Exception as exc:
        return {"error": str(exc)}


def select_method():
    """Select a statistical method from the current StatPlan using deterministic Python rules."""
    try:
        plan = state.get_current_plan()
        if plan is None:
            return {"error": "No StatPlan exists. Call make_analysis_plan first."}
        plan = select_method_for_two_group(plan)
        state.set_current_plan(plan)
        return {
            "selected_method": plan.selected_method,
            "method_rationale": plan.method_rationale,
            "plan_summary": plan.summary(),
        }
    except Exception as exc:
        return {"error": str(exc)}


def run_independent_ttest(value_column: str, group_column: str, equal_var: bool = True):
    """Run an independent-samples t-test for exactly two groups and return effect size and CI."""
    try:
        groups, x1, x2, raw1, raw2 = _two_group_values(value_column, group_column, include_raw=True)
        result = _independent_ttest_result(groups, x1, x2, raw1, raw2, equal_var)
        _store_result(result)
        return result
    except Exception as exc:
        return {"error": str(exc)}


def run_welch_ttest(value_col: str, group_col: str):
    """Run Welch's independent-samples t-test."""
    return run_independent_ttest(value_col, group_col, equal_var=False)


def run_mannwhitney(value_col: str, group_col: str):
    """Run Mann-Whitney U test for two independent groups."""
    try:
        groups, x1, x2, raw1, raw2 = _two_group_values(value_col, group_col, include_raw=True)
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
        x, y = _paired_values(col1, col2)
        diff = x - y
        stat, p_value = stats.ttest_rel(x, y)
        n = len(diff)
        mean_diff = float(diff.mean())
        se = float(diff.std(ddof=1) / math.sqrt(n))
        dfree = n - 1
        t_crit = stats.t.ppf(0.975, dfree)
        ci = [mean_diff - t_crit * se, mean_diff + t_crit * se]
        dz = mean_diff / float(diff.std(ddof=1)) if float(diff.std(ddof=1)) else float("nan")
        result = {
            "method": "paired_t",
            "statistic": float(stat),
            "p_value": float(p_value),
            "effect_size": {"name": "cohens_dz", "value": float(dz), "magnitude": _cohens_magnitude(dz)},
            "ci_95": [float(ci[0]), float(ci[1])],
            "group_stats": {
                col1: _one_group_stats(x),
                col2: _one_group_stats(y),
            },
            "interpretation_hints": _hints(float(p_value), float(y.mean() - x.mean()), n, n),
        }
        _store_result(result)
        return result
    except Exception as exc:
        return {"error": str(exc)}


def run_wilcoxon(col1: str, col2: str):
    """Run Wilcoxon signed-rank test for paired columns."""
    try:
        x, y = _paired_values(col1, col2)
        stat, p_value = stats.wilcoxon(x, y)
        n = len(x)
        z_approx = stats.norm.isf(float(p_value) / 2)
        r_value = abs(z_approx) / math.sqrt(n)
        result = {
            "method": "wilcoxon",
            "statistic": float(stat),
            "p_value": float(p_value),
            "effect_size": {"name": "r", "value": float(r_value), "magnitude": _r_magnitude(r_value)},
            "ci_95": [None, None],
            "group_stats": {col1: _one_group_stats(x), col2: _one_group_stats(y)},
            "interpretation_hints": _hints(float(p_value), float(y.median() - x.median()), n, n),
        }
        _store_result(result)
        return result
    except Exception as exc:
        return {"error": str(exc)}


def plot_boxplot(value_col: str, group_col: str, save_path: str):
    """Save a boxplot under output/ and record the path in the current plan."""
    try:
        import matplotlib.pyplot as plt

        df = _require_data()
        groups = [g for g in df[group_col].dropna().unique().tolist()]
        data = [pd.to_numeric(df[df[group_col] == g][value_col], errors="coerce").dropna() for g in groups]
        path = _output_path(save_path)
        plt.figure(figsize=(6, 4))
        plt.boxplot(data, labels=[str(g) for g in groups])
        plt.ylabel(value_col)
        plt.xlabel(group_col)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        _store_plot(path)
        return {"plot_path": path}
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
        path = _output_path(save_path)
        plt.figure(figsize=(5, 5))
        stats.probplot(values, dist="norm", plot=plt)
        plt.title(f"Q-Q plot: {title}")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        _store_plot(path)
        return {"plot_path": path}
    except Exception as exc:
        return {"error": str(exc)}


def _require_data():
    if _DATA_CACHE is None:
        raise ValueError("No data loaded. Call load_data first.")
    return _DATA_CACHE


def _two_group_values(value_col, group_col, include_raw=False):
    df = _require_data()
    if value_col not in df.columns:
        raise ValueError(f"Value column not found: {value_col}")
    if group_col not in df.columns:
        raise ValueError(f"Group column not found: {group_col}")
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
    if column not in {"diff", "difference", "paired_diff"} or not plan or not plan.paired_columns:
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


def _one_group_stats(values):
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "median": float(values.median()),
    }


def _hints(p_value, diff_second_minus_first, n1, n2, groups=None):
    if p_value >= 0.05:
        direction = "no_diff"
    elif groups is None:
        direction = "after > before" if diff_second_minus_first > 0 else "before > after"
    else:
        direction = f"{groups[1]} > {groups[0]}" if diff_second_minus_first > 0 else f"{groups[0]} > {groups[1]}"
    return {
        "significant": bool(p_value < 0.05),
        "direction": direction,
        "practical_caveat": "样本量较小,结果需谨慎" if min(n1, n2) < 5 else None,
    }


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


def _store_result(result):
    plan = state.get_current_plan()
    if plan:
        plan.results = result


def _store_plot(path):
    plan = state.get_current_plan()
    if plan and path not in plan.plots:
        plan.plots.append(path)


def _output_path(save_path):
    path = save_path.replace("\\", "/")
    if not path.startswith("output/"):
        path = f"output/{os.path.basename(path)}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path
