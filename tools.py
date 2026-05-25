import math

import numpy as np
import pandas as pd
from scipy import stats


_DATA_CACHE = None


def load_data(file_path: str):
    """Load a CSV file, cache its DataFrame globally, and return a JSON-serializable summary."""
    global _DATA_CACHE
    try:
        df = pd.read_csv(file_path)
        _DATA_CACHE = df
        preview = df.head().replace({np.nan: None}).to_dict(orient="records")
        return {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "preview": preview,
        }
    except Exception as exc:
        return {"error": str(exc)}


def check_normality(column: str, group_column: str = None, group_value: str = None):
    """Run a Shapiro-Wilk normality test on a numeric column, optionally filtered to one group."""
    try:
        if _DATA_CACHE is None:
            return {"error": "No data loaded. Call load_data first."}
        df = _DATA_CACHE
        if column not in df.columns:
            return {"error": f"Column not found: {column}"}
        if group_column is not None:
            if group_column not in df.columns:
                return {"error": f"Group column not found: {group_column}"}
            df = df[df[group_column] == group_value]
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        n = int(len(values))
        if n < 3:
            return {"error": "Shapiro-Wilk test requires at least 3 valid numeric observations."}

        stat, p_value = stats.shapiro(values)
        return {
            "column": column,
            "group_column": group_column,
            "group_value": group_value,
            "n": n,
            "statistic": float(stat),
            "p_value": float(p_value),
            "reject_normality_alpha_0_05": bool(p_value < 0.05),
        }
    except Exception as exc:
        return {"error": str(exc)}


def run_independent_ttest(value_column: str, group_column: str, equal_var: bool = True):
    """Run an independent-samples t-test for exactly two groups and return effect size and CI."""
    try:
        if _DATA_CACHE is None:
            return {"error": "No data loaded. Call load_data first."}
        df = _DATA_CACHE
        if value_column not in df.columns:
            return {"error": f"Value column not found: {value_column}"}
        if group_column not in df.columns:
            return {"error": f"Group column not found: {group_column}"}

        groups = [g for g in df[group_column].dropna().unique().tolist()]
        if len(groups) != 2:
            return {"error": f"Expected exactly 2 groups in {group_column}, found {len(groups)}."}

        x1 = pd.to_numeric(df[df[group_column] == groups[0]][value_column], errors="coerce").dropna()
        x2 = pd.to_numeric(df[df[group_column] == groups[1]][value_column], errors="coerce").dropna()
        n1, n2 = int(len(x1)), int(len(x2))
        if n1 < 2 or n2 < 2:
            return {"error": "Each group needs at least 2 valid numeric observations."}

        t_stat, p_value = stats.ttest_ind(x1, x2, equal_var=equal_var)
        mean1, mean2 = float(x1.mean()), float(x2.mean())
        var1, var2 = float(x1.var(ddof=1)), float(x2.var(ddof=1))

        if equal_var:
            dfree = n1 + n2 - 2
            pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / dfree
            se = math.sqrt(pooled_var * (1 / n1 + 1 / n2))
        else:
            se = math.sqrt(var1 / n1 + var2 / n2)
            numerator = (var1 / n1 + var2 / n2) ** 2
            denominator = (var1 ** 2) / ((n1 ** 2) * (n1 - 1)) + (var2 ** 2) / ((n2 ** 2) * (n2 - 1))
            dfree = numerator / denominator

        mean_diff = mean1 - mean2
        t_crit = stats.t.ppf(0.975, dfree)
        ci_low = mean_diff - t_crit * se
        ci_high = mean_diff + t_crit * se
        pooled_sd = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        cohens_d = mean_diff / pooled_sd if pooled_sd else float("nan")

        return {
            "test": "Student independent t-test" if equal_var else "Welch independent t-test",
            "group_column": group_column,
            "value_column": value_column,
            "groups": [str(groups[0]), str(groups[1])],
            "n": {str(groups[0]): n1, str(groups[1]): n2},
            "means": {str(groups[0]): mean1, str(groups[1]): mean2},
            "mean_difference_first_minus_second": float(mean_diff),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "degrees_of_freedom": float(dfree),
            "confidence_interval_95": [float(ci_low), float(ci_high)],
            "cohens_d": float(cohens_d),
        }
    except Exception as exc:
        return {"error": str(exc)}
