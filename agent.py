import json
import os
import sys
from typing import Any, Callable

from anthropic import Anthropic
from dotenv import load_dotenv

import state
from prompts import SYSTEM_PROMPT
from tools import (
    build_contingency_table,
    check_bivariate_normality,
    check_expected_frequencies,
    check_linearity,
    check_homoscedasticity,
    check_independence,
    check_multicollinearity,
    check_normality,
    check_outliers,
    check_residual_normality,
    check_sphericity,
    check_variance_equality,
    load_data,
    make_analysis_plan,
    plot_boxplot,
    plot_grouped_boxplot,
    plot_mosaic,
    plot_qq,
    plot_qq_residuals,
    plot_regression_fit,
    plot_residuals,
    plot_scatter,
    power_analysis_anova,
    power_analysis_correlation,
    power_analysis_ttest,
    run_chi_square,
    run_fisher_exact,
    run_fisher_freeman_halton,
    run_friedman,
    run_independent_ttest,
    run_kendall,
    run_kruskal_wallis,
    run_mannwhitney,
    run_mcnemar,
    run_mcnemar_bowker,
    run_multiple_linear_regression,
    run_ols_robust_se,
    run_ols_with_log_y,
    run_one_way_anova,
    run_paired_ttest,
    run_pearson,
    run_posthoc_dunn,
    run_posthoc_games_howell,
    run_posthoc_tukey,
    run_repeated_anova,
    run_simple_linear_regression,
    run_spearman,
    run_welch_anova,
    run_welch_ttest,
    run_wilcoxon,
    select_final_model,
    select_method,
)


load_dotenv()
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def _tool(name, description, properties=None, required=None):
    return {
        "name": name,
        "description": description,
        "input_schema": {"type": "object", "properties": properties or {}, "required": required or []},
    }


TOOLS = [
    _tool(
        "make_analysis_plan",
        "Initialize the central StatPlan. Supports two-group, multi-group, correlation, categorical, and regression intents.",
        {
            "research_question": {"type": "string"},
            "intent": {
                "type": "string",
                "enum": ["compare_two_groups", "compare_multi_groups", "correlation", "categorical_test", "regression"],
            },
            "target_variable": {"type": "string"},
            "grouping_variable": {"type": "string"},
            "paired_columns": {"type": "array", "items": {"type": "string"}},
            "design": {"type": "string", "enum": ["independent", "paired", "repeated_measures"]},
            "x_variable": {"type": "string"},
            "y_variable": {"type": "string"},
            "x_type": {"type": "string", "enum": ["continuous", "ordinal"]},
            "y_type": {"type": "string", "enum": ["continuous", "ordinal"]},
            "row_variable": {"type": "string"},
            "col_variable": {"type": "string"},
            "paired": {"type": "boolean"},
            "x_variables": {"type": "array", "items": {"type": "string"}},
            "regression_type": {"type": "string", "enum": ["linear"]},
        },
        ["research_question", "intent"],
    ),
    _tool("load_data", "Load a CSV file and sync data quality to StatPlan.", {"file_path": {"type": "string"}}, ["file_path"]),
    _tool(
        "check_normality",
        "Run Shapiro-Wilk normality test on a column, group, or paired diff.",
        {"column": {"type": "string"}, "group_column": {"type": "string"}, "group_value": {"type": "string"}},
        ["column"],
    ),
    _tool("check_variance_equality", "Run Levene's test. Use group_values to restrict to specific groups when the column has more than 2 groups.", {"value_col": {"type": "string"}, "group_col": {"type": "string"}, "group_values": {"type": "array", "items": {"type": "string"}}}, ["value_col", "group_col"]),
    _tool("check_sphericity", "Run Mauchly sphericity test.", {"subject_col": {"type": "string"}, "within_col": {"type": "string"}, "value_col": {"type": "string"}}, ["subject_col", "within_col", "value_col"]),
    _tool("check_bivariate_normality", "Check simplified bivariate normality.", {"x_col": {"type": "string"}, "y_col": {"type": "string"}}, ["x_col", "y_col"]),
    _tool("check_linearity", "Check simplified linearity.", {"x_col": {"type": "string"}, "y_col": {"type": "string"}}, ["x_col", "y_col"]),
    _tool("check_expected_frequencies", "Check expected frequencies for contingency table.", {"row_col": {"type": "string"}, "col_col": {"type": "string"}}, ["row_col", "col_col"]),
    _tool("build_contingency_table", "Build contingency table.", {"row_col": {"type": "string"}, "col_col": {"type": "string"}}, ["row_col", "col_col"]),
    _tool("select_method", "Select the statistical method from current StatPlan using deterministic rules."),
    _tool("run_simple_linear_regression", "Run simple linear regression.", {"y_col": {"type": "string"}, "x_col": {"type": "string"}}, ["y_col", "x_col"]),
    _tool("run_multiple_linear_regression", "Run multiple linear regression.", {"y_col": {"type": "string"}, "x_cols": {"type": "array", "items": {"type": "string"}}}, ["y_col", "x_cols"]),
    _tool("run_ols_with_log_y", "Run OLS after log-transforming Y.", {"y_col": {"type": "string"}, "x_cols": {"type": "array", "items": {"type": "string"}}}, ["y_col", "x_cols"]),
    _tool("run_ols_robust_se", "Run OLS with HC3 robust standard errors.", {"y_col": {"type": "string"}, "x_cols": {"type": "array", "items": {"type": "string"}}}, ["y_col", "x_cols"]),
    _tool("select_final_model", "Select final regression model from diagnostics."),
    _tool("check_residual_normality", "Run Shapiro-Wilk on OLS residuals.", {"y_col": {"type": "string"}, "x_cols": {"type": "array", "items": {"type": "string"}}}, ["y_col", "x_cols"]),
    _tool("check_homoscedasticity", "Run Breusch-Pagan regression diagnostic.", {"y_col": {"type": "string"}, "x_cols": {"type": "array", "items": {"type": "string"}}}, ["y_col", "x_cols"]),
    _tool("check_independence", "Compute Durbin-Watson regression diagnostic.", {"y_col": {"type": "string"}, "x_cols": {"type": "array", "items": {"type": "string"}}}, ["y_col", "x_cols"]),
    _tool("check_multicollinearity", "Compute VIF for regression predictors.", {"x_cols": {"type": "array", "items": {"type": "string"}}}, ["x_cols"]),
    _tool("check_outliers", "Check Cook's distance influential points.", {"y_col": {"type": "string"}, "x_cols": {"type": "array", "items": {"type": "string"}}}, ["y_col", "x_cols"]),
    _tool("run_independent_ttest", "Run Student independent t-test. If the grouping column has more than 2 groups but the user wants to compare only 2 specific groups, pass group_values=[\"GroupA\", \"GroupB\"] to filter.", {"value_column": {"type": "string"}, "group_column": {"type": "string"}, "equal_var": {"type": "boolean"}, "group_values": {"type": "array", "items": {"type": "string"}}}, ["value_column", "group_column"]),
    _tool("run_welch_ttest", "Run Welch independent t-test. Use group_values to filter to 2 specific groups when the column has more than 2 groups.", {"value_col": {"type": "string"}, "group_col": {"type": "string"}, "group_values": {"type": "array", "items": {"type": "string"}}}, ["value_col", "group_col"]),
    _tool("run_mannwhitney", "Run Mann-Whitney U test. Use group_values to filter to 2 specific groups when the column has more than 2 groups.", {"value_col": {"type": "string"}, "group_col": {"type": "string"}, "group_values": {"type": "array", "items": {"type": "string"}}}, ["value_col", "group_col"]),
    _tool("run_paired_ttest", "Run paired t-test.", {"col1": {"type": "string"}, "col2": {"type": "string"}}, ["col1", "col2"]),
    _tool("run_wilcoxon", "Run Wilcoxon signed-rank test.", {"col1": {"type": "string"}, "col2": {"type": "string"}}, ["col1", "col2"]),
    _tool("run_one_way_anova", "Run one-way ANOVA.", {"value_col": {"type": "string"}, "group_col": {"type": "string"}}, ["value_col", "group_col"]),
    _tool("run_welch_anova", "Run Welch ANOVA.", {"value_col": {"type": "string"}, "group_col": {"type": "string"}}, ["value_col", "group_col"]),
    _tool("run_kruskal_wallis", "Run Kruskal-Wallis test.", {"value_col": {"type": "string"}, "group_col": {"type": "string"}}, ["value_col", "group_col"]),
    _tool("run_repeated_anova", "Run repeated-measures ANOVA.", {"subject_col": {"type": "string"}, "within_col": {"type": "string"}, "value_col": {"type": "string"}}, ["subject_col", "within_col", "value_col"]),
    _tool("run_friedman", "Run Friedman test.", {"subject_col": {"type": "string"}, "within_col": {"type": "string"}, "value_col": {"type": "string"}}, ["subject_col", "within_col", "value_col"]),
    _tool("run_posthoc_tukey", "Run Tukey HSD posthoc.", {"value_col": {"type": "string"}, "group_col": {"type": "string"}}, ["value_col", "group_col"]),
    _tool("run_posthoc_games_howell", "Run Games-Howell posthoc.", {"value_col": {"type": "string"}, "group_col": {"type": "string"}}, ["value_col", "group_col"]),
    _tool("run_posthoc_dunn", "Run Dunn posthoc.", {"value_col": {"type": "string"}, "group_col": {"type": "string"}, "p_adjust": {"type": "string"}}, ["value_col", "group_col"]),
    _tool("run_pearson", "Run Pearson correlation.", {"x_col": {"type": "string"}, "y_col": {"type": "string"}}, ["x_col", "y_col"]),
    _tool("run_spearman", "Run Spearman correlation.", {"x_col": {"type": "string"}, "y_col": {"type": "string"}}, ["x_col", "y_col"]),
    _tool("run_kendall", "Run Kendall correlation.", {"x_col": {"type": "string"}, "y_col": {"type": "string"}}, ["x_col", "y_col"]),
    _tool("run_chi_square", "Run chi-square test.", {"row_col": {"type": "string"}, "col_col": {"type": "string"}, "yates": {"type": "boolean"}}, ["row_col", "col_col"]),
    _tool("run_fisher_exact", "Run Fisher exact test.", {"row_col": {"type": "string"}, "col_col": {"type": "string"}}, ["row_col", "col_col"]),
    _tool("run_fisher_freeman_halton", "Run approximate Fisher-Freeman-Halton test.", {"row_col": {"type": "string"}, "col_col": {"type": "string"}}, ["row_col", "col_col"]),
    _tool("run_mcnemar", "Run McNemar test.", {"row_col": {"type": "string"}, "col_col": {"type": "string"}}, ["row_col", "col_col"]),
    _tool("run_mcnemar_bowker", "Run McNemar-Bowker test.", {"row_col": {"type": "string"}, "col_col": {"type": "string"}}, ["row_col", "col_col"]),
    _tool("plot_boxplot", "Save a boxplot.", {"value_col": {"type": "string"}, "group_col": {"type": "string"}, "save_path": {"type": "string"}}, ["value_col", "group_col"]),
    _tool("plot_qq", "Save a Q-Q plot.", {"column": {"type": "string"}, "group_col": {"type": "string"}, "group_value": {"type": "string"}, "save_path": {"type": "string"}}, ["column"]),
    _tool("plot_grouped_boxplot", "Save grouped boxplot.", {"value_col": {"type": "string"}, "group_col": {"type": "string"}}, ["value_col", "group_col"]),
    _tool("plot_scatter", "Save scatter plot.", {"x_col": {"type": "string"}, "y_col": {"type": "string"}, "fit_line": {"type": "boolean"}}, ["x_col", "y_col"]),
    _tool("plot_mosaic", "Save categorical mosaic-like plot.", {"row_col": {"type": "string"}, "col_col": {"type": "string"}}, ["row_col", "col_col"]),
    _tool("plot_residuals", "Save residuals vs fitted plot.", {"y_col": {"type": "string"}, "x_cols": {"type": "array", "items": {"type": "string"}}}, ["y_col", "x_cols"]),
    _tool("plot_qq_residuals", "Save residual Q-Q plot.", {"y_col": {"type": "string"}, "x_cols": {"type": "array", "items": {"type": "string"}}}, ["y_col", "x_cols"]),
    _tool("plot_regression_fit", "Save simple regression fit plot.", {"y_col": {"type": "string"}, "x_col": {"type": "string"}}, ["y_col", "x_col"]),
    _tool("power_analysis_ttest", "Power analysis for two-sample t-test.", {"effect_size": {"type": ["number", "null"]}, "alpha": {"type": ["number", "null"]}, "power": {"type": ["number", "null"]}, "n": {"type": ["number", "null"]}}),
    _tool("power_analysis_anova", "Power analysis for one-way ANOVA.", {"effect_size": {"type": ["number", "null"]}, "n_groups": {"type": "integer"}, "alpha": {"type": ["number", "null"]}, "power": {"type": ["number", "null"]}, "n": {"type": ["number", "null"]}}, ["n_groups"]),
    _tool("power_analysis_correlation", "Power analysis for Pearson correlation.", {"r": {"type": ["number", "null"]}, "alpha": {"type": ["number", "null"]}, "power": {"type": ["number", "null"]}, "n": {"type": ["number", "null"]}}),
]


class StatAgent:
    def __init__(self):
        self.model = os.environ.get("ANTHROPIC_MODEL")
        self.max_tokens = int(os.environ.get("ANTHROPIC_MAX_TOKENS", "2048"))
        base_url = os.environ.get("ANTHROPIC_BASE_URL")

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ValueError("Missing ANTHROPIC_API_KEY in .env or environment.")
        if not base_url:
            raise ValueError("Missing ANTHROPIC_BASE_URL in .env or environment.")
        if not self.model:
            raise ValueError("Missing ANTHROPIC_MODEL in .env or environment.")

        self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"), base_url=base_url)
        self.tool_map = {
            "make_analysis_plan": make_analysis_plan,
            "load_data": load_data,
            "check_normality": check_normality,
            "check_variance_equality": check_variance_equality,
            "check_sphericity": check_sphericity,
            "check_bivariate_normality": check_bivariate_normality,
            "check_linearity": check_linearity,
            "check_expected_frequencies": check_expected_frequencies,
            "build_contingency_table": build_contingency_table,
            "select_method": select_method,
            "run_simple_linear_regression": run_simple_linear_regression,
            "run_multiple_linear_regression": run_multiple_linear_regression,
            "run_ols_with_log_y": run_ols_with_log_y,
            "run_ols_robust_se": run_ols_robust_se,
            "select_final_model": select_final_model,
            "check_residual_normality": check_residual_normality,
            "check_homoscedasticity": check_homoscedasticity,
            "check_independence": check_independence,
            "check_multicollinearity": check_multicollinearity,
            "check_outliers": check_outliers,
            "run_independent_ttest": run_independent_ttest,
            "run_welch_ttest": run_welch_ttest,
            "run_mannwhitney": run_mannwhitney,
            "run_paired_ttest": run_paired_ttest,
            "run_wilcoxon": run_wilcoxon,
            "run_one_way_anova": run_one_way_anova,
            "run_welch_anova": run_welch_anova,
            "run_kruskal_wallis": run_kruskal_wallis,
            "run_repeated_anova": run_repeated_anova,
            "run_friedman": run_friedman,
            "run_posthoc_tukey": run_posthoc_tukey,
            "run_posthoc_games_howell": run_posthoc_games_howell,
            "run_posthoc_dunn": run_posthoc_dunn,
            "run_pearson": run_pearson,
            "run_spearman": run_spearman,
            "run_kendall": run_kendall,
            "run_chi_square": run_chi_square,
            "run_fisher_exact": run_fisher_exact,
            "run_fisher_freeman_halton": run_fisher_freeman_halton,
            "run_mcnemar": run_mcnemar,
            "run_mcnemar_bowker": run_mcnemar_bowker,
            "plot_boxplot": plot_boxplot,
            "plot_qq": plot_qq,
            "plot_grouped_boxplot": plot_grouped_boxplot,
            "plot_scatter": plot_scatter,
            "plot_mosaic": plot_mosaic,
            "plot_residuals": plot_residuals,
            "plot_qq_residuals": plot_qq_residuals,
            "plot_regression_fit": plot_regression_fit,
            "power_analysis_ttest": power_analysis_ttest,
            "power_analysis_anova": power_analysis_anova,
            "power_analysis_correlation": power_analysis_correlation,
        }

    def run(self, user_question: str):
        messages = [{"role": "user", "content": user_question}]
        self.messages = messages

        for _ in range(15):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                messages=messages,
                tools=TOOLS,
            )
            assistant_content = [block.model_dump(exclude_none=True) for block in response.content]
            messages.append({"role": "assistant", "content": assistant_content})

            tool_results = []
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    print(f"\n🤖 Agent:\n{block.text.strip()}")
                elif block.type == "tool_use":
                    print(f"\n🔧 Tool call: {block.name}")
                    print(self._pretty(block.input))
                    result = self.tool_map[block.name](**block.input)
                    print("\n📊 Tool result:")
                    print(self._pretty(result))
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )

            if response.stop_reason == "end_turn":
                return
            if tool_results:
                messages.append({"role": "user", "content": tool_results})
                if state.current_plan:
                    messages.append(
                        {
                            "role": "user",
                            "content": f"[StatPlan 当前状态]\n\n{state.current_plan.summary()}",
                        }
                    )
            else:
                print("\nAgent stopped without tool results.")
                return

        print("\nReached max loop count: 15")

    @staticmethod
    def _pretty(obj):
        return json.dumps(obj, ensure_ascii=False, indent=2)


def run_agent_with_callbacks(
    user_msg: str,
    on_text: Callable[[str], None] = None,
    on_tool_call: Callable[[str, dict], None] = None,
    on_tool_result: Callable[[str, dict], None] = None,
    on_plan_update: Callable[[Any], None] = None,
    on_complete: Callable[[str], None] = None,
) -> str:
    """Run StatAgent with optional callbacks for UI display."""
    agent = StatAgent()
    messages = [{"role": "user", "content": user_msg}]
    agent.messages = messages
    final_text = []

    for _ in range(15):
        response = agent.client.messages.create(
            model=agent.model,
            max_tokens=agent.max_tokens,
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=TOOLS,
        )
        assistant_content = [block.model_dump(exclude_none=True) for block in response.content]
        messages.append({"role": "assistant", "content": assistant_content})

        tool_results = []
        for block in response.content:
            if block.type == "text" and block.text.strip():
                final_text.append(block.text.strip())
                if on_text:
                    on_text(block.text.strip())
            elif block.type == "tool_use":
                if on_tool_call:
                    on_tool_call(block.name, block.input)
                result = agent.tool_map[block.name](**block.input)
                if on_tool_result:
                    on_tool_result(block.name, result)
                if state.current_plan and on_plan_update:
                    on_plan_update(state.current_plan)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

        if response.stop_reason == "end_turn":
            output = "\n\n".join(final_text)
            if on_complete:
                on_complete(output)
            return output
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
            if state.current_plan:
                messages.append(
                    {
                        "role": "user",
                        "content": f"[StatPlan 当前状态]\n\n{state.current_plan.summary()}",
                    }
                )
                if on_plan_update:
                    on_plan_update(state.current_plan)
        else:
            break

    output = "\n\n".join(final_text)
    if on_complete:
        on_complete(output)
    return output
