import json
import os
import sys

from anthropic import Anthropic
from dotenv import load_dotenv

from prompts import SYSTEM_PROMPT
from tools import (
    check_normality,
    check_variance_equality,
    load_data,
    make_analysis_plan,
    plot_boxplot,
    plot_qq,
    run_independent_ttest,
    run_mannwhitney,
    run_paired_ttest,
    run_welch_ttest,
    run_wilcoxon,
    select_method,
)


load_dotenv()
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


TOOLS = [
    {
        "name": "make_analysis_plan",
        "description": "Initialize the central StatPlan from the user's research question.",
        "input_schema": {
            "type": "object",
            "properties": {
                "research_question": {"type": "string"},
                "intent": {"type": "string", "enum": ["compare_two_groups"]},
                "target_variable": {"type": "string"},
                "grouping_variable": {"type": "string"},
                "paired_columns": {"type": "array", "items": {"type": "string"}},
                "design": {"type": "string", "enum": ["independent", "paired"]},
            },
            "required": ["research_question", "intent", "target_variable", "design"],
        },
    },
    {
        "name": "load_data",
        "description": "Load a CSV file, cache the DataFrame, and sync data quality to StatPlan.",
        "input_schema": {
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"],
        },
    },
    {
        "name": "check_normality",
        "description": "Run Shapiro-Wilk normality test on a column, group, or paired diff.",
        "input_schema": {
            "type": "object",
            "properties": {
                "column": {"type": "string"},
                "group_column": {"type": "string"},
                "group_value": {"type": "string"},
            },
            "required": ["column"],
        },
    },
    {
        "name": "check_variance_equality",
        "description": "Run Levene's test for equal variance across exactly two groups.",
        "input_schema": {
            "type": "object",
            "properties": {
                "value_col": {"type": "string"},
                "group_col": {"type": "string"},
            },
            "required": ["value_col", "group_col"],
        },
    },
    {
        "name": "select_method",
        "description": "Select the statistical method from current StatPlan using deterministic rules.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "run_independent_ttest",
        "description": "Run Student's independent-samples t-test.",
        "input_schema": {
            "type": "object",
            "properties": {
                "value_column": {"type": "string"},
                "group_column": {"type": "string"},
                "equal_var": {"type": "boolean"},
            },
            "required": ["value_column", "group_column"],
        },
    },
    {
        "name": "run_welch_ttest",
        "description": "Run Welch's independent-samples t-test.",
        "input_schema": {
            "type": "object",
            "properties": {"value_col": {"type": "string"}, "group_col": {"type": "string"}},
            "required": ["value_col", "group_col"],
        },
    },
    {
        "name": "run_mannwhitney",
        "description": "Run Mann-Whitney U test.",
        "input_schema": {
            "type": "object",
            "properties": {"value_col": {"type": "string"}, "group_col": {"type": "string"}},
            "required": ["value_col", "group_col"],
        },
    },
    {
        "name": "run_paired_ttest",
        "description": "Run paired-samples t-test.",
        "input_schema": {
            "type": "object",
            "properties": {"col1": {"type": "string"}, "col2": {"type": "string"}},
            "required": ["col1", "col2"],
        },
    },
    {
        "name": "run_wilcoxon",
        "description": "Run Wilcoxon signed-rank test.",
        "input_schema": {
            "type": "object",
            "properties": {"col1": {"type": "string"}, "col2": {"type": "string"}},
            "required": ["col1", "col2"],
        },
    },
    {
        "name": "plot_boxplot",
        "description": "Save a boxplot under output/.",
        "input_schema": {
            "type": "object",
            "properties": {
                "value_col": {"type": "string"},
                "group_col": {"type": "string"},
                "save_path": {"type": "string"},
            },
            "required": ["value_col", "group_col", "save_path"],
        },
    },
    {
        "name": "plot_qq",
        "description": "Save a Q-Q plot under output/.",
        "input_schema": {
            "type": "object",
            "properties": {
                "column": {"type": "string"},
                "group_col": {"type": "string"},
                "group_value": {"type": "string"},
                "save_path": {"type": "string"},
            },
            "required": ["column", "save_path"],
        },
    },
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
            "select_method": select_method,
            "run_independent_ttest": run_independent_ttest,
            "run_welch_ttest": run_welch_ttest,
            "run_mannwhitney": run_mannwhitney,
            "run_paired_ttest": run_paired_ttest,
            "run_wilcoxon": run_wilcoxon,
            "plot_boxplot": plot_boxplot,
            "plot_qq": plot_qq,
        }

    def run(self, user_question: str):
        messages = [{"role": "user", "content": user_question}]

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
            else:
                print("\nAgent stopped without tool results.")
                return

        print("\nReached max loop count: 15")

    @staticmethod
    def _pretty(obj):
        return json.dumps(obj, ensure_ascii=False, indent=2)
