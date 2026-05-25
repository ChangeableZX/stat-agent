import json
import os
import sys

from anthropic import Anthropic
from dotenv import load_dotenv

from prompts import SYSTEM_PROMPT
from tools import check_normality, load_data, run_independent_ttest


load_dotenv()
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


TOOLS = [
    {
        "name": "load_data",
        "description": "Load a CSV file and cache the DataFrame for later statistical tools.",
        "input_schema": {
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"],
        },
    },
    {
        "name": "check_normality",
        "description": "Run Shapiro-Wilk normality test on a numeric column, optionally within one group.",
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
        "name": "run_independent_ttest",
        "description": "Run an independent-samples t-test for exactly two groups.",
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

        self.client = Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            base_url=base_url,
        )
        self.tool_map = {
            "load_data": load_data,
            "check_normality": check_normality,
            "run_independent_ttest": run_independent_ttest,
        }

    def run(self, user_question: str):
        messages = [{"role": "user", "content": user_question}]

        for _ in range(10):
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
                    print(f"\n📊 Tool result:")
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

        print("\nReached max loop count: 10")

    @staticmethod
    def _pretty(obj):
        return json.dumps(obj, ensure_ascii=False, indent=2)
