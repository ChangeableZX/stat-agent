import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from tools import (
    load_data,
    make_analysis_plan,
    run_independent_ttest,
    run_one_way_anova,
)
from agent import StatAgent
from prompts import SYSTEM_PROMPT


def main():
    print("Type guard test 1: TwoGroupPlan -> run_one_way_anova")
    make_analysis_plan(
        "两组比较",
        "compare_two_groups",
        target_variable="value",
        grouping_variable="group",
        design="independent",
    )
    load_data("data/demo.csv")
    result = run_one_way_anova("value", "group")
    print(result)
    assert "error" in result and "MultiGroupPlan" in result["error"]

    print("\nType guard test 2: CorrelationPlan -> run_independent_ttest")
    make_analysis_plan("相关性", "correlation", x_variable="x", y_variable="y")
    result = run_independent_ttest("value", "group")
    print(result)
    assert "error" in result and "TwoGroupPlan" in result["error"]

    print("\nType guard test 3: LLM-facing reaction to explicit tool error")
    agent = StatAgent()
    messages = [
        {
            "role": "user",
            "content": (
                "工具返回了这个错误：run_one_way_anova 要求 MultiGroupPlan,当前是 TwoGroupPlan。"
                "请用一句中文回答下一步应该怎么处理，必须包含 MultiGroupPlan 或 compare_multi_groups。不要调用工具。"
            ),
        }
    ]
    response = agent.client.messages.create(
        model=agent.model,
        max_tokens=256,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    answer = "\n".join(getattr(block, "text", "") for block in response.content if block.type == "text")
    print(answer)
    assert answer.strip(), [block.model_dump(exclude_none=True) for block in response.content]
    assert "MultiGroupPlan" in answer or "compare_multi_groups" in answer or "重新" in answer
    print("PASS")


if __name__ == "__main__":
    main()
