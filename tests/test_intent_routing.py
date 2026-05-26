import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from tools import make_analysis_plan


def main():
    print("Intent routing test 1: missing required field")
    result = make_analysis_plan(
        "比较多组 value",
        "compare_multi_groups",
        target_variable="value",
    )
    print(result)
    assert "error" in result and "grouping_variable" in result["error"]

    print("\nIntent routing test 2: retry with missing field fixed")
    result = make_analysis_plan(
        "比较多组 value",
        "compare_multi_groups",
        target_variable="value",
        grouping_variable="group",
    )
    print(result["plan_type"], result["intent"], result["target_variable"], result["grouping_variable"])
    assert result["plan_type"] == "MultiGroupPlan"
    assert result["intent"] == "compare_multi_groups"

    print("\nIntent routing test 3: invalid intent")
    result = make_analysis_plan("坏意图", "not_a_real_intent", target_variable="y")
    print(result)
    assert "error" in result and "intent 必须" in result["error"]
    print("PASS")


if __name__ == "__main__":
    main()
