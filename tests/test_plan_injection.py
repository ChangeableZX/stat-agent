import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from agent import StatAgent
from prompts import SYSTEM_PROMPT


def main():
    agent = StatAgent()
    question = (
        "我有一份数据在 data/demo.csv，里面有 group 列(A/B 两组)和 value 列，"
        "请帮我看看两组的 value 有没有显著差异。"
    )
    agent.run(question)

    followup = "你刚才走的决策路径里，第二步检查了什么？Levene 检验的 p 值是多少？"
    agent.messages.append({"role": "user", "content": followup})
    response = agent.client.messages.create(
        model=agent.model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=agent.messages,
    )

    print("\n===== Follow-up Answer =====")
    for block in response.content:
        if block.type == "text":
            print(block.text)


if __name__ == "__main__":
    main()
