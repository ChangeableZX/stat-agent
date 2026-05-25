from agent import StatAgent


if __name__ == "__main__":
    agent = StatAgent()
    user_question = (
        "我有一份数据在 data/demo.csv，里面有 group 列(A/B 两组)和 value 列，"
        "请帮我看看两组的 value 有没有显著差异。"
    )
    agent.run(user_question)
