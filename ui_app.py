import json
import os
from datetime import datetime

import pandas as pd
import streamlit as st

import state
from agent import run_agent_with_callbacks
from state import RegressionPlan, get_current_plan


DATA_DIR = "data"


def init_session():
    defaults = {
        "messages": [],
        "uploaded_path": None,
        "uploaded_name": None,
        "current_plan_snapshot": "",
        "latest_images": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def save_upload(uploaded_file):
    os.makedirs(DATA_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = os.path.basename(uploaded_file.name).replace(" ", "_")
    path = os.path.join(DATA_DIR, f"uploaded_{timestamp}_{safe_name}")
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state.uploaded_path = path.replace("\\", "/")
    st.session_state.uploaded_name = uploaded_file.name


def render_plan(container):
    with container.container():
        st.markdown("### StatPlan 当前状态")
        plan = get_current_plan()
        snapshot = st.session_state.get("current_plan_snapshot", "")
        if not plan and not snapshot:
            st.info("尚未开始分析")
            return
        if not plan:
            st.text(snapshot)
            return

        st.markdown(f"**类型**: `{type(plan).__name__}`")
        st.markdown(f"**研究问题**: {plan.research_question}")
        st.markdown(f"**意图**: `{plan.intent}`")
        if plan.selected_method:
            st.success(f"方法: {plan.selected_method}")
        else:
            st.info("方法: 尚未决定")

        if plan.method_rationale:
            st.markdown("**决策路径**:")
            for item in plan.method_rationale:
                st.markdown(f"- {item}")

        if isinstance(plan, RegressionPlan):
            if plan.final_model:
                st.success(f"最终模型: {plan.final_model}")
            if plan.transformations:
                st.markdown("**修正轨迹**:")
                for item in plan.transformations:
                    st.json(item)
            if plan.diagnostics:
                with st.expander("回归诊断详情", expanded=True):
                    for check, result in plan.diagnostics.items():
                        passed = result.get("passed")
                        marker = "通过" if passed else "警告"
                        p_value = result.get("p_value", result.get("p", "N/A"))
                        st.markdown(f"**{check}**: {marker}")
                        st.caption(f"p = {p_value}")

        if plan.results:
            with st.expander("统计结果摘要"):
                st.json(plan.results)


def render_sidebar():
    st.sidebar.header("数据上传")
    uploaded = st.sidebar.file_uploader("上传 CSV", type=["csv"])
    if uploaded and uploaded.name != st.session_state.get("uploaded_name"):
        save_upload(uploaded)

    if st.session_state.uploaded_path:
        st.sidebar.success(st.session_state.uploaded_path)
        try:
            df = pd.read_csv(st.session_state.uploaded_path)
            st.sidebar.markdown("### 数据预览")
            st.sidebar.write(f"{len(df)} 行, {len(df.columns)} 列")
            st.sidebar.dataframe(df.head(10), use_container_width=True)
            with st.sidebar.expander("列类型"):
                for col, dtype in df.dtypes.items():
                    st.markdown(f"- `{col}`: {dtype}")
        except Exception as exc:
            st.sidebar.error(f"读取上传数据失败: {exc}")
    else:
        st.sidebar.info("可上传 CSV；也可直接让 Agent 读取 data/ 下已有样例。")

    plan_box = st.sidebar.empty()
    render_plan(plan_box)
    return plan_box


def construct_user_msg(user_input):
    parts = [user_input]
    if st.session_state.uploaded_path:
        parts.append(f"\n请使用这个 CSV 文件路径: {st.session_state.uploaded_path}")
    elif "case 15" in user_input.lower() or "case15" in user_input.lower():
        parts.append("\n请使用这个 CSV 文件路径: data/case15_regression_log_y.csv")
    elif "demo" in user_input.lower():
        parts.append("\n请使用这个 CSV 文件路径: data/demo.csv")
    return "\n".join(parts)


def compact_result(result):
    if not isinstance(result, dict):
        return result
    keep = {}
    for key in [
        "error",
        "rows",
        "columns",
        "selected_method",
        "final_model",
        "method",
        "p_value",
        "statistic",
        "r_squared",
        "plot_path",
        "saved_to",
    ]:
        if key in result:
            keep[key] = result[key]
    return keep or result


def collect_images():
    plan = get_current_plan()
    if not plan:
        return []
    return [path for path in plan.plots if isinstance(path, str) and os.path.exists(path)]


def generate_analysis_report(messages, plan):
    lines = [
        "# 统计分析报告",
        "",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    if plan:
        lines.extend(
            [
                f"**研究问题**: {plan.research_question}",
                f"**意图**: {plan.intent}",
                "",
                "## 分析方法",
                f"- 选定方法: {plan.selected_method or '尚未决定'}",
            ]
        )
        if getattr(plan, "final_model", None):
            lines.append(f"- 最终模型: {plan.final_model}")
        if plan.method_rationale:
            lines.append("- 决策路径:")
            lines.extend(f"  - {item}" for item in plan.method_rationale)
        if plan.results:
            lines.extend(["", "## 统计结果", "```json", json.dumps(plan.results, ensure_ascii=False, indent=2), "```"])
        if getattr(plan, "diagnostics", None):
            lines.extend(["", "## 回归诊断", "```json", json.dumps(plan.diagnostics, ensure_ascii=False, indent=2), "```"])

    lines.extend(["", "## 附录:完整对话记录"])
    for msg in messages:
        role = "用户" if msg["role"] == "user" else "Agent"
        lines.extend(["", f"### {role}", msg["content"]])
    return "\n".join(lines)


def main():
    st.set_page_config(page_title="StatAgent", layout="wide")
    init_session()
    st.title("StatAgent - 智能统计推断助手")

    plan_box = render_sidebar()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            for image in msg.get("images", []):
                if os.path.exists(image):
                    st.image(image)

    if user_input := st.chat_input("输入问题..."):
        state.set_current_plan(None)
        st.session_state.current_plan_snapshot = ""
        st.session_state.latest_images = []
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            text_placeholder = st.empty()
            tool_log = st.expander("工具调用过程", expanded=True)
            agent_output = []

            def on_text(text):
                agent_output.append(text)
                text_placeholder.markdown("\n\n".join(agent_output))

            def on_tool_call(name, inputs):
                with tool_log:
                    st.markdown(f"**调用工具**: `{name}`")
                    st.json(inputs)

            def on_tool_result(name, result):
                with tool_log:
                    if isinstance(result, dict) and "error" in result:
                        st.error(f"{name} 返回错误: {result['error']}")
                    else:
                        st.success(f"{name} 完成")
                        with st.expander(f"{name} 返回值"):
                            st.json(compact_result(result))

            def on_plan_update(plan):
                st.session_state.current_plan_snapshot = plan.summary()
                render_plan(plan_box)

            final_response = run_agent_with_callbacks(
                user_msg=construct_user_msg(user_input),
                on_text=on_text,
                on_tool_call=on_tool_call,
                on_tool_result=on_tool_result,
                on_plan_update=on_plan_update,
            )

            images = collect_images()
            for image in images:
                st.image(image)
            content = "\n\n".join(agent_output) or final_response
            st.session_state.messages.append({"role": "assistant", "content": content, "images": images.copy()})

    if st.session_state.messages:
        st.download_button(
            label="下载分析报告 (Markdown)",
            data=generate_analysis_report(st.session_state.messages, get_current_plan()),
            file_name=f"stat_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
        )


if __name__ == "__main__":
    main()
