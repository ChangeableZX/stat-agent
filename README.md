# StatAgent

StatAgent 是一个基于 LLM 的统计推断 Agent。它面向 CSV 数据分析场景，通过自然语言问题自动完成统计分析计划、数据加载、前提检验、方法选择、统计计算、可视化和中文解释。

项目当前已覆盖 5 类分析意图：两组比较、多组比较、相关性分析、分类变量检验、线性回归；同时提供独立的 Power 分析工具组。最新版本还提供 Streamlit UI，可实时展示 Agent 的工具调用过程和 `StatPlan` 内部状态。

## 核心特性

- 自然语言驱动：用户用中文描述分析问题，Agent 自动规划统计流程。
- 确定性方法选择：统计方法必须通过 `decision.py` 中的规则路由选择，避免 LLM 自行拍脑袋决定方法。
- 状态可追踪：`StatPlan` 保存研究问题、意图、前提检验、方法选择、结果、图表和解释。
- 工具化执行：所有统计计算都通过 `tools.py` 中的工具函数完成。
- 回归诊断链：线性回归支持“初始模型 -> 诊断 -> 修正 -> 再拟合”的多步推理。
- Streamlit UI：支持上传 CSV、聊天式提问、实时工具日志、侧边栏 StatPlan 状态、图表展示和 Markdown 报告下载。

## 支持的统计能力

### 两组比较

- 独立样本 Student t 检验
- Welch t 检验
- Mann-Whitney U 检验
- 配对 t 检验
- Wilcoxon signed-rank 检验
- Shapiro-Wilk 正态性检验
- Levene 方差齐性检验

### 多组比较

- One-way ANOVA
- Welch ANOVA
- Kruskal-Wallis 检验
- 重复测量 ANOVA
- Friedman 检验
- Tukey HSD、Games-Howell、Dunn-BH 事后检验

### 相关性分析

- Pearson 相关
- Spearman 相关
- Kendall 相关
- 简化双变量正态性检查
- 线性关系检查

### 分类变量检验

- Chi-square 检验
- Yates 校正卡方检验
- Fisher exact 检验
- Fisher-Freeman-Halton 近似检验
- McNemar 检验
- McNemar-Bowker 检验

### 线性回归

- 简单线性回归
- 多元线性回归
- 对数 Y 变换后 OLS
- HC3 / White 稳健标准误
- 残差正态性诊断
- Breusch-Pagan 异方差诊断
- Durbin-Watson 独立性诊断
- VIF 多重共线性诊断
- Cook's distance 异常/影响点诊断

### Power 分析

Power 分析不进入 intent 体系，属于事前实验设计工具。

- t 检验 Power / 样本量 / 效应量计算
- ANOVA Power / 样本量 / 效应量计算
- 相关性 Power / 样本量 / 效应量计算

## 项目结构

```text
stat-agent/
  agent.py                  # Agent 主循环、工具注册、UI 回调运行入口
  decision.py               # 统计方法选择规则
  state.py                  # StatPlan 状态对象
  tools.py                  # 统计工具、诊断工具、绘图工具、Power 工具
  prompts.py                # 系统提示词和工作流约束
  main.py                   # 命令行 demo 入口
  ui_app.py                 # Streamlit UI 入口
  run_ui.bat                # Windows 一键启动 UI
  requirements.txt          # Python 依赖
  data/                     # 示例数据与测试数据生成脚本
  tests/                    # 验证测试
  docs/test-logs/           # 教科书数据验证报告
  output/                   # 生成图表输出目录
```

## 安装

推荐使用项目当前验证过的 Conda 环境：

```powershell
D:\anaconda3\envs\agent\python.exe -m pip install -r requirements.txt
```

也可以在自己的 Python 环境中安装：

```bash
pip install -r requirements.txt
```

## 环境变量

在 `stat-agent/.env` 中配置 Anthropic-compatible 接口：

```env
ANTHROPIC_API_KEY=your_api_key_here
ANTHROPIC_BASE_URL=https://your-compatible-endpoint
ANTHROPIC_MODEL=your-model-name
ANTHROPIC_MAX_TOKENS=2048
```

`agent.py` 会通过 `python-dotenv` 自动加载 `.env`。

## 命令行运行

在 `stat-agent/` 目录下执行：

```powershell
D:\anaconda3\envs\agent\python.exe main.py
```

`main.py` 会运行内置 demo：读取 `data/demo.csv`，比较 A/B 两组的 `value` 是否存在显著差异。

## Streamlit UI 运行

方式一：命令行启动。

```powershell
cd C:\Users\changeable\Desktop\agent\stat-agent
D:\anaconda3\envs\agent\python.exe -m streamlit run ui_app.py
```

方式二：双击或执行一键脚本。

```powershell
run_ui.bat
```

启动后浏览器打开：

```text
http://localhost:8501
```

### UI 使用流程

1. 在左侧栏上传 CSV 文件，例如 `data/demo.csv`。
2. 确认左侧栏显示数据预览、行列数和列类型。
3. 在聊天框输入自然语言问题，例如：

```text
A 组和 B 组的 value 有没有显著差异?
```

4. 主区域会实时显示 Agent 文本、工具调用和工具结果。
5. 左侧栏会实时显示 `StatPlan` 当前状态，包括意图、方法、决策路径和回归诊断。
6. 分析完成后，主区域显示生成的图表。
7. 点击“下载分析报告 (Markdown)”导出报告。

回归演示可以输入：

```text
加载 case 15 的数据,做 y 关于 x 的回归
```

UI 会自动补充 `data/case15_regression_log_y.csv` 路径，并展示回归诊断到 `ols_log_y` 的修正链。

## 验证命令

以下命令已作为当前版本验收使用：

```powershell
D:\anaconda3\envs\agent\python.exe main.py
D:\anaconda3\envs\agent\python.exe tests\test_textbook_validation.py
D:\anaconda3\envs\agent\python.exe tests\test_regression_workflow.py
D:\anaconda3\envs\agent\python.exe tests\test_regression_validation.py
D:\anaconda3\envs\agent\python.exe tests\test_power_analysis.py
D:\anaconda3\envs\agent\python.exe tests\test_type_guard.py
D:\anaconda3\envs\agent\python.exe tests\test_intent_routing.py
D:\anaconda3\envs\agent\python.exe tests\test_plan_injection.py
D:\anaconda3\envs\agent\python.exe data\generate_test_cases.py
```

Streamlit UI 启动检查：

```powershell
D:\anaconda3\envs\agent\python.exe -m streamlit run ui_app.py
```

## 回归诊断示例

Case 14 用于验证“残差正态 + 异方差 -> 稳健标准误”路径：

```text
check_residual_normality:
  passed = True
  p_value = 0.21163645213093962

check_homoscedasticity / Breusch-Pagan:
  passed = False
  p_value = 1.4052212848372664e-05

select_final_model:
  final_model = ols_robust_se
  rationale = 残差正态 + 异方差,使用 White 稳健标准误
```

Case 15 用于验证“残差非正态 + 异方差 -> log(Y) 后重新拟合”路径。

Case 16 用于验证多重共线性诊断，`x1` 与 `x2` 的 VIF 大于 10，最终标记为 `ols_drop_collinear`。

## 设计原则

- LLM 负责理解问题和调度工具，不直接手算统计量。
- 统计方法选择由 `decision.py` 中的规则完成。
- 工具函数负责真实计算和写回 `StatPlan`。
- UI 只做展示层，不重新实现统计业务逻辑。
- `main.py` 命令行入口保持可独立运行。

## 已知限制

- Streamlit 每次交互会 rerun 脚本，因此 UI 在每次新问题前主动重置全局 `current_plan`，并用 `session_state` 保存展示快照。
- UI 当前是同步 Agent 调用，不是 token-level streaming。
- 图表保存在本地 `output/` 目录，未实现图表文件生命周期管理。
- 暂不支持多用户、登录、会话持久化或数据库存储。
- 本轮只支持线性回归，不支持 Logistic 回归、Ridge 或 Lasso。
