# stat-agent

一个基于 LLM 的统计推断 Agent Day 1 最小骨架：读取 CSV，检查两组正态性，并完成独立样本 t 检验。

## 安装

```bash
pip install -r requirements.txt
```

设置 API Key：

打开 `.env` 文件，填写或调整大模型接口配置：

```env
ANTHROPIC_API_KEY=your_api_key_here
ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
ANTHROPIC_MODEL=deepseek-v4-flash
ANTHROPIC_MAX_TOKENS=2048
```

## 运行

```bash
python main.py
```

## 当前功能

Day 1 支持两组数值列比较的端到端流程：加载 CSV、分别做 Shapiro-Wilk 正态性检验、运行独立样本 t 检验，并用中文解释结果。

## 后续计划

后续会扩展更多统计检验方法、变量类型识别和更完整的状态管理。
