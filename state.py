from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class StatPlan:
    research_question: str
    intent: str
    target_variable: str
    grouping_variable: Optional[str] = None
    paired_columns: Optional[tuple] = None
    design: str = "independent"
    data_quality: dict = field(default_factory=dict)
    sample_sizes: dict = field(default_factory=dict)
    assumption_checks: dict = field(default_factory=dict)
    selected_method: Optional[str] = None
    method_rationale: list = field(default_factory=list)
    results: Optional[dict] = None
    plots: list = field(default_factory=list)
    interpretation: Optional[str] = None

    def summary(self) -> str:
        lines = [
            f"研究问题: {self.research_question}",
            f"意图: {self.intent}",
            f"设计: {self.design}",
            f"目标变量: {self.target_variable}",
            f"分组变量: {self.grouping_variable}",
            f"配对列: {self.paired_columns}",
            f"数据质量: {self.data_quality}",
            f"样本量: {self.sample_sizes}",
            f"前提检验: {self.assumption_checks}",
            f"选择方法: {self.selected_method}",
            f"决策路径: {self.method_rationale}",
            f"结果: {self.results}",
            f"图: {self.plots}",
        ]
        if self.interpretation:
            lines.append(f"解释: {self.interpretation}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "research_question": self.research_question,
            "intent": self.intent,
            "target_variable": self.target_variable,
            "grouping_variable": self.grouping_variable,
            "paired_columns": list(self.paired_columns) if self.paired_columns else None,
            "design": self.design,
            "data_quality": self.data_quality,
            "sample_sizes": self.sample_sizes,
            "assumption_checks": self.assumption_checks,
            "selected_method": self.selected_method,
            "method_rationale": self.method_rationale,
            "results": self.results,
            "plots": self.plots,
            "interpretation": self.interpretation,
            "summary": self.summary(),
        }


current_plan: Optional[StatPlan] = None


def set_current_plan(plan: StatPlan) -> StatPlan:
    global current_plan
    current_plan = plan
    return plan


def get_current_plan() -> Optional[StatPlan]:
    return current_plan
