from dataclasses import dataclass, field
from typing import Optional


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
            "## 当前分析状态(StatPlan 快照)",
            "",
            f"**研究问题**: {self.research_question}",
            f"**意图**: {self.intent}",
            f"**目标变量**: {self.target_variable}",
        ]
        if self.design == "paired":
            lines.append(f"**配对列**: {self.paired_columns}")
        elif self.grouping_variable:
            lines.append(f"**分组变量**: {self.grouping_variable}")
        lines.append(f"**实验设计**: {self.design}")
        if self.data_quality:
            lines.extend(["", f"**数据质量**: {self.data_quality}"])
        if self.sample_sizes:
            lines.extend(["", f"**样本量**: {self.sample_sizes}"])
        if self.assumption_checks:
            lines.extend(["", "**前提检验**:"])
            for name, value in self.assumption_checks.items():
                lines.append(f"- {name}: {value}")
        if self.selected_method:
            lines.extend(["", f"**选定方法**: {self.selected_method}"])
        if self.method_rationale:
            lines.extend(["", "**决策路径**:"])
            for item in self.method_rationale:
                lines.append(f"- {item}")
        if self.results:
            lines.extend(["", f"**统计结果**: {self.results}"])
        if self.plots:
            lines.extend(["", f"**已生成图表**: {self.plots}"])
        if self.interpretation:
            lines.extend(["", f"**解释**: {self.interpretation}"])
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
