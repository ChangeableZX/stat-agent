from dataclasses import dataclass, field
from typing import Optional, Union


@dataclass
class StatPlan:
    research_question: str
    intent: str
    data_quality: dict = field(default_factory=dict)
    sample_sizes: dict = field(default_factory=dict)
    assumption_checks: dict = field(default_factory=dict)
    selected_method: Optional[str] = None
    method_rationale: list = field(default_factory=list)
    results: Optional[dict] = None
    plots: list = field(default_factory=list)
    interpretation: Optional[str] = None

    def _base_lines(self) -> list:
        lines = [
            "## 当前分析状态(StatPlan 快照)",
            "",
            f"**研究问题**: {self.research_question}",
            f"**意图**: {self.intent}",
        ]
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
        return lines

    def summary(self) -> str:
        return "\n".join(self._base_lines())

    def to_dict(self) -> dict:
        return {
            "plan_type": type(self).__name__,
            "research_question": self.research_question,
            "intent": self.intent,
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


@dataclass
class TwoGroupPlan(StatPlan):
    target_variable: str = ""
    grouping_variable: Optional[str] = None
    paired_columns: Optional[tuple] = None
    design: str = "independent"

    def summary(self) -> str:
        lines = self._base_lines()
        insert = [
            f"**目标变量**: {self.target_variable}",
            f"**实验设计**: {self.design}",
        ]
        if self.design == "paired" and self.paired_columns:
            insert.insert(1, f"**配对列**: {self.paired_columns}")
        elif self.grouping_variable:
            insert.insert(1, f"**分组变量**: {self.grouping_variable}")
        return "\n".join(lines[:4] + insert + lines[4:])

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update({
            "target_variable": self.target_variable,
            "grouping_variable": self.grouping_variable,
            "paired_columns": list(self.paired_columns) if self.paired_columns else None,
            "design": self.design,
            "summary": self.summary(),
        })
        return data


@dataclass
class MultiGroupPlan(StatPlan):
    target_variable: str = ""
    grouping_variable: str = ""
    group_levels: list = field(default_factory=list)
    design: str = "independent"
    posthoc_needed: bool = False
    posthoc_method: Optional[str] = None

    def summary(self) -> str:
        lines = self._base_lines()
        insert = [
            f"**目标变量**: {self.target_variable}",
            f"**分组变量**: {self.grouping_variable}",
            f"**实验设计**: {self.design}",
        ]
        if self.group_levels:
            insert.append(f"**组水平**: {self.group_levels}")
        if self.posthoc_needed:
            insert.append(f"**需要事后检验**: {self.posthoc_needed}")
        if self.posthoc_method:
            insert.append(f"**事后检验方法**: {self.posthoc_method}")
        return "\n".join(lines[:4] + insert + lines[4:])

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update({
            "target_variable": self.target_variable,
            "grouping_variable": self.grouping_variable,
            "group_levels": self.group_levels,
            "design": self.design,
            "posthoc_needed": self.posthoc_needed,
            "posthoc_method": self.posthoc_method,
            "summary": self.summary(),
        })
        return data


@dataclass
class CorrelationPlan(StatPlan):
    x_variable: str = ""
    y_variable: str = ""
    x_type: str = "continuous"
    y_type: str = "continuous"

    def summary(self) -> str:
        lines = self._base_lines()
        insert = [
            f"**X 变量**: {self.x_variable}",
            f"**Y 变量**: {self.y_variable}",
            f"**X 类型**: {self.x_type}",
            f"**Y 类型**: {self.y_type}",
        ]
        return "\n".join(lines[:4] + insert + lines[4:])

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update({
            "x_variable": self.x_variable,
            "y_variable": self.y_variable,
            "x_type": self.x_type,
            "y_type": self.y_type,
            "summary": self.summary(),
        })
        return data


@dataclass
class CategoricalPlan(StatPlan):
    row_variable: str = ""
    col_variable: str = ""
    table_shape: tuple = (0, 0)
    paired: bool = False

    def summary(self) -> str:
        lines = self._base_lines()
        insert = [
            f"**行变量**: {self.row_variable}",
            f"**列变量**: {self.col_variable}",
            f"**列联表形状**: {self.table_shape}",
            f"**是否配对**: {self.paired}",
        ]
        return "\n".join(lines[:4] + insert + lines[4:])

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update({
            "row_variable": self.row_variable,
            "col_variable": self.col_variable,
            "table_shape": list(self.table_shape),
            "paired": self.paired,
            "summary": self.summary(),
        })
        return data


PlanType = Union[TwoGroupPlan, MultiGroupPlan, CorrelationPlan, CategoricalPlan]
current_plan: Optional[PlanType] = None


def set_current_plan(plan: PlanType) -> PlanType:
    global current_plan
    current_plan = plan
    return plan


def get_current_plan() -> Optional[PlanType]:
    return current_plan
