from state import StatPlan


def select_method_for_two_group(plan: StatPlan) -> StatPlan:
    if plan.intent != "compare_two_groups":
        raise ValueError(f"Unsupported intent: {plan.intent}")

    checks = plan.assumption_checks
    rationale = []

    if plan.design == "paired":
        diff = checks.get("normality_diff", {})
        if diff.get("passed"):
            plan.selected_method = "paired_t"
            rationale = ["差值正态,配对设计", "→ paired_t"]
        else:
            plan.selected_method = "wilcoxon"
            rationale = ["差值非正态,改用非参", "→ wilcoxon"]
        plan.method_rationale = rationale
        return plan

    if plan.design != "independent":
        raise ValueError(f"Unsupported design: {plan.design}")

    normality = [v for k, v in checks.items() if k.startswith("normality_")]
    if len(normality) < 2:
        raise ValueError("Need normality checks for both groups before method selection.")

    both_normal = all(item.get("passed") for item in normality[:2])
    min_n = min(plan.sample_sizes.values()) if plan.sample_sizes else 0
    variance = checks.get("variance_equality", {})

    if both_normal:
        if variance.get("equal_variance"):
            plan.selected_method = "student_t"
            rationale = ["双正态+方差齐", "→ Student's t"]
        else:
            plan.selected_method = "welch_t"
            rationale = ["双正态+方差不齐", "→ Welch's t"]
    elif min_n >= 30:
        plan.selected_method = "welch_t"
        rationale = ["非正态但大样本,CLT 兜底", "→ Welch's t"]
    else:
        plan.selected_method = "mannwhitney"
        rationale = ["非正态+小样本,改用非参", "→ Mann-Whitney U"]

    plan.method_rationale = rationale
    return plan
