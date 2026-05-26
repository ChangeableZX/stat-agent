from state import CategoricalPlan, CorrelationPlan, MultiGroupPlan, StatPlan, TwoGroupPlan


def select_method(plan: StatPlan) -> StatPlan:
    if isinstance(plan, TwoGroupPlan):
        return select_method_for_two_group(plan)
    if isinstance(plan, MultiGroupPlan):
        return select_method_for_multi_group(plan)
    if isinstance(plan, CorrelationPlan):
        return select_method_for_correlation(plan)
    if isinstance(plan, CategoricalPlan):
        return select_method_for_categorical(plan)
    raise TypeError(f"Unknown plan type: {type(plan)}")


def select_method_for_two_group(plan: TwoGroupPlan) -> TwoGroupPlan:
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


def select_method_for_multi_group(plan: MultiGroupPlan) -> MultiGroupPlan:
    checks = plan.assumption_checks
    if plan.design == "repeated_measures":
        residual = checks.get("residual_normality", {})
        sphericity = checks.get("sphericity", {})
        if residual.get("passed") and sphericity.get("passed"):
            plan.selected_method = "repeated_anova"
            plan.posthoc_method = "tukey_hsd"
            rationale = ["残差正态+球形假设满足", "→ repeated_anova"]
        elif residual.get("passed"):
            plan.selected_method = "repeated_anova_gg"
            plan.posthoc_method = "tukey_hsd"
            rationale = ["残差正态+球形违反,GG 校正", "→ repeated_anova_gg"]
        else:
            plan.selected_method = "friedman"
            plan.posthoc_method = "wilcoxon_bh"
            rationale = ["残差非正态,Friedman", "→ friedman"]
        plan.posthoc_needed = True
        plan.method_rationale = rationale
        return plan

    normality = [v for k, v in checks.items() if k.startswith("normality_")]
    all_normal = bool(normality) and all(item.get("passed") for item in normality)
    variance = checks.get("variance_equality", {})
    if all_normal:
        if variance.get("equal_variance"):
            plan.selected_method = "one_way_anova"
            plan.posthoc_method = "tukey_hsd"
            rationale = ["各组正态+方差齐", "→ one_way_anova"]
        else:
            plan.selected_method = "welch_anova"
            plan.posthoc_method = "games_howell"
            rationale = ["各组正态+方差不齐", "→ welch_anova"]
    else:
        plan.selected_method = "kruskal_wallis"
        plan.posthoc_method = "dunn_bh"
        rationale = ["至少一组非正态", "→ kruskal_wallis"]
    plan.posthoc_needed = True
    plan.method_rationale = rationale
    return plan


def select_method_for_correlation(plan: CorrelationPlan) -> CorrelationPlan:
    if plan.x_type == "continuous" and plan.y_type == "continuous":
        normal = plan.assumption_checks.get("bivariate_normality", {}).get("passed")
        linear = plan.assumption_checks.get("linearity", {}).get("passed")
        if normal and linear:
            plan.selected_method = "pearson"
            plan.method_rationale = ["双正态+线性", "→ pearson"]
        else:
            plan.selected_method = "spearman"
            plan.method_rationale = ["非正态或非线性", "→ spearman"]
    elif plan.x_type == "continuous" and plan.y_type == "ordinal":
        plan.selected_method = "spearman"
        plan.method_rationale = ["连续 × 有序", "→ spearman"]
    elif plan.x_type == "ordinal" and plan.y_type == "ordinal":
        plan.selected_method = "kendall"
        plan.method_rationale = ["双有序变量", "→ kendall"]
    elif plan.x_type == "ordinal" and plan.y_type == "continuous":
        plan.selected_method = "spearman"
        plan.method_rationale = ["有序 × 连续", "→ spearman"]
    else:
        raise ValueError(f"Unsupported variable types: {plan.x_type}, {plan.y_type}")
    return plan


def select_method_for_categorical(plan: CategoricalPlan) -> CategoricalPlan:
    expected = plan.assumption_checks.get("expected_frequencies", {})
    shape = tuple(plan.table_shape)
    low_ratio = expected.get("low_frequency_ratio", 0.0)
    all_ge_5 = expected.get("all_expected_ge_5", False)

    if plan.paired:
        if shape == (2, 2):
            plan.selected_method = "mcnemar"
            plan.method_rationale = ["配对 2x2", "→ mcnemar"]
        else:
            plan.selected_method = "mcnemar_bowker"
            plan.method_rationale = ["配对多分类", "→ mcnemar_bowker"]
    elif shape == (2, 2):
        if all_ge_5:
            plan.selected_method = "chi_square_yates"
            plan.method_rationale = ["2x2 + 期望频数充足,Yates 校正", "→ chi_square_yates"]
        else:
            plan.selected_method = "fisher_exact"
            plan.method_rationale = ["2x2 + 期望频数不足,Fisher", "→ fisher_exact"]
    elif low_ratio < 0.2:
        plan.selected_method = "chi_square"
        plan.method_rationale = ["R×C + < 20% 格子低频", "→ chi_square"]
    else:
        plan.selected_method = "fisher_freeman_halton"
        plan.method_rationale = ["R×C + ≥ 20% 格子低频", "→ fisher_freeman_halton"]
    return plan
