from state import CategoricalPlan, CorrelationPlan, MultiGroupPlan, RegressionPlan, StatPlan, TwoGroupPlan


def select_method(plan: StatPlan) -> StatPlan:
    if isinstance(plan, TwoGroupPlan):
        return select_method_for_two_group(plan)
    if isinstance(plan, MultiGroupPlan):
        return select_method_for_multi_group(plan)
    if isinstance(plan, CorrelationPlan):
        return select_method_for_correlation(plan)
    if isinstance(plan, CategoricalPlan):
        return select_method_for_categorical(plan)
    if isinstance(plan, RegressionPlan):
        return select_method_for_regression(plan)
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


def select_method_for_regression(plan: RegressionPlan) -> RegressionPlan:
    if plan.intent != "regression":
        raise ValueError(f"Unsupported intent: {plan.intent}")
    if plan.regression_type != "linear":
        raise ValueError("Only linear regression is supported in this round.")
    if not plan.x_variables:
        raise ValueError("Regression requires at least one x variable.")

    if len(plan.x_variables) == 1:
        plan.selected_method = "simple_linear_regression"
        plan.method_rationale = ["linear regression + one predictor", "-> simple_linear_regression"]
    else:
        plan.selected_method = "multiple_linear_regression"
        plan.method_rationale = ["linear regression + multiple predictors", "-> multiple_linear_regression"]
    return plan


def select_final_model(plan: RegressionPlan) -> RegressionPlan:
    if not isinstance(plan, RegressionPlan):
        raise TypeError(f"select_final_model requires RegressionPlan, got {type(plan).__name__}")

    diagnostics = plan.diagnostics or {}
    normality = diagnostics.get("residual_normality", {})
    homoscedasticity = diagnostics.get("homoscedasticity", {})
    multicollinearity = diagnostics.get("multicollinearity", {})
    independence = diagnostics.get("independence", {})
    outliers = diagnostics.get("outliers", {})

    normal_passed = normality.get("passed", True)
    homo_passed = homoscedasticity.get("passed", True)
    multi_passed = multicollinearity.get("passed", True)
    independent_passed = independence.get("passed", True)
    outlier_passed = outliers.get("passed", True)
    max_vif = float(multicollinearity.get("max_vif", 0) or 0)

    failed = [
        not normal_passed,
        not homo_passed,
        not multi_passed,
        not independent_passed,
        not outlier_passed,
    ]

    plan.transformations = []
    if max_vif > 10 or not multi_passed:
        plan.final_model = "ols_drop_collinear"
        plan.method_rationale = ["存在多重共线性,建议删除高 VIF 变量"]
    elif not normal_passed and not homo_passed:
        plan.final_model = "ols_log_y"
        plan.transformations.append({"type": "log_y", "reason": "残差非正态+异方差"})
        plan.method_rationale = ["残差非正态+异方差,对 Y 取对数后重新拟合"]
    elif normal_passed and not homo_passed:
        plan.final_model = "ols_robust_se"
        plan.method_rationale = ["残差正态 + 异方差,使用 White 稳健标准误"]
    elif sum(bool(item) for item in failed) >= 3:
        plan.final_model = "flag_for_glm"
        plan.method_rationale = ["严重违反 OLS 假设,建议改用 GLM"]
    elif not any(failed):
        plan.final_model = "ols"
        plan.method_rationale = ["诊断全通过,采用 OLS"]
    else:
        plan.final_model = "ols"
        plan.method_rationale = ["存在轻微诊断提醒,当前仍采用 OLS 并在解释中提示"]
    return plan
