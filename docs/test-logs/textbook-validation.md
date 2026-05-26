# 教科书数据集统计正确性验证

**日期**: 2026-05-26 11:13:22
**Git commit**: 00bc680539a9560a994c8ce1a7f03c177db30e41
**验证依据**: R 内置数据集与标准统计教科书

## 验证结果汇总

| Case | 数据集 | 期望方法 | 实际方法 | 期望值 | 实际值 | 一致? |
|---|---|---|---|---|---|---|
| 1.1 | PlantGrowth | one_way_anova | one_way_anova | F=4.846, p=0.01591, df=(2,27) | F=4.84609, p=0.01591, df=(2,27) | ✓ |
| 1.2 | PlantGrowth Tukey | tukey_hsd | tukey_hsd | trt2-trt1 diff=0.865, p_adj=0.012 | trt2-trt1 diff=0.8650000000000002, p_adj=0.012006423979493142 | ✓ |
| 2 | Anscombe quartet I | pearson | pearson | r=0.8164, p=0.00217, CI≈[0.424,0.951] | r=0.816421, p=0.00216963, CI=[0.424,0.951] | ✓ |
| 3 | Titanic 1st vs 3rd | chi_square_yates | chi_square_yates | χ²=130.94, p≈2.55e-30, df=1 | χ²=130.944, p=2.5473e-30, df=1 | ✓ |
| 4 | Fisher tea | fisher_exact | fisher_exact | OR=9.0, p=0.4857, CI≈[0.21,626.2] | OR=9, p=0.485714 | ✓ |

## 详细输出

### Case 1.1 - PlantGrowth

- 通过: 是
- 期望方法: `one_way_anova`
- 实际方法: `one_way_anova`
- 期望值: F=4.846, p=0.01591, df=(2,27)
- 实际值: F=4.84609, p=0.01591, df=(2,27)

```text
{'anova': {'method': 'one_way_anova', 'statistic': 4.846087862380136, 'p_value': 0.015909958325622895, 'effect_size': {'name': 'eta_squared', 'value': 0.26414829683211977, 'magnitude': 'large'}, 'group_stats': {'ctrl': {'n': 10, 'mean': 5.031999999999999, 'std': 0.5830913783924057, 'median': 5.154999999999999}, 'trt1': {'n': 10, 'mean': 4.661, 'std': 0.7936756964347033, 'median': 4.550000000000001}, 'trt2': {'n': 10, 'mean': 5.526, 'std': 0.44257328332278606, 'median': 5.4350000000000005}}, 'interpretation_hints': {'significant': True, 'direction': '某些组之间有差异,需事后检验', 'practical_caveat': None}}}
```

### Case 1.2 - PlantGrowth Tukey

- 通过: 是
- 期望方法: `tukey_hsd`
- 实际方法: `tukey_hsd`
- 期望值: trt2-trt1 diff=0.865, p_adj=0.012
- 实际值: trt2-trt1 diff=0.8650000000000002, p_adj=0.012006423979493142

```text
{'tukey': {'method': 'tukey_hsd', 'comparisons': [{'A': 'ctrl', 'B': 'trt1', 'mean_A': 5.032, 'mean_B': 4.661, 'diff': 0.37100000000000044, 'se': 0.27878160840554955, 'T': 1.3307908011646838, 'p_tukey': 0.39087114420210534, 'hedges': 0.5102373664920888}, {'A': 'ctrl', 'B': 'trt2', 'mean_A': 5.032, 'mean_B': 5.526, 'diff': -0.4939999999999998, 'se': 0.27878160840554955, 'T': -1.7719963767529723, 'p_tukey': 0.1979959912995718, 'hedges': -0.9140377642312715}, {'A': 'trt1', 'B': 'trt2', 'mean_A': 4.661, 'mean_B': 5.526, 'diff': -0.8650000000000002, 'se': 0.27878160840554955, 'T': -3.102787177917656, 'p_tukey': 0.012006423979493142, 'hedges': -1.2892771189382815}]}}
```

### Case 2 - Anscombe quartet I

- 通过: 是
- 期望方法: `pearson`
- 实际方法: `pearson`
- 期望值: r=0.8164, p=0.00217, CI≈[0.424,0.951]
- 实际值: r=0.816421, p=0.00216963, CI=[0.424,0.951]

```text
{'pearson': {'method': 'pearson', 'statistic': 0.8164205163448396, 'p_value': 0.002169628873078806, 'effect_size': {'name': 'pearson_r', 'value': 0.8164205163448396, 'magnitude': 'large'}, 'group_stats': {'n': 11, 'x': {'n': 11, 'mean': 9.0, 'std': 3.3166247903554, 'median': 9.0}, 'y': {'n': 11, 'mean': 7.500909090909093, 'std': 2.031568135925815, 'median': 7.58}}, 'interpretation_hints': {'significant': True, 'direction': '变量之间存在相关关系', 'practical_caveat': None}}, 'ci_95': [0.4243912133932155, 0.9506932537865604]}
```

### Case 3 - Titanic 1st vs 3rd

- 通过: 是
- 期望方法: `chi_square_yates`
- 实际方法: `chi_square_yates`
- 期望值: χ²=130.94, p≈2.55e-30, df=1
- 实际值: χ²=130.944, p=2.5473e-30, df=1

```text
{'expected_frequencies': {'table_shape': [2, 2], 'expected': [[204.89815712900096, 120.10184287099904], [445.101842870999, 260.898157129001]], 'all_expected_ge_5': True, 'low_frequency_ratio': 0.0, 'chi2_preview': 132.53767675894528, 'p_value_preview': 1.1412017148838804e-30, 'dof': 1}, 'chi_square_yates': {'method': 'chi_square_yates', 'statistic': 130.9436970973435, 'p_value': 2.5472972407656084e-30, 'effect_size': {'name': 'cramers_v', 'value': 0.35637970723614737, 'magnitude': 'medium'}, 'group_stats': {'observed': {'no': {'1st': 122, '3rd': 528}, 'yes': {'1st': 203, '3rd': 178}}, 'expected': [[204.89815712900096, 120.10184287099904], [445.101842870999, 260.898157129001]]}, 'interpretation_hints': {'significant': True, 'direction': '分类变量之间有关联', 'practical_caveat': None}}}
```

### Case 4 - Fisher tea

- 通过: 是
- 期望方法: `fisher_exact`
- 实际方法: `fisher_exact`
- 期望值: OR=9.0, p=0.4857, CI≈[0.21,626.2]
- 实际值: OR=9, p=0.485714

```text
{'fisher_exact': {'method': 'fisher_exact', 'statistic': 9.0, 'p_value': 0.48571428571428565, 'effect_size': {'name': 'odds_ratio', 'value': 9.0, 'magnitude': 'not_classified'}, 'group_stats': {'observed': {'milk_first': {'milk_first': 3, 'tea_first': 1}, 'tea_first': {'milk_first': 1, 'tea_first': 3}}}, 'interpretation_hints': {'significant': False, 'direction': 'no_diff', 'practical_caveat': None}}}
```
