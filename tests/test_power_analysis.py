import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from tools import power_analysis_anova, power_analysis_correlation, power_analysis_ttest


def test_ttest_power_solves_sample_size_per_group():
    result = power_analysis_ttest(effect_size=0.5, alpha=0.05, power=0.8, n=None)
    assert result["computed"] == "n"
    assert np.isclose(result["value"], 63.77, atol=0.2)


def test_ttest_power_solves_power_from_n_per_group():
    result = power_analysis_ttest(effect_size=0.5, alpha=0.05, power=None, n=30)
    assert result["computed"] == "power"
    assert np.isclose(result["value"], 0.48, atol=0.02)


def test_ttest_power_rejects_over_specified_inputs():
    result = power_analysis_ttest(effect_size=0.5, alpha=0.05, power=0.8, n=30)
    assert "error" in result
    assert "exactly one" in result["error"]


def test_ttest_power_handles_tiny_effect_large_power():
    result = power_analysis_ttest(effect_size=0.01, alpha=0.05, power=0.99, n=None)
    assert result["computed"] == "n"
    assert result["value"] > 100000


def test_anova_power_solves_sample_size_per_group():
    result = power_analysis_anova(effect_size=0.25, n_groups=3, alpha=0.05, power=0.8, n=None)
    assert result["computed"] == "n"
    assert np.isclose(result["value"], 52.40, atol=0.2)


def test_correlation_power_solves_sample_size_and_power():
    n_result = power_analysis_correlation(r=0.3, alpha=0.05, power=0.8, n=None)
    power_result = power_analysis_correlation(r=0.3, alpha=0.05, power=None, n=85)
    assert n_result["computed"] == "n"
    assert np.isclose(n_result["value"], 84.93, atol=0.2)
    assert power_result["computed"] == "power"
    assert np.isclose(power_result["value"], 0.80, atol=0.01)


def main():
    test_ttest_power_solves_sample_size_per_group()
    test_ttest_power_solves_power_from_n_per_group()
    test_ttest_power_rejects_over_specified_inputs()
    test_ttest_power_handles_tiny_effect_large_power()
    test_anova_power_solves_sample_size_per_group()
    test_correlation_power_solves_sample_size_and_power()
    print("PASS")


if __name__ == "__main__":
    main()
