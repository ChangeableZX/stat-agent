import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from tools import load_data, make_analysis_plan, run_simple_linear_regression, select_method


MTCARS_MPG = [
    21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4,
    22.8, 19.2, 17.8, 16.4, 17.3, 15.2, 10.4, 10.4,
    14.7, 32.4, 30.4, 33.9, 21.5, 15.5, 15.2, 13.3,
    19.2, 27.3, 26.0, 30.4, 15.8, 19.7, 15.0, 21.4,
]

MTCARS_WT = [
    2.620, 2.875, 2.320, 3.215, 3.440, 3.460, 3.570, 3.190,
    3.150, 3.440, 3.440, 4.070, 3.730, 3.780, 5.250, 5.424,
    5.345, 2.200, 1.615, 1.835, 2.465, 3.520, 3.435, 3.840,
    3.845, 1.935, 2.140, 1.513, 3.170, 2.770, 3.570, 2.780,
]


def test_mtcars_mpg_wt_regression_matches_known_answer():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "mtcars.csv")
        pd.DataFrame({"mpg": MTCARS_MPG, "wt": MTCARS_WT}).to_csv(path, index=False)

        plan = make_analysis_plan("mtcars mpg ~ wt", "regression", y_variable="mpg", x_variables=["wt"])
        assert plan["plan_type"] == "RegressionPlan"
        load_data(path)
        selected = select_method()
        assert selected["selected_method"] == "simple_linear_regression"
        result = run_simple_linear_regression("mpg", "wt")

    assert np.isclose(result["coefficients"]["intercept"], 37.285, atol=0.001)
    assert np.isclose(result["coefficients"]["wt"], -5.344, atol=0.001)
    assert np.isclose(result["r_squared"], 0.7528, atol=0.0001)
    assert result["p_values"]["wt"] < 0.001


def main():
    test_mtcars_mpg_wt_regression_matches_known_answer()
    print("PASS")


if __name__ == "__main__":
    main()
