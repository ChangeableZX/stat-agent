import os

import pandas as pd


DATA_DIR = os.path.dirname(__file__)


def write_plantgrowth():
    plantgrowth = {
        "group": ["ctrl"] * 10 + ["trt1"] * 10 + ["trt2"] * 10,
        "weight": [
            4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14,
            4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69,
            6.31, 5.12, 5.54, 5.50, 5.37, 5.29, 4.92, 6.15, 5.80, 5.26,
        ],
    }
    pd.DataFrame(plantgrowth).to_csv(os.path.join(DATA_DIR, "textbook_anova_plantgrowth.csv"), index=False)


def write_anscombe():
    anscombe1 = {
        "x": [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
        "y": [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68],
    }
    pd.DataFrame(anscombe1).to_csv(os.path.join(DATA_DIR, "textbook_corr_anscombe.csv"), index=False)


def write_titanic():
    rows = []
    rows += [{"class": "1st", "survived": "yes"}] * 203
    rows += [{"class": "1st", "survived": "no"}] * 122
    rows += [{"class": "3rd", "survived": "yes"}] * 178
    rows += [{"class": "3rd", "survived": "no"}] * 528
    pd.DataFrame(rows).to_csv(os.path.join(DATA_DIR, "textbook_chi2_titanic.csv"), index=False)


def write_tea():
    rows = []
    rows += [{"actual": "milk_first", "guess": "milk_first"}] * 3
    rows += [{"actual": "milk_first", "guess": "tea_first"}] * 1
    rows += [{"actual": "tea_first", "guess": "milk_first"}] * 1
    rows += [{"actual": "tea_first", "guess": "tea_first"}] * 3
    pd.DataFrame(rows).to_csv(os.path.join(DATA_DIR, "textbook_fisher_tea.csv"), index=False)


def main():
    write_plantgrowth()
    write_anscombe()
    write_titanic()
    write_tea()
    print("Generated textbook validation datasets:")
    for name in [
        "textbook_anova_plantgrowth.csv",
        "textbook_corr_anscombe.csv",
        "textbook_chi2_titanic.csv",
        "textbook_fisher_tea.csv",
    ]:
        print(f"- data/{name}")


if __name__ == "__main__":
    main()
