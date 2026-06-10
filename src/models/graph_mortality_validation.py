import pandas as pd
from scipy.stats import mannwhitneyu

print("\n=== GRAPH MORTALITY VALIDATION ===\n")

df = pd.read_csv(
    "outputs/tgnn_dataset.csv"
)

features = [
    "mean_HR_RR",
    "std_HR_RR",
    "mean_HR_SpO2",
    "std_HR_SpO2",
    "mean_RR_SpO2",
    "std_RR_SpO2",
    "num_windows"
]

alive = df[
    df["mortality_label"] == 0
]

dead = df[
    df["mortality_label"] == 1
]

for feature in features:

    stat, p = mannwhitneyu(
        alive[feature],
        dead[feature],
        alternative="two-sided"
    )

    print("\n", feature)

    print(
        "Alive Mean:",
        round(alive[feature].mean(),4)
    )

    print(
        "Dead Mean:",
        round(dead[feature].mean(),4)
    )

    print(
        "p-value:",
        p
    )

print(
    "\nGRAPH VALIDATION COMPLETED."
)