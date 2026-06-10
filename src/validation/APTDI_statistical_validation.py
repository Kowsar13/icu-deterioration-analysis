import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import mannwhitneyu

print("\n=== APTDI STATISTICAL VALIDATION ===\n")

df = pd.read_csv(
    "outputs/APTDI_summary.csv"
)

# ====================================
# CORRELATION
# ====================================
corr, p = spearmanr(
    df["APTDI"],
    df["mortality_label"]
)

print("\nSpearman Correlation")
print("Correlation:", round(corr,4))
print("p-value:", p)

# ====================================
# SURVIVED VS EXPIRED
# ====================================
alive = df[
    df["mortality_label"] == 0
]["APTDI"]

dead = df[
    df["mortality_label"] == 1
]["APTDI"]

stat, p2 = mannwhitneyu(
    alive,
    dead,
    alternative="two-sided"
)

print("\nMann-Whitney U Test")
print("Statistic:", stat)
print("p-value:", p2)

print("\nMean APTDI")

print(
    "Alive:",
    round(alive.mean(),4)
)

print(
    "Expired:",
    round(dead.mean(),4)
)

print("\nVALIDATION COMPLETED.")