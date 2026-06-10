print("start")
import pandas as pd
import numpy as np
import os

print("\n=== APTDI FRAMEWORK ===\n")

os.makedirs("outputs", exist_ok=True)

# =====================================================
# LOAD
# =====================================================
df = pd.read_csv(
    "outputs/mortality_dataset.csv"
)

print("Loaded:", df.shape)

# =====================================================
# PHENOTYPE RISK
# =====================================================
counts = (
    df["phenotype"]
    .value_counts()
)

total = len(df)

risk_map = {}

for pheno, count in counts.items():

    risk_map[pheno] = 1 - (count / total)

df["phenotype_risk"] = (
    df["phenotype"]
    .map(risk_map)
)

# =====================================================
# CLINICAL CONTEXT
# =====================================================
clinical_cols = [
    "predictedhospitalmortality",
    "predictedicumortality"
]

for col in clinical_cols:

    if col not in df.columns:

        df[col] = 0

df[clinical_cols] = (
    df[clinical_cols]
    .fillna(0)
)

df["clinical_context"] = (
    df["predictedhospitalmortality"]
    +
    df["predictedicumortality"]
) / 2

# =====================================================
# NORMALIZATION
# =====================================================
def normalize(series):

    mn = series.min()
    mx = series.max()

    if mx == mn:
        return np.zeros(len(series))

    return (
        series - mn
    ) / (
        mx - mn
    )

I = normalize(
    df["peak_instability"]
)

A = normalize(
    df["max_attention"]
)

G = normalize(
    df["graph_score"]
)

P = normalize(
    df["phenotype_risk"]
)

C = normalize(
    df["clinical_context"]
)

# =====================================================
# APTDI
# =====================================================
df["APTDI"] = (
    0.35 * I
    +
    0.20 * A
    +
    0.15 * G
    +
    0.15 * P
    +
    0.15 * C
)

# =====================================================
# SAVE
# =====================================================
save_path = (
    "outputs/APTDI_summary.csv"
)

df.to_csv(
    save_path,
    index=False
)

# =====================================================
# TOP PATIENTS
# =====================================================
top = (
    df.sort_values(
        "APTDI",
        ascending=False
    )
    .head(10)
)

print("\nSaved:")
print(save_path)

print("\n=== TOP APTDI PATIENTS ===")
print(
    top[
        [
            "patient_id",
            "APTDI",
            "mortality_label"
        ]
    ]
)

print(
    "\nAPTDI COMPLETED."
)