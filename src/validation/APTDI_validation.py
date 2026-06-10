print("RUN")
import pandas as pd

print("\n=== APTDI VALIDATION ===\n")

# =====================================================
# LOAD
# =====================================================
df = pd.read_csv(
    "outputs/APTDI_summary.csv"
)

print("Loaded:", df.shape)

# =====================================================
# CREATE RISK GROUPS
# =====================================================
df["risk_group"] = pd.qcut(
    df["APTDI"],
    q=3,
    labels=[
        "Low",
        "Medium",
        "High"
    ]
)

# =====================================================
# MORTALITY RATE
# =====================================================
summary = (
    df.groupby("risk_group")
    ["mortality_label"]
    .agg(
        ["count", "sum", "mean"]
    )
)

summary.columns = [
    "Patients",
    "Deaths",
    "Mortality_Rate"
]

summary["Mortality_Rate"] = (
    summary["Mortality_Rate"] * 100
)

print("\n=== MORTALITY BY APTDI GROUP ===\n")
print(summary)

# =====================================================
# SAVE
# =====================================================
summary.to_csv(
    "outputs/APTDI_validation.csv"
)

print(
    "\nSaved: outputs/APTDI_validation.csv"
)

print(
    "\nAPTDI VALIDATION COMPLETED."
)