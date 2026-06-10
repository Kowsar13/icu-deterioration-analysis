print("Start... ... ...")
import pandas as pd

print("\n=== APACHE VS APTDI ===\n")

df = pd.read_csv(
    "outputs/APTDI_summary.csv"
)

# ==========================
# APTDI GROUPS
# ==========================
df["APTDI_group"] = pd.qcut(
    df["APTDI"],
    q=3,
    labels=["Low","Medium","High"]
)

aptdi = (
    df.groupby("APTDI_group")
    ["mortality_label"]
    .mean()
    * 100
)

print("\nAPTDI Mortality (%)")
print(aptdi)

# ==========================
# APACHE GROUPS
# ==========================
df["APACHE_group"] = pd.qcut(
    df["predictedhospitalmortality"],
    q=3,
    labels=["Low","Medium","High"]
)

apache = (
    df.groupby("APACHE_group")
    ["mortality_label"]
    .mean()
    * 100
)

print("\nAPACHE Mortality (%)")
print(apache)

# ==========================
# SAVE
# ==========================
comparison = pd.DataFrame({
    "APTDI": aptdi,
    "APACHE": apache
})

comparison.to_csv(
    "outputs/APACHE_vs_APTDI.csv"
)

print("\nSaved:")
print("outputs/APACHE_vs_APTDI.csv")

print("\nCOMPARISON COMPLETED.")