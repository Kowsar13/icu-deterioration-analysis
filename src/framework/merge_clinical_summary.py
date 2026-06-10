import pandas as pd
import os

print("\n=== MERGING CLINICAL SUMMARY ===\n")

os.makedirs("outputs", exist_ok=True)

# =====================================================
# LOAD
# =====================================================
instability = pd.read_csv(
    "outputs/instability_summary.csv"
)

attention = pd.read_csv(
    "outputs/temporal_attention_summary.csv"
)

phenotype = pd.read_csv(
    "outputs/temporal_phenotypes.csv"
)

atdi = pd.read_csv(
    "outputs/ATDI_summary.csv"
)

# =====================================================
# MERGE
# =====================================================
merged = instability.merge(
    attention,
    on="patient_id",
    how="inner"
)

merged = merged.merge(
    phenotype,
    on="patient_id",
    how="inner"
)

merged = merged.merge(
    atdi,
    on="patient_id",
    how="inner"
)

# =====================================================
# SAVE
# =====================================================
save_path = (
    "outputs/final_clinical_summary.csv"
)

merged.to_csv(
    save_path,
    index=False
)

print("Saved:")
print(save_path)

print("\nShape:")
print(
    merged.shape
)

print("\n=== SAMPLE ===")
print(
    merged.head()
)

print(
    "\nMERGE COMPLETED."
)