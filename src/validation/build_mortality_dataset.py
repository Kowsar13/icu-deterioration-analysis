print("Start")
import pandas as pd
import os

print("\n=== BUILDING MORTALITY DATASET ===\n")

os.makedirs("outputs", exist_ok=True)

# =====================================================
# LOAD OUR SUMMARY
# =====================================================
summary = pd.read_csv(
    "outputs/final_clinical_summary.csv"
)

print("Summary shape:", summary.shape)

# =====================================================
# LOAD PATIENT TABLE
# =====================================================
patient = pd.read_csv(
    "data/eicu-database2.0.1/patient.csv.gz",
    compression="gzip"
)

patient = patient[[
    "patientunitstayid",
    "hospitaldischargestatus",
    "unitdischargestatus"
]]

patient = patient.rename(columns={
    "patientunitstayid": "patient_id"
})

# =====================================================
# LOAD APACHE RESULTS
# =====================================================
apache = pd.read_csv(
    "data/eicu-database2.0.1/apachePatientResult.csv.gz",
    compression="gzip"
)

apache = apache[[
    "patientunitstayid",
    "actualhospitalmortality",
    "actualicumortality",
    "predictedhospitalmortality",
    "predictedicumortality"
]]

apache = apache.rename(columns={
    "patientunitstayid": "patient_id"
})

# =====================================================
# REMOVE DUPLICATES
# =====================================================
apache = apache.drop_duplicates(
    subset=["patient_id"]
)

patient = patient.drop_duplicates(
    subset=["patient_id"]
)

# =====================================================
# MERGE
# =====================================================
merged = summary.merge(
    patient,
    on="patient_id",
    how="left"
)

merged = merged.merge(
    apache,
    on="patient_id",
    how="left"
)

# =====================================================
# CREATE BINARY LABEL
# =====================================================
merged["mortality_label"] = (
    merged["hospitaldischargestatus"]
    .fillna("")
    .str.lower()
    .eq("expired")
).astype(int)

# =====================================================
# SAVE
# =====================================================
save_path = (
    "outputs/mortality_dataset.csv"
)

merged.to_csv(
    save_path,
    index=False
)

print("\nSaved:")
print(save_path)

print("\nShape:")
print(merged.shape)

print("\nMortality counts:")
print(
    merged["mortality_label"]
    .value_counts(dropna=False)
)

print("\nSample:")
print(
    merged.head()
)

print(
    "\nMORTALITY DATASET COMPLETED."
)