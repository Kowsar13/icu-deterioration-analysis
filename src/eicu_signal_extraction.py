import pandas as pd
import os

print("\n=== REAL ICU SIGNAL EXTRACTION ===\n")

# =====================================================
# CREATE OUTPUT FOLDER
# =====================================================
os.makedirs("outputs", exist_ok=True)

# =====================================================
# LOAD VITAL PERIODIC TABLE
# =====================================================
file_path = "data/eicu-database2.0.1/vitalPeriodic.csv.gz"

print("Loading vitalPeriodic table...")

vitals = pd.read_csv(
    file_path,
    compression="gzip"
)

print("Original shape:", vitals.shape)

# =====================================================
# SELECT IMPORTANT COLUMNS
# =====================================================
selected_columns = [
    "patientunitstayid",
    "observationoffset",
    "heartrate",
    "respiration",
    "sao2",
    "temperature",
    "systemicsystolic",
    "systemicdiastolic",
    "systemicmean"
]

vitals = vitals[selected_columns]

print("\nSelected columns loaded.")

# =====================================================
# REMOVE ROWS WITH CRITICAL MISSING VALUES
# =====================================================
print("\nCleaning missing values...")

vitals = vitals.dropna(
    subset=[
        "heartrate",
        "respiration"
    ]
)

print("Shape after cleaning:", vitals.shape)

# =====================================================
# RENAME COLUMNS
# =====================================================
vitals = vitals.rename(columns={
    "patientunitstayid": "patient_id",
    "observationoffset": "time",
    "heartrate": "HR",
    "respiration": "RR",
    "sao2": "SpO2",
    "temperature": "Temp",
    "systemicsystolic": "SBP",
    "systemicdiastolic": "DBP",
    "systemicmean": "MAP"
})

# =====================================================
# SELECT LARGE COHORT
# =====================================================
print("\nSelecting larger ICU cohort...")

# count rows per patient
patient_counts = (
    vitals["patient_id"]
    .value_counts()
)

# keep patients with enough observations
eligible_patients = patient_counts[
    patient_counts >= 50
].index

print(
    "Eligible patients:",
    len(eligible_patients)
)

# choose first 1000
selected_patients = eligible_patients[:1000]

print(
    "Selected cohort size:",
    len(selected_patients)
)

subset = vitals[
    vitals["patient_id"]
    .isin(selected_patients)
].copy()

# =====================================================
# SORT TEMPORALLY
# =====================================================
subset = subset.sort_values(
    by=["patient_id", "time"]
)

# =====================================================
# SAVE
# =====================================================
output_path = "outputs/real_icu_signals.csv"

subset.to_csv(
    output_path,
    index=False
)

print("\nSaved cleaned ICU signals:")
print(output_path)

# =====================================================
# SHOW SAMPLE
# =====================================================
print("\n=== SAMPLE DATA ===")
print(subset.head())

print("\nFinal shape:", subset.shape)

print("\nREAL ICU EXTRACTION COMPLETED.")