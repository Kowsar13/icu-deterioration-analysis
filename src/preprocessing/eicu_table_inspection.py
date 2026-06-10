print("SCRIPT STARTED")
import pandas as pd

print("\n=== eICU TABLE INSPECTION ===\n")

# =========================================
# FILE PATHS
# =========================================
patient_file = "data/eicu-database2.0.1/patient.csv.gz"
vital_file = "data/eicu-database2.0.1/vitalPeriodic.csv.gz"
nurse_file = "data/eicu-database2.0.1/nurseCharting.csv.gz"

# =========================================
# LOAD SMALL SAMPLES
# =========================================
print("Loading patient table...")
patient = pd.read_csv(patient_file, compression="gzip", nrows=5)

print("Loading vitalPeriodic table...")
vital = pd.read_csv(vital_file, compression="gzip", nrows=5)

print("Loading nurseCharting table...")
nurse = pd.read_csv(nurse_file, compression="gzip", nrows=5)

# =========================================
# SHOW COLUMN NAMES
# =========================================
print("\n=== PATIENT COLUMNS ===")
print(patient.columns.tolist())

print("\n=== VITAL PERIODIC COLUMNS ===")
print(vital.columns.tolist())

print("\n=== NURSE CHARTING COLUMNS ===")
print(nurse.columns.tolist())

# =========================================
# SHOW SAMPLE DATA
# =========================================
print("\n=== PATIENT SAMPLE ===")
print(patient.head())

print("\n=== VITAL SAMPLE ===")
print(vital.head())

print("\n=== NURSE SAMPLE ===")
print(nurse.head())

print("\nInspection completed successfully.")