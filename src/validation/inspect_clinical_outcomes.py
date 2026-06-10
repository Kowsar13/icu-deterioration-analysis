import pandas as pd

print("\n=== PATIENT TABLE ===\n")

patient = pd.read_csv(
    "data/eicu-database2.0.1/patient.csv.gz",
    compression="gzip",
    nrows=5
)

print(patient.columns.tolist())

print("\nSample:")
print(patient.head())

print("\n=== APACHE PATIENT RESULT ===\n")

apache = pd.read_csv(
    "data/eicu-database2.0.1/apachePatientResult.csv.gz",
    compression="gzip",
    nrows=5
)

print(apache.columns.tolist())

print("\nSample:")
print(apache.head())

print("\n=== APACHE APS VAR ===\n")

aps = pd.read_csv(
    "data/eicu-database2.0.1/apacheApsVar.csv.gz",
    compression="gzip",
    nrows=5
)

print(aps.columns.tolist())

print("\n=== APACHE PRED VAR ===\n")

pred = pd.read_csv(
    "data/eicu-database2.0.1/apachePredVar.csv.gz",
    compression="gzip",
    nrows=5
)

print(pred.columns.tolist())