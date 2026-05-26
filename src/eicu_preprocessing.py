import pandas as pd
import os

print("\n=== LARGE COHORT ICU PREPROCESSING ===\n")

# =====================================================
# CREATE OUTPUT
# =====================================================
os.makedirs("outputs", exist_ok=True)

# =====================================================
# LOAD
# =====================================================
file_path = "outputs/real_icu_signals.csv"

print("Loading ICU signals...")

df = pd.read_csv(file_path)

print("Loaded shape:", df.shape)

# =====================================================
# SORT
# =====================================================
df = df.sort_values(
    by=["patient_id", "time"]
)

# =====================================================
# SIGNALS
# =====================================================
signals = [
    "HR",
    "RR",
    "SpO2",
    "Temp",
    "SBP",
    "DBP",
    "MAP"
]

# =====================================================
# PROCESS EACH PATIENT
# =====================================================
processed_patients = []

patient_ids = df["patient_id"].unique()

total_patients = len(patient_ids)

print(
    "Patients to preprocess:",
    total_patients
)

for i, pid in enumerate(patient_ids, start=1):

    if i % 100 == 0:
        print(
            f"Processed {i}/{total_patients}"
        )

    patient_df = df[
        df["patient_id"] == pid
    ].copy()

    patient_df = patient_df.sort_values(
        "time"
    )

    # =============================================
    # INTERPOLATION
    # =============================================
    patient_df[signals] = (
        patient_df[signals]
        .interpolate()
    )

    # =============================================
    # FORWARD FILL
    # =============================================
    patient_df[signals] = (
        patient_df[signals]
        .ffill()
    )

    # =============================================
    # BACKWARD FILL
    # =============================================
    patient_df[signals] = (
        patient_df[signals]
        .bfill()
    )

    # =============================================
    # NORMALIZATION
    # =============================================
    for col in signals:

        mean_val = (
            patient_df[col].mean()
        )

        std_val = (
            patient_df[col].std()
        )

        if (
            pd.notna(std_val)
            and std_val != 0
        ):
            patient_df[col] = (
                (
                    patient_df[col]
                    - mean_val
                ) / std_val
            )

    processed_patients.append(
        patient_df
    )

# =====================================================
# COMBINE
# =====================================================
final_df = pd.concat(
    processed_patients,
    ignore_index=True
)

# =====================================================
# SAVE
# =====================================================
save_path = (
    "outputs/eicu_preprocessed.csv"
)

final_df.to_csv(
    save_path,
    index=False
)

print("\nSaved:")
print(save_path)

print(
    "\nFinal shape:",
    final_df.shape
)

print(
    "\nPREPROCESSING COMPLETED."
)