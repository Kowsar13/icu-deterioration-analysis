import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print("\n=== REAL ICU PREPROCESSING ===\n")

# =====================================================
# CREATE OUTPUT DIRECTORY
# =====================================================
os.makedirs("outputs", exist_ok=True)

# =====================================================
# LOAD REAL ICU SIGNALS
# =====================================================
file_path = "outputs/real_icu_signals.csv"

print("Loading ICU signals...")

df = pd.read_csv(file_path)

print("Original shape:", df.shape)

# =====================================================
# SIGNAL COLUMNS
# =====================================================
signal_columns = [
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

print("\nProcessing patients...\n")

for pid in patient_ids:

    print(f"Processing patient {pid}...")

    patient_df = df[df["patient_id"] == pid].copy()

    # sort by time
    patient_df = patient_df.sort_values("time")

    # =================================================
    # INTERPOLATION
    # =================================================
    patient_df[signal_columns] = (
        patient_df[signal_columns]
        .interpolate()
    )

    # =================================================
    # FILL REMAINING MISSING VALUES
    # =================================================
    patient_df[signal_columns] = (
        patient_df[signal_columns]
        .ffill()
        .bfill()
    )

    # =================================================
    # SMOOTHING
    # =================================================
    for col in signal_columns:

        patient_df[col] = (
            patient_df[col]
            .rolling(window=3, min_periods=1)
            .mean()
        )

    # =================================================
    # NORMALIZATION
    # =================================================
    for col in signal_columns:

        mean = patient_df[col].mean()
        std = patient_df[col].std()

        if std != 0:

            patient_df[col] = (
                patient_df[col] - mean
            ) / std

    processed_patients.append(patient_df)

# =====================================================
# COMBINE ALL PATIENTS
# =====================================================
processed_df = pd.concat(processed_patients)

print("\nProcessed shape:", processed_df.shape)

# =====================================================
# SAVE PREPROCESSED DATA
# =====================================================
output_csv = "outputs/eicu_preprocessed.csv"

processed_df.to_csv(output_csv, index=False)

print("\nSaved preprocessed dataset:")
print(output_csv)

# =====================================================
# VISUALIZE ONE PATIENT
# =====================================================
sample_patient = patient_ids[0]

plot_df = processed_df[
    processed_df["patient_id"] == sample_patient
]

plt.figure(figsize=(12,6))

plt.plot(
    plot_df["time"],
    plot_df["HR"],
    label="HR"
)

plt.plot(
    plot_df["time"],
    plot_df["RR"],
    label="RR"
)

plt.plot(
    plot_df["time"],
    plot_df["SpO2"],
    label="SpO2"
)

plt.title(f"Preprocessed ICU Signals - Patient {sample_patient}")

plt.xlabel("Time Offset")
plt.ylabel("Normalized Value")

plt.legend()
plt.grid()

plot_path = "outputs/eicu_preprocessing_plot.png"

plt.savefig(plot_path)

print("\nSaved visualization:")
print(plot_path)

print("\nREAL ICU PREPROCESSING COMPLETED.")