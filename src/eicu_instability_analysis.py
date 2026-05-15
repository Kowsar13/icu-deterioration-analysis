import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print("\n=== REAL ICU INSTABILITY ANALYSIS ===\n")

# =====================================================
# CREATE OUTPUT FOLDER
# =====================================================
os.makedirs("outputs", exist_ok=True)

# =====================================================
# LOAD PREPROCESSED ICU DATA
# =====================================================
file_path = "outputs/eicu_preprocessed.csv"

print("Loading preprocessed ICU data...")

df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)

# =====================================================
# SIGNALS USED FOR INSTABILITY
# =====================================================
signals = [
    "HR",
    "RR",
    "SpO2"
]

# =====================================================
# PROCESS EACH PATIENT
# =====================================================
patient_ids = df["patient_id"].unique()

results = []

print("\nAnalyzing instability...\n")

for pid in patient_ids:

    print(f"Processing patient {pid}...")

    patient_df = df[
        df["patient_id"] == pid
    ].copy()

    patient_df = patient_df.sort_values("time")

    # =================================================
    # INSTABILITY SCORE
    # =================================================
    instability_score = np.zeros(len(patient_df))

    for signal in signals:

        rolling_mean = (
            patient_df[signal]
            .rolling(window=5, min_periods=1)
            .mean()
        )

        deviation = np.abs(
            patient_df[signal] - rolling_mean
        )

        instability_score += deviation

    patient_df["instability_score"] = instability_score

    # =================================================
    # DETECT INSTABILITY POINT
    # =================================================
    threshold = (
        patient_df["instability_score"]
        .mean()
        + patient_df["instability_score"].std()
    )

    unstable_points = patient_df[
        patient_df["instability_score"] > threshold
    ]

    if len(unstable_points) > 0:

        detected_time = unstable_points.iloc[0]["time"]

    else:

        detected_time = None

    results.append({
        "patient_id": pid,
        "detected_instability_time": detected_time,
        "mean_instability": patient_df["instability_score"].mean()
    })

    # =================================================
    # VISUALIZATION
    # =================================================
    plt.figure(figsize=(12,6))

    plt.plot(
        patient_df["time"],
        patient_df["instability_score"],
        label="Instability Score"
    )

    plt.axhline(
        threshold,
        color="red",
        linestyle="--",
        label="Threshold"
    )

    if detected_time is not None:

        plt.axvline(
            detected_time,
            color="green",
            linestyle="--",
            label="Detected Instability"
        )

    plt.title(
        f"ICU Instability Analysis - Patient {pid}"
    )

    plt.xlabel("Time Offset")
    plt.ylabel("Instability Score")

    plt.legend()
    plt.grid()

    save_path = (
        f"outputs/eicu_instability_{pid}.png"
    )

    plt.savefig(save_path)

    plt.close()

# =====================================================
# SAVE RESULTS
# =====================================================
results_df = pd.DataFrame(results)

results_csv = "outputs/eicu_instability_results.csv"

results_df.to_csv(results_csv, index=False)

print("\nSaved instability results:")
print(results_csv)

# =====================================================
# SHOW RESULTS
# =====================================================
print("\n=== RESULTS ===")
print(results_df)

print("\nREAL ICU INSTABILITY ANALYSIS COMPLETED.")