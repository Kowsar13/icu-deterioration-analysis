print("Running")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print("\n=== ADAPTIVE EXPLAINABLE INSTABILITY MODEL ===\n")

# =====================================================
# CREATE OUTPUT DIRECTORY
# =====================================================
os.makedirs("outputs", exist_ok=True)

# =====================================================
# LOAD PREPROCESSED ICU DATA
# =====================================================
file_path = "outputs/eicu_preprocessed.csv"

print("Loading preprocessed ICU dataset...")

df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)

# =====================================================
# SIGNALS
# =====================================================
signals = [
    "HR",
    "RR",
    "SpO2",
    "MAP"
]

# =====================================================
# PATIENT PROCESSING
# =====================================================
patient_ids = df["patient_id"].unique()

summary_results = []

print("\nRunning adaptive explainable analysis...\n")

for pid in patient_ids:

    print(f"Processing patient {pid}...")

    patient_df = df[
        df["patient_id"] == pid
    ].copy()

    patient_df = patient_df.sort_values("time")

    # =================================================
    # ADAPTIVE BASELINE MODEL
    # =================================================
    contribution_matrix = []

    instability_score = np.zeros(len(patient_df))

    for signal in signals:

        # rolling patient-specific baseline
        baseline = (
            patient_df[signal]
            .rolling(window=8, min_periods=1)
            .mean()
        )

        # adaptive deviation
        deviation = np.abs(
            patient_df[signal] - baseline
        )

        contribution_matrix.append(deviation)

        instability_score += deviation

    # =================================================
    # SAVE CONTRIBUTIONS
    # =================================================
    contribution_matrix = np.array(contribution_matrix)

    patient_df["instability_score"] = instability_score

    # =================================================
    # ADAPTIVE THRESHOLD
    # =================================================
    adaptive_threshold = (
        instability_score.mean()
        + 1.5 * instability_score.std()
    )

    patient_df["high_risk"] = (
        patient_df["instability_score"]
        > adaptive_threshold
    )

    # =================================================
    # DETECT EARLY INSTABILITY
    # =================================================
    risk_points = patient_df[
        patient_df["high_risk"] == True
    ]

    if len(risk_points) > 0:

        detected_time = risk_points.iloc[0]["time"]

    else:

        detected_time = None

    # =================================================
    # SIGNAL CONTRIBUTION ANALYSIS
    # =================================================
    mean_contributions = {}

    for i, signal in enumerate(signals):

        mean_contributions[signal] = (
            contribution_matrix[i].mean()
        )

    dominant_signal = max(
        mean_contributions,
        key=mean_contributions.get
    )

    # =================================================
    # SAVE SUMMARY
    # =================================================
    summary_results.append({
        "patient_id": pid,
        "detected_time": detected_time,
        "dominant_signal": dominant_signal,
        "mean_instability": instability_score.mean()
    })

    # =================================================
    # VISUALIZATION
    # =================================================
    plt.figure(figsize=(13,6))

    plt.plot(
        patient_df["time"],
        patient_df["instability_score"],
        label="Adaptive Instability Score"
    )

    plt.axhline(
        adaptive_threshold,
        color="red",
        linestyle="--",
        label="Adaptive Threshold"
    )

    if detected_time is not None:

        plt.axvline(
            detected_time,
            color="green",
            linestyle="--",
            label="Detected Instability"
        )

    plt.title(
        f"Adaptive Explainable Instability - Patient {pid}"
    )

    plt.xlabel("Time")
    plt.ylabel("Risk Score")

    plt.legend()
    plt.grid()

    save_path = (
        f"outputs/adaptive_instability_{pid}.png"
    )

    plt.savefig(save_path)

    plt.close()

    # =================================================
    # CONTRIBUTION BARPLOT
    # =================================================
    plt.figure(figsize=(8,5))

    plt.bar(
        mean_contributions.keys(),
        mean_contributions.values()
    )

    plt.title(
        f"Signal Contribution Analysis - Patient {pid}"
    )

    plt.ylabel("Contribution Score")

    contribution_path = (
        f"outputs/contribution_analysis_{pid}.png"
    )

    plt.savefig(contribution_path)

    plt.close()

# =====================================================
# SAVE RESULTS
# =====================================================
summary_df = pd.DataFrame(summary_results)

summary_path = (
    "outputs/adaptive_instability_summary.csv"
)

summary_df.to_csv(summary_path, index=False)

print("\n=== FINAL RESULTS ===")
print(summary_df)

print("\nSaved summary:")
print(summary_path)

print("\nADAPTIVE EXPLAINABLE ANALYSIS COMPLETED.")