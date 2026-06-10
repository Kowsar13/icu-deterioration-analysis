import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print("\n=== PHYSIOLOGICAL GRAPH MODEL ===\n")

# =====================================================
# CREATE OUTPUT DIRECTORY
# =====================================================
os.makedirs("outputs", exist_ok=True)

# =====================================================
# LOAD ICU DATA
# =====================================================
file_path = "outputs/eicu_preprocessed.csv"

print("Loading ICU dataset...")

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
# PROCESS PATIENTS
# =====================================================
patient_ids = df["patient_id"].unique()

summary_results = []

print("\nBuilding physiological interaction graphs...\n")

for pid in patient_ids:

    print(f"Processing patient {pid}...")

    patient_df = df[
        df["patient_id"] == pid
    ].copy()

    patient_df = patient_df.sort_values("time")

    # =================================================
    # COMPUTE CORRELATION MATRIX
    # =================================================
    correlation_matrix = (
        patient_df[signals]
        .corr()
    )

    # =================================================
    # GRAPH INSTABILITY SCORE
    # =================================================
    graph_instability = (
        np.abs(correlation_matrix.values)
        .mean()
    )

    summary_results.append({
        "patient_id": pid,
        "graph_instability_score":
            graph_instability
    })

    # =================================================
    # VISUALIZATION
    # =================================================
    plt.figure(figsize=(7,6))

    plt.imshow(
        correlation_matrix,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    plt.xticks(
        range(len(signals)),
        signals
    )

    plt.yticks(
        range(len(signals)),
        signals
    )

    plt.colorbar(
        label="Correlation"
    )

    plt.title(
        f"Physiological Interaction Graph - Patient {pid}"
    )

    # =================================================
    # ADD VALUES INSIDE MATRIX
    # =================================================
    for i in range(len(signals)):
        for j in range(len(signals)):

            value = round(
                correlation_matrix.iloc[i,j],
                2
            )

            plt.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color="black"
            )

    save_path = (
        f"outputs/graph_patient_{pid}.png"
    )

    plt.savefig(save_path)

    plt.close()

# =====================================================
# SAVE SUMMARY
# =====================================================
summary_df = pd.DataFrame(summary_results)

summary_path = (
    "outputs/physiological_graph_summary.csv"
)

summary_df.to_csv(summary_path, index=False)

print("\n=== GRAPH SUMMARY ===")
print(summary_df)

print("\nSaved graph summary:")
print(summary_path)

print("\nPHYSIOLOGICAL GRAPH MODEL COMPLETED.")