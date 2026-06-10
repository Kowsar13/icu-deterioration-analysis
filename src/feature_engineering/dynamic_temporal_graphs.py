print("\n=== DYNAMIC TEMPORAL GRAPH ANALYSIS ===\n")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os



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

print("\nAnalyzing temporal graph evolution...\n")

for pid in patient_ids:

    print(f"Processing patient {pid}...")

    patient_df = df[
        df["patient_id"] == pid
    ].copy()

    patient_df = patient_df.sort_values("time")

    # =================================================
    # TEMPORAL WINDOWS
    # =================================================
    window_size = 20

    graph_scores = []

    time_points = []

    for start in range(
        0,
        len(patient_df) - window_size,
        5
    ):

        end = start + window_size

        window_df = patient_df.iloc[start:end]

        correlation_matrix = (
            window_df[signals]
            .corr()
        )

        graph_strength = (
            np.abs(
                correlation_matrix.values
            ).mean()
        )

        graph_scores.append(graph_strength)

        time_points.append(
            window_df["time"].mean()
        )

    # =================================================
    # DETECT DYNAMIC INSTABILITY
    # =================================================
    graph_scores = np.array(graph_scores)

    threshold = (
        graph_scores.mean()
        + graph_scores.std()
    )

    unstable_windows = (
        graph_scores > threshold
    )

    if np.any(unstable_windows):

        instability_time = (
            time_points[
                np.argmax(unstable_windows)
            ]
        )

    else:

        instability_time = None

    summary_results.append({
        "patient_id": pid,
        "dynamic_instability_time":
            instability_time,
        "mean_graph_strength":
            graph_scores.mean()
    })

    # =================================================
    # VISUALIZATION
    # =================================================
    plt.figure(figsize=(12,6))

    plt.plot(
        time_points,
        graph_scores,
        label="Graph Interaction Strength"
    )

    plt.axhline(
        threshold,
        color="red",
        linestyle="--",
        label="Dynamic Threshold"
    )

    if instability_time is not None:

        plt.axvline(
            instability_time,
            color="green",
            linestyle="--",
            label="Dynamic Instability"
        )

    plt.title(
        f"Dynamic Temporal Graph Evolution - Patient {pid}"
    )

    plt.xlabel("Time")
    plt.ylabel("Interaction Strength")

    plt.legend()
    plt.grid()

    save_path = (
        f"outputs/dynamic_graph_{pid}.png"
    )

    plt.savefig(save_path)

    plt.close()

# =====================================================
# SAVE RESULTS
# =====================================================
summary_df = pd.DataFrame(summary_results)

summary_path = (
    "outputs/dynamic_graph_summary.csv"
)

summary_df.to_csv(summary_path, index=False)

print("\n=== DYNAMIC GRAPH SUMMARY ===")
print(summary_df)

print("\nSaved summary:")
print(summary_path)

print("\nDYNAMIC TEMPORAL GRAPH ANALYSIS COMPLETED.")