import pandas as pd
import numpy as np
import os

print("\n=== ADAPTIVE TEMPORAL DETERIORATION INDEX (ATDI) ===\n")

os.makedirs("outputs", exist_ok=True)

# =====================================================
# LOAD
# =====================================================
df = pd.read_csv(
    "outputs/eicu_preprocessed.csv"
)

print("Loaded:", df.shape)

signals = [
    "HR",
    "RR",
    "SpO2",
    "MAP"
]

patient_ids = df["patient_id"].unique()

print(
    "Patients:",
    len(patient_ids)
)

# =====================================================
# WEIGHTS
# =====================================================
alpha = 0.5
beta = 0.3
gamma = 0.2

results = []

# =====================================================
# SOFTMAX
# =====================================================
def softmax(x):

    x = np.array(x)

    x = np.nan_to_num(
        x,
        nan=0
    )

    exp_x = np.exp(
        x - np.max(x)
    )

    denom = exp_x.sum()

    if denom == 0:
        return np.zeros_like(x)

    return exp_x / denom

# =====================================================
# PROCESS
# =====================================================
for i, pid in enumerate(
    patient_ids,
    start=1
):

    if i % 100 == 0:
        print(
            f"Processed {i}/{len(patient_ids)}"
        )

    patient_df = df[
        df["patient_id"] == pid
    ].copy()

    patient_df = patient_df.sort_values(
        "time"
    )

    deviations = []

    valid_signals = []

    for sig in signals:

        if patient_df[sig].isna().all():
            continue

        signal = (
            patient_df[sig]
            .interpolate()
            .ffill()
            .bfill()
        )

        if signal.isna().all():
            continue

        baseline = (
            signal
            .rolling(
                window=12,
                min_periods=1
            )
            .mean()
        )

        deviation = (
            signal - baseline
        ).abs()

        deviation = deviation.fillna(0)

        deviations.append(
            deviation.values
        )

        valid_signals.append(
            signal.values
        )

    if len(deviations) == 0:
        continue

    # =========================
    # I(t) instability
    # =========================
    instability = np.sum(
        deviations,
        axis=0
    )

    # =========================
    # A(t) attention
    # =========================
    attention = softmax(
        instability
    )

    # =========================
    # G(t) graph interaction
    # =========================
    signal_matrix = np.array(
        valid_signals
    )

    if signal_matrix.shape[0] > 1:

        corr = np.corrcoef(
            signal_matrix
        )

        corr = np.nan_to_num(
            corr,
            nan=0
        )

        graph_score = float(
            np.mean(
                np.abs(corr)
            )
        )

    else:
        graph_score = 0

    # =========================
    # ATDI
    # =========================
    atdi_series = (
        alpha * instability
        +
        beta * attention
        +
        gamma * graph_score
    )

    peak_idx = int(
        np.argmax(
            atdi_series
        )
    )

    peak_time = float(
        patient_df.iloc[
            peak_idx
        ]["time"]
    )

    results.append({
        "patient_id":
            pid,

        "peak_ATDI":
            float(
                np.max(
                    atdi_series
                )
            ),

        "mean_ATDI":
            float(
                np.mean(
                    atdi_series
                )
            ),

        "critical_time":
            peak_time,

        "graph_score":
            graph_score
    })

# =====================================================
# SAVE
# =====================================================
summary = pd.DataFrame(
    results
)

save_path = (
    "outputs/ATDI_summary.csv"
)

summary.to_csv(
    save_path,
    index=False
)

# =====================================================
# TOP HIGH-RISK
# =====================================================
top = (
    summary
    .sort_values(
        "peak_ATDI",
        ascending=False
    )
    .head(10)
)

print("\nSaved:")
print(save_path)

print("\n=== TOP HIGH-RISK PATIENTS ===")
print(top)

print(
    "\nATDI COMPLETED."
)