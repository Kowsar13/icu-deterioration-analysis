import pandas as pd
import numpy as np
import os

print("\n=== TEMPORAL ATTENTION MODEL ===\n")

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

results = []

patient_ids = df["patient_id"].unique()

print(
    "Patients:",
    len(patient_ids)
)

# =====================================================
# ATTENTION FUNCTION
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

        deviations.append(
            deviation.values
        )

    if len(deviations) == 0:
        continue

    total_deviation = np.sum(
        deviations,
        axis=0
    )

    attention = softmax(
        total_deviation
    )

    critical_idx = int(
        np.argmax(attention)
    )

    critical_time = float(
        patient_df.iloc[
            critical_idx
        ]["time"]
    )

    results.append({
        "patient_id": pid,
        "critical_time":
            critical_time,
        "mean_attention":
            float(
                np.mean(
                    attention
                )
            ),
        "max_attention":
            float(
                np.max(
                    attention
                )
            )
    })

# =====================================================
# SAVE
# =====================================================
summary = pd.DataFrame(
    results
)

save_path = (
    "outputs/temporal_attention_summary.csv"
)

summary.to_csv(
    save_path,
    index=False
)

print("\nSaved:")
print(save_path)

print("\n=== SAMPLE ===")
print(summary.head())

print(
    "\nTEMPORAL ATTENTION COMPLETED."
)