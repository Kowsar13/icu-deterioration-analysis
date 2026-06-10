import pandas as pd
import numpy as np
import os

print("\n=== STAGE 1: INSTABILITY ANALYSIS ===\n")

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

    instability_list = []

    for sig in signals:

        # skip missing signal completely
        if patient_df[sig].isna().all():
            continue

        signal = (
            patient_df[sig]
            .interpolate()
            .ffill()
            .bfill()
        )

        # if still invalid skip
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

        instability_list.append(
            deviation.values
        )

    # if no usable signals
    if len(instability_list) == 0:
        continue

    total_instability = np.sum(
        instability_list,
        axis=0
    )

    peak_idx = int(
        np.argmax(
            total_instability
        )
    )

    peak_time = float(
        patient_df.iloc[
            peak_idx
        ]["time"]
    )

    results.append({
        "patient_id": pid,
        "peak_instability":
            float(
                np.max(
                    total_instability
                )
            ),
        "peak_time":
            peak_time,
        "mean_instability":
            float(
                np.mean(
                    total_instability
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
    "outputs/instability_summary.csv"
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
    "\nINSTABILITY ANALYSIS COMPLETED."
)