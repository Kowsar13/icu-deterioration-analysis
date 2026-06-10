import pandas as pd
import numpy as np
import os

print("\n=== BUILDING GRAPH SEQUENCES ===\n")

os.makedirs("outputs/graph_sequences", exist_ok=True)

df = pd.read_csv(
    "outputs/eicu_preprocessed.csv"
)

signals = [
    "HR",
    "RR",
    "SpO2"
]

patients = df["patient_id"].unique()

WINDOW_SIZE = 50

print("Patients:", len(patients))

saved = 0

for idx, pid in enumerate(patients):

    patient = (
        df[df["patient_id"] == pid]
        .sort_values("time")
    )

    sequence_rows = []

    for start in range(
        0,
        len(patient),
        WINDOW_SIZE
    ):

        chunk = patient.iloc[
            start:start+WINDOW_SIZE
        ]

        if len(chunk) < 10:
            continue

        corr = chunk[
            signals
        ].corr()

        sequence_rows.append({

            "window":

                len(sequence_rows),

            "HR_RR":

                corr.loc[
                    "HR",
                    "RR"
                ],

            "HR_SpO2":

                corr.loc[
                    "HR",
                    "SpO2"
                ],

            "RR_SpO2":

                corr.loc[
                    "RR",
                    "SpO2"
                ]

        })

    if len(sequence_rows) == 0:
        continue

    seq_df = pd.DataFrame(
        sequence_rows
    )

    seq_df = seq_df.fillna(0)

    save_path = (
        f"outputs/graph_sequences/{pid}.csv"
    )

    seq_df.to_csv(
        save_path,
        index=False
    )

    saved += 1

    if (idx + 1) % 100 == 0:

        print(
            f"Processed {idx+1}/{len(patients)}"
        )

print("\nSaved graph sequences:", saved)

print(
    "\nGRAPH SEQUENCE GENERATION COMPLETED."
)