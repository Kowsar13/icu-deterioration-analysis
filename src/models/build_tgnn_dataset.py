import pandas as pd
import numpy as np
import os

print("\n=== BUILDING TGNN DATASET ===\n")

sequence_folder = "outputs/graph_sequences"

mortality_df = pd.read_csv(
    "outputs/mortality_dataset.csv"
)

results = []

files = [
    f for f in os.listdir(sequence_folder)
    if f.endswith(".csv")
]

print("Graph sequences found:", len(files))

for i, file in enumerate(files):

    patient_id = int(
        file.replace(".csv", "")
    )

    seq = pd.read_csv(
        os.path.join(
            sequence_folder,
            file
        )
    )

    if len(seq) < 2:
        continue

    row = {
        "patient_id": patient_id,

        "mean_HR_RR":
            seq["HR_RR"].mean(),

        "std_HR_RR":
            seq["HR_RR"].std(),

        "mean_HR_SpO2":
            seq["HR_SpO2"].mean(),

        "std_HR_SpO2":
            seq["HR_SpO2"].std(),

        "mean_RR_SpO2":
            seq["RR_SpO2"].mean(),

        "std_RR_SpO2":
            seq["RR_SpO2"].std(),

        "num_windows":
            len(seq)
    }

    results.append(row)

    if (i + 1) % 100 == 0:
        print(
            f"Processed {i+1}/{len(files)}"
        )

graph_df = pd.DataFrame(
    results
)

final_df = graph_df.merge(
    mortality_df[
        [
            "patient_id",
            "mortality_label"
        ]
    ],
    on="patient_id",
    how="inner"
)

save_path = (
    "outputs/tgnn_dataset.csv"
)

final_df.to_csv(
    save_path,
    index=False
)

print("\nSaved:")
print(save_path)

print("\nShape:")
print(final_df.shape)

print("\nMortality Counts:")
print(
    final_df[
        "mortality_label"
    ].value_counts()
)

print("\nSample:")
print(
    final_df.head()
)

print(
    "\nTGNN DATASET COMPLETED."
)