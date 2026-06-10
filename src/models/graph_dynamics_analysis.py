print("Starting")
import pandas as pd
import numpy as np
import os

print("\n=== GRAPH DYNAMICS ANALYSIS ===\n")

sequence_folder = "outputs/graph_sequences"

results = []

files = [
    f for f in os.listdir(sequence_folder)
    if f.endswith(".csv")
]

print("Patients:", len(files))

for i, file in enumerate(files):

    pid = int(file.replace(".csv",""))

    seq = pd.read_csv(
        os.path.join(sequence_folder,file)
    )

    if len(seq) < 2:
        continue

    features = [
        "HR_RR",
        "HR_SpO2",
        "RR_SpO2"
    ]

    diffs = []

    for j in range(1,len(seq)):

        prev = seq.iloc[j-1][features].values
        curr = seq.iloc[j][features].values

        change = np.linalg.norm(
            curr - prev
        )

        diffs.append(change)

    results.append({

        "patient_id": pid,

        "mean_graph_change":
            np.mean(diffs),

        "max_graph_change":
            np.max(diffs),

        "graph_volatility":
            np.std(diffs)

    })

    if (i+1) % 100 == 0:
        print(
            f"Processed {i+1}/{len(files)}"
        )

out = pd.DataFrame(results)

save_path = (
    "outputs/graph_dynamics_summary.csv"
)

out.to_csv(
    save_path,
    index=False
)

print("\nSaved:")
print(save_path)

print("\nShape:")
print(out.shape)

print("\nSample:")
print(out.head())

print(
    "\nGRAPH DYNAMICS COMPLETED."
)