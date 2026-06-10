import pandas as pd
import numpy as np
import os

print("\n=== PHYSIOLOGICAL GRAPH BUILDER ===\n")

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv(
    "outputs/eicu_preprocessed.csv"
)

signals = [
    "HR",
    "RR",
    "SpO2",
    "MAP"
]

patients = df["patient_id"].unique()

graph_rows = []

print("Patients:", len(patients))

for i, pid in enumerate(patients):

    patient = df[
        df["patient_id"] == pid
    ][signals]

    corr = patient.corr()

    graph_rows.append({
        "patient_id": pid,

        "HR_RR":
            corr.loc["HR","RR"],

        "HR_SpO2":
            corr.loc["HR","SpO2"],

        "HR_MAP":
            corr.loc["HR","MAP"],

        "RR_SpO2":
            corr.loc["RR","SpO2"],

        "RR_MAP":
            corr.loc["RR","MAP"],

        "SpO2_MAP":
            corr.loc["SpO2","MAP"]
    })

    if (i+1) % 100 == 0:
        print(
            f"Processed {i+1}/{len(patients)}"
        )

graph_df = pd.DataFrame(
    graph_rows
)

graph_df = graph_df.fillna(0)

save_path = (
    "outputs/physiological_graph_features.csv"
)

graph_df.to_csv(
    save_path,
    index=False
)

print("\nSaved:")
print(save_path)

print("\nShape:")
print(graph_df.shape)

print("\nSample:")
print(graph_df.head())

print(
    "\nGRAPH BUILDING COMPLETED."
)