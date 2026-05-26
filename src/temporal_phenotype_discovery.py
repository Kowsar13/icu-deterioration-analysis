import pandas as pd
import numpy as np
import os

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

print("\n=== TEMPORAL PHENOTYPE DISCOVERY ===\n")

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
# FEATURE EXTRACTION
# =====================================================
feature_rows = []

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

    features = []

    for sig in signals:

        if patient_df[sig].isna().all():

            features.extend(
                [0, 0, 0, 0]
            )

            continue

        signal = (
            patient_df[sig]
            .interpolate()
            .ffill()
            .bfill()
        )

        features.extend([
            float(signal.mean()),
            float(signal.std()),
            float(signal.min()),
            float(signal.max())
        ])

    feature_rows.append(
        features
    )

# =====================================================
# MATRIX
# =====================================================
X = np.array(
    feature_rows
)

X = np.nan_to_num(
    X,
    nan=0
)

print(
    "\nFeature matrix:",
    X.shape
)

# =====================================================
# PCA
# =====================================================
print("\nRunning PCA...")

pca = PCA(
    n_components=3
)

reduced = pca.fit_transform(
    X
)

# =====================================================
# CLUSTERING
# =====================================================
print(
    "Running clustering..."
)

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

labels = kmeans.fit_predict(
    reduced
)

# =====================================================
# SAVE
# =====================================================
summary = pd.DataFrame({
    "patient_id":
        patient_ids,
    "phenotype":
        labels
})

save_path = (
    "outputs/temporal_phenotypes.csv"
)

summary.to_csv(
    save_path,
    index=False
)

print("\nSaved:")
print(save_path)

print("\n=== SAMPLE ===")
print(summary.head())

print("\n=== COUNTS ===")
print(
    summary["phenotype"]
    .value_counts()
)

print(
    "\nPHENOTYPE DISCOVERY COMPLETED."
)