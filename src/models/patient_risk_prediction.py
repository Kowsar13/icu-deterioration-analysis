import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

print("\n=== ICU PATIENT RISK PREDICTION ===\n")

os.makedirs("outputs", exist_ok=True)

# =====================================================
# LOAD
# =====================================================
df = pd.read_csv(
    "outputs/final_clinical_summary.csv"
)

print("Loaded:", df.shape)

# =====================================================
# CREATE LABEL
# =====================================================
# high risk = top 25% ATDI
threshold = (
    df["peak_ATDI"]
    .quantile(0.75)
)

df["high_risk"] = (
    df["peak_ATDI"] >= threshold
).astype(int)

print(
    "High-risk threshold:",
    round(threshold, 3)
)

# =====================================================
# FEATURES
# =====================================================
features = [
    "peak_instability",
    "mean_instability",
    "mean_attention",
    "max_attention",
    "phenotype",
    "peak_ATDI",
    "mean_ATDI",
    "graph_score"
]

X = df[
    features
].copy()

X = X.fillna(0)

y = df["high_risk"]

# =====================================================
# TRAIN TEST SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(
    "Train:",
    X_train.shape
)

print(
    "Test:",
    X_test.shape
)

# =====================================================
# MODEL
# =====================================================
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

# =====================================================
# PREDICT
# =====================================================
pred = model.predict(
    X_test
)

# =====================================================
# METRICS
# =====================================================
acc = accuracy_score(
    y_test,
    pred
)

print("\nAccuracy:")
print(
    round(acc, 4)
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        pred
    )
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        pred
    )
)

# =====================================================
# FEATURE IMPORTANCE
# =====================================================
importance = pd.DataFrame({
    "feature":
        features,
    "importance":
        model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

save_path = (
    "outputs/feature_importance.csv"
)

importance.to_csv(
    save_path,
    index=False
)

print("\nSaved:")
print(save_path)

print("\n=== FEATURE IMPORTANCE ===")
print(importance)

print(
    "\nRISK PREDICTION COMPLETED."
)