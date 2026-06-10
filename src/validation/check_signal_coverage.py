print("check")
import pandas as pd

df = pd.read_csv(
    "outputs/eicu_preprocessed.csv"
)

signals = [
    "HR",
    "RR",
    "SpO2",
    "MAP",
    "Temp"
]

print("\n=== SIGNAL COVERAGE ===\n")

for col in signals:

    print(
        col,
        "non-null:",
        df[col].notna().sum(),
        "unique:",
        df[col].nunique()
    )