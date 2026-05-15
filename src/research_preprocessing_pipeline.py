import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

print("STARTING PIPELINE")

# ======================================
# CREATE OUTPUT DIRECTORY
# ======================================
os.makedirs("outputs", exist_ok=True)

# ======================================
# GENERATE SIMPLE ICU DATA
# ======================================
np.random.seed(42)

time = np.arange(100)

hr = 75 + np.random.normal(0, 2, 100)
bp = 120 + np.random.normal(0, 3, 100)
rr = 16 + np.random.normal(0, 1, 100)

# deterioration
hr[60:] += np.linspace(0, 20, 40)
bp[60:] -= np.linspace(0, 25, 40)

# create dataframe
df = pd.DataFrame({
    "time": time,
    "HR": hr,
    "BP": bp,
    "RR": rr
})

print("DATAFRAME CREATED")

# ======================================
# SAVE CSV
# ======================================
csv_path = "outputs/processed_icu_data.csv"

df.to_csv(csv_path, index=False)

print("CSV SAVED:", csv_path)

# ======================================
# PLOT
# ======================================
plt.figure(figsize=(10,5))

plt.plot(df["time"], df["HR"], label="Heart Rate")
plt.plot(df["time"], df["BP"], label="Blood Pressure")

plt.axvline(60, color="red", linestyle="--")

plt.legend()
plt.grid()

image_path = "outputs/preprocessing_plot.png"

plt.savefig(image_path)

print("PLOT SAVED:", image_path)

print("PIPELINE COMPLETED SUCCESSFULLY")