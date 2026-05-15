import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt
import os

# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)

# ==============================
# Step 1: Load Data
# ==============================
data = pd.read_csv("data/patient_data.csv")

# ==============================
# Step 2: Handle Missing Values
# ==============================
data = data.ffill()  # FIXED

# ==============================
# Step 3: Extract Signals
# ==============================
time = data["time"].values
hr = data["HR"].values
bp = data["BP"].values
rr = data["RR"].values

# ==============================
# Step 4: Smoothing (Moving Average)
# ==============================
def smooth(signal, window=3):
    return pd.Series(signal).rolling(window=window, min_periods=1).mean().values

hr_s = smooth(hr)
bp_s = smooth(bp)
rr_s = smooth(rr)

# ==============================
# Step 5: Combine Signals
# ==============================
signal = np.vstack([hr_s, bp_s, rr_s]).T

# ==============================
# Step 6: Change Point Detection
# ==============================
model = rpt.Pelt(model="rbf").fit(signal)
breakpoints = model.predict(pen=3)

change_point = breakpoints[0] if len(breakpoints) > 1 else None

# ==============================
# Step 7: Instability Score
# ==============================
baseline = np.mean(signal[:5], axis=0)
instability_score = np.linalg.norm(signal - baseline, axis=1)

# Normalize
instability_score = (instability_score - np.min(instability_score)) / (np.max(instability_score) - np.min(instability_score))

# ==============================
# Step 8: Plot Smoothed Signals
# ==============================
plt.figure(figsize=(10, 6))
plt.plot(time, hr_s, label="HR (smoothed)")
plt.plot(time, bp_s, label="BP (smoothed)")
plt.plot(time, rr_s, label="RR (smoothed)")

if change_point is not None:
    plt.axvline(x=change_point, linestyle="--", color="red", label="Detected Instability")

plt.title("Smoothed Physiological Signals with Instability Detection")
plt.xlabel("Time")
plt.ylabel("Values")
plt.legend()
plt.grid()

plt.savefig("outputs/smoothed_signals.png")
plt.show()

# ==============================
# Step 9: Plot Instability Score
# ==============================
plt.figure(figsize=(10, 4))
plt.plot(time, instability_score, color="purple", label="Instability Score I(t)")

if change_point is not None:
    plt.axvline(x=change_point, linestyle="--", color="red", label="Detected Instability")

plt.title("Normalized Instability Score Over Time")
plt.xlabel("Time")
plt.ylabel("Score")
plt.legend()
plt.grid()

plt.savefig("outputs/instability_score.png")
plt.show()

# ==============================
# Step 10: Print Results
# ==============================
print("Detected change points:", breakpoints)
print("Estimated instability time:", change_point)