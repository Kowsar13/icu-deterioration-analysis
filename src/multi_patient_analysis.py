import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ruptures as rpt
import os

os.makedirs("outputs", exist_ok=True)

# ==============================
# Generate Multiple Patients
# ==============================
def generate_patient(seed):
    np.random.seed(seed)
    n = 100

    hr = 75 + np.random.normal(0, 1, n)
    bp = 120 + np.random.normal(0, 2, n)
    rr = 16 + np.random.normal(0, 0.5, n)

    # Random instability start
    change = np.random.randint(50, 80)

    hr[change:] += np.linspace(0, 20, n-change)
    bp[change:] -= np.linspace(0, 25, n-change)
    rr[change:] += np.linspace(0, 10, n-change)

    return hr, bp, rr, change

# ==============================
# Detect Instability
# ==============================
def detect_instability(hr, bp, rr):
    signal = np.vstack([hr, bp, rr]).T

    model = rpt.Pelt(model="rbf").fit(signal)
    breakpoints = model.predict(pen=3)

    detected = breakpoints[0] if len(breakpoints) > 1 else None
    return detected

# ==============================
# Run Experiment
# ==============================
results = []

for i in range(5):
    hr, bp, rr, true_change = generate_patient(i)

    detected_change = detect_instability(hr, bp, rr)

    results.append({
        "patient": i,
        "true_change": true_change,
        "detected_change": detected_change,
        "error": abs(true_change - detected_change) if detected_change else None
    })

    # Plot each patient
    plt.figure(figsize=(8, 4))
    plt.plot(hr, label="HR")
    plt.plot(bp, label="BP")
    plt.plot(rr, label="RR")

    plt.axvline(true_change, color="green", linestyle="--", label="True Change")
    if detected_change:
        plt.axvline(detected_change, color="red", linestyle="--", label="Detected")

    plt.title(f"Patient {i}")
    plt.legend()
    plt.grid()

    plt.savefig(f"outputs/patient_{i}.png")
    plt.close()

# ==============================
# Results Table
# ==============================
df = pd.DataFrame(results)
print("\nRESULTS:\n")
print(df)

# Save results
df.to_csv("outputs/results.csv", index=False)