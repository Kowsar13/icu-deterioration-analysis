import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ruptures as rpt
import os

os.makedirs("outputs", exist_ok=True)

# ==============================
# Generate Patient Data
# ==============================
def generate_patient(seed):
    np.random.seed(seed)
    n = 100

    hr = 75 + np.random.normal(0, 1, n)
    bp = 120 + np.random.normal(0, 2, n)
    rr = 16 + np.random.normal(0, 0.5, n)

    # True clinical event happens later
    instability_start = np.random.randint(40, 60)
    event_time = instability_start + np.random.randint(10, 20)

    hr[instability_start:] += np.linspace(0, 20, n - instability_start)
    bp[instability_start:] -= np.linspace(0, 25, n - instability_start)
    rr[instability_start:] += np.linspace(0, 10, n - instability_start)

    return hr, bp, rr, instability_start, event_time


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
    hr, bp, rr, instability_start, event_time = generate_patient(i)

    detected_time = detect_instability(hr, bp, rr)

    lead_time = None
    if detected_time is not None:
        lead_time = event_time - detected_time

    results.append({
        "patient": i,
        "instability_start": instability_start,
        "clinical_event_time": event_time,
        "detected_time": detected_time,
        "lead_time": lead_time
    })

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(hr, label="HR")
    plt.plot(bp, label="BP")
    plt.plot(rr, label="RR")

    plt.axvline(instability_start, color="blue", linestyle="--", label="True Instability")
    plt.axvline(event_time, color="green", linestyle="--", label="Clinical Event")

    if detected_time:
        plt.axvline(detected_time, color="red", linestyle="--", label="Detected")

    plt.title(f"Patient {i}")
    plt.legend()
    plt.grid()

    plt.savefig(f"outputs/leadtime_patient_{i}.png")
    plt.close()

# ==============================
# Results
# ==============================
df = pd.DataFrame(results)
print("\nLEAD TIME RESULTS:\n")
print(df)

df.to_csv("outputs/lead_time_results.csv", index=False)

# ==============================
# Summary
# ==============================
avg_lead = df["lead_time"].mean()

print("\nAverage Lead Time:", avg_lead)