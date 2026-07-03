import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ruptures as rpt
import os

print("Starting Stage 2 (Improved DTW).....")

os.makedirs("outputs", exist_ok=True)

# ==============================
# MULTIVARIATE DTW
# ==============================
def dtw_multivariate(s1, s2):
    n, m = len(s1), len(s2)
    dtw = np.full((n+1, m+1), np.inf)
    dtw[0, 0] = 0

    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = np.linalg.norm(s1[i-1] - s2[j-1])
            dtw[i, j] = cost + min(
                dtw[i-1, j],
                dtw[i, j-1],
                dtw[i-1, j-1]
            )
    return dtw[n, m]

# ==============================
# Generate Patient
# ==============================
def generate_patient(seed):
    np.random.seed(seed)
    n = 100

    hr = 75 + np.random.normal(0, 1, n)
    bp = 120 + np.random.normal(0, 2, n)
    rr = 16 + np.random.normal(0, 0.5, n)

    change = np.random.randint(40, 60)
    pattern = np.random.choice(["infection", "respiratory", "circulatory"])

    if pattern == "infection":
        hr[change:] += np.linspace(0, 20, n-change)
        bp[change:] -= np.linspace(0, 20, n-change)
        rr[change:] += np.linspace(0, 10, n-change)

    elif pattern == "respiratory":
        rr[change:] += np.linspace(0, 15, n-change)
        hr[change:] += np.linspace(0, 10, n-change)

    elif pattern == "circulatory":
        bp[change:] -= np.linspace(0, 30, n-change)
        hr[change:] += np.linspace(0, 15, n-change)

    return hr, bp, rr, change, pattern

# ==============================
# PROTOTYPES (MULTIVARIATE)
# ==============================
def get_prototypes(length=20):
    t = np.arange(length)

    infection = np.vstack([
        np.linspace(0, 1, length),
        np.linspace(0, -1, length),
        np.linspace(0, 1, length)
    ]).T

    respiratory = np.vstack([
        np.linspace(0, 0.5, length),
        np.zeros(length),
        np.linspace(0, 1.5, length)
    ]).T

    circulatory = np.vstack([
        np.linspace(0, 1, length),
        np.linspace(0, -2, length),
        np.zeros(length)
    ]).T

    return {
        "infection": infection,
        "respiratory": respiratory,
        "circulatory": circulatory
    }

# ==============================
# DETECTION
# ==============================
def detect_instability(hr, bp, rr):
    signal = np.vstack([hr, bp, rr]).T
    model = rpt.Pelt(model="rbf").fit(signal)
    breakpoints = model.predict(pen=3)
    return breakpoints[0] if len(breakpoints) > 1 else None

# ==============================
# MAIN
# ==============================
prototypes = get_prototypes()
results = []

for i in range(5):
    print(f"\nPatient {i}")

    hr, bp, rr, change, true_pattern = generate_patient(i)
    detected = detect_instability(hr, bp, rr)

    if detected is None:
        continue

    segment = np.vstack([
        hr[detected:detected+20],
        bp[detected:detected+20],
        rr[detected:detected+20]
    ]).T

    if len(segment) < 20:
        continue

    # Normalize
    segment = (segment - np.mean(segment, axis=0)) / np.std(segment, axis=0)

    scores = {}
    for name, proto in prototypes.items():
        scores[name] = dtw_multivariate(segment, proto)

    predicted = min(scores, key=scores.get)

    print(f"True: {true_pattern}, Pred: {predicted}")

    results.append({
        "patient": i,
        "true_pattern": true_pattern,
        "predicted_pattern": predicted
    })

# ==============================
# RESULTS
# ==============================
df = pd.DataFrame(results)
print("\nFINAL RESULTS:\n")
print(df)

df.to_csv("outputs/stage2_results_improved.csv", index=False)
