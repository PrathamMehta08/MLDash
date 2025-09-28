import json
import numpy as np

with open("../inference_results/inference_results.json", "r") as f:
    data = json.load(f)

# Parameters
beta = 0.5
total_odlc = 46
thresholds = np.linspace(0, 1, 101)

best_threshold = None
best_fbeta = -1
best_counts = None

for threshold in thresholds:
    odlc_count = 0
    non_odlc_count = 0
    for item in data["detections_per_image"]:
        if any(conf > threshold for conf in item["confidences"]):
            if "_odlc" in item["image"]:
                odlc_count += 1
            else:
                non_odlc_count += 1

    f_beta = ((1 + beta**2) * odlc_count) / ((1 + beta**2) * odlc_count + non_odlc_count + beta**2 * (total_odlc - odlc_count))
    
    if f_beta > best_fbeta:
        best_fbeta = f_beta
        best_threshold = threshold
        best_counts = (odlc_count, non_odlc_count)

print(f"Best threshold: {best_threshold}")
print(f"ODLC images above threshold: {best_counts[0]}")
print(f"Non-ODLC images above threshold: {best_counts[1]}")
print(f"Max F-beta-like score: {best_fbeta}")
