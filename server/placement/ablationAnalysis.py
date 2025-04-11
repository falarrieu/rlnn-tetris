import json
import pandas as pd
import matplotlib.pyplot as plt

# Load results
with open("ablation_results.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Replace 'none' with 'all' for clarity in plots
df["excluded"] = df["excluded"].replace({"none": "all"})

# Sort so 'all' is first
df = df.sort_values("excluded", key=lambda col: col != "all")

# Plot 1: Lines Cleared
plt.figure(figsize=(10, 5))
plt.bar(df["excluded"], df["avg_lines_cleared"])
plt.title("Ablation Study: Avg Lines Cleared")
plt.xlabel("Heuristic Removed")
plt.ylabel("Avg Lines Cleared")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("ablation_lines.png")


# Plot 2: Frames Survived
plt.figure(figsize=(10, 5))
plt.bar(df["excluded"], df["avg_frames_survived"])
plt.title("Ablation Study: Avg Frames Survived")
plt.xlabel("Heuristic Removed")
plt.ylabel("Avg Frames Survived")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("ablation_frames.png")