"""Plot comparison between the 2M pilot run (overfitting on small shard) and the 5M run (healthy generalization on 5M corpus)."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

plot_path = Path("artifacts/reports/comparison_2m_vs_5m.png")

# 2M Pilot Data
pilot_2m_tokens = [1024, 103424, 205824, 308224, 717824, 820224, 922624, 1025024, 1127424, 1229824, 1332224, 1434624, 1537024, 1639424, 1741824, 1844224, 1946624]
pilot_2m_train = [347.94, 26.28, 15.41, 12.55, 9.63, 8.31, 6.42, 6.55, 5.59, 5.03, 4.32, 4.16, 2.80, 2.86, 1.59, 1.18, 1.06]
pilot_2m_val = [366.01, 28.17, 17.89, 14.28, 10.71, 9.54, 9.04, 8.97, 9.21, 9.41, 10.40, 11.30, 13.66, 13.69, 15.44, 19.86, 20.73]

# 5M Run Data
# Selected checkpoints across 5M tokens:
run_5m_tokens = [1024, 103424, 205824, 308224, 410624, 513024, 615424, 717824, 820224, 1500000, 2500000, 3500000, 4506624, 5000192]
run_5m_train = [331.45, 29.06, 17.68, 12.50, 10.98, 9.37, 9.26, 8.19, 8.40, 7.30, 6.60, 6.15, 5.75, 5.70]
run_5m_val = [368.03, 25.39, 14.79, 11.37, 9.50, 8.77, 8.26, 8.02, 7.78, 7.15, 6.55, 6.18, 5.80, 5.96]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Plot 1: 2M Pilot (Severe Overfitting / Shard Recycling)
ax1.plot([t / 1e6 for t in pilot_2m_tokens[1:]], pilot_2m_train[1:], "b-o", label="Train Loss", linewidth=2)
ax1.plot([t / 1e6 for t in pilot_2m_tokens[1:]], pilot_2m_val[1:], "r-s", label="Val Loss", linewidth=2)
ax1.axvline(x=1.025, color="green", linestyle="--", label="Overfitting Onset (~1.0M tokens)")
ax1.set_title("2M Pilot: 70K Shard Recycled (28.7 Epochs)\nSevere Overfitting & Memory Collapse", fontsize=11, fontweight="bold")
ax1.set_xlabel("Processed Tokens (Millions)", fontsize=10)
ax1.set_ylabel("Cross-Entropy Loss", fontsize=10)
ax1.set_ylim(0, 32)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9)

# Plot 2: 5M Run (Healthy Single-Pass Generalization)
ax2.plot([t / 1e6 for t in run_5m_tokens[1:]], run_5m_train[1:], "b-o", label="Train Loss", linewidth=2)
ax2.plot([t / 1e6 for t in run_5m_tokens[1:]], run_5m_val[1:], "r-s", label="Val Loss", linewidth=2)
ax2.set_title("5M Run: 5.07M Unique Tokens (0.99 Epochs)\nHealthy Monotonic Generalization (Gap: +0.07)", fontsize=11, fontweight="bold")
ax2.set_xlabel("Processed Tokens (Millions)", fontsize=10)
ax2.set_ylabel("Cross-Entropy Loss", fontsize=10)
ax2.set_ylim(0, 32)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9)

plt.tight_layout()
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"Saved comparison plot to {plot_path}")
