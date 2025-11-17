# -*- coding: utf-8 -*-
"""
Final version: 3×2 subplot figure with 6 macaron colors and uniform x-axis (1–7)
"""

import os
import collections
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "/Users/wangyulin/Time Series/629/gait-phase-cnn/data/raw"

# Count trial numbers per subject
trial_counts = collections.defaultdict(int)
for f in os.listdir(DATA_DIR):
    if f.endswith(".txt"):
        subj = f.split("_")[0]    # subject ID such as GaCo05
        trial_counts[subj] += 1

# Group into Ga/Ju/Si × Co/Pt
groups = {"Ga": {"Co": [], "Pt": []},
          "Ju": {"Co": [], "Pt": []},
          "Si": {"Co": [], "Pt": []}}

for subj, n in trial_counts.items():
    exp = subj[:2]
    grp = "Co" if "Co" in subj else "Pt"
    if exp in groups:
        groups[exp][grp].append(n)

# 6 pastel colors (macaron)
colors = {
    ("Ga", "Co"): "#E8DFF5",
    ("Ga", "Pt"): "#C7B9E5",
    ("Ju", "Co"): "#F6D6E7",
    ("Ju", "Pt"): "#E9A6C9",
    ("Si", "Co"): "#D7EEF2",
    ("Si", "Pt"): "#A6C9D8"
}

experiments = ["Ga", "Ju", "Si"]
groups_order = ["Co", "Pt"]

fig, axes = plt.subplots(3, 2, figsize=(11, 12))

for row, exp in enumerate(experiments):
    for col, grp in enumerate(groups_order):
        ax = axes[row, col]

        data = groups[exp][grp]
        # Use fixed x-axis: 1–7
        x_values = list(range(1, 8))
        y_values = [data.count(x) for x in x_values]

        bars = ax.bar(x_values, y_values,
                      color=colors[(exp, grp)],
                      edgecolor="gray")

        # Label bars
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2,
                        h + 0.2,
                        str(int(h)),
                        ha="center", fontsize=9)

        ax.set_title(f"{exp} - {'Control' if grp=='Co' else 'Parkinson'}",
                     fontsize=12, pad=10)
        ax.set_xticks(range(1, 8))
        ax.set_xlabel("Trials per Subject")
        ax.set_ylabel("Number of Subjects")

plt.tight_layout()
plt.show()
