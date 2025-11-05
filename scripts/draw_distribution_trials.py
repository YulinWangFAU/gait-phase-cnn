# -*- coding: utf-8 -*-
"""
Enhanced version with per-bar labels and pastel (macaron) colors.
"""

import os
import collections
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "/Users/wangyulin/Time Series/629/gait-phase-cnn/data/raw"

if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"❌ Data directory not found: {DATA_DIR}")

# === 统计每个subject的trial数 ===
trial_counts = collections.defaultdict(int)
for f in os.listdir(DATA_DIR):
    if not f.endswith(".txt"):
        continue
    subj = f[:6]  # e.g., GaCo05, JuPt07
    trial_counts[subj] += 1

# === 实验分组 ===
groups = {"Ga": {"Co": [], "Pt": []},
          "Ju": {"Co": [], "Pt": []},
          "Si": {"Co": [], "Pt": []}}

for k, v in trial_counts.items():
    prefix = k[:2]
    group = "Co" if "Co" in k else "Pt"
    if prefix in groups:
        groups[prefix][group].append(v)

# === 绘图数据准备 ===
bins = np.arange(1, max(trial_counts.values()) + 1)
width = 0.12
gap = 0.3

# 马卡龙配色
colors = {
    "Ga": ("#A7C7E7", "#5B8FC0"),  # 蓝
    "Ju": ("#A9E4C0", "#4CA874"),  # 绿
    "Si": ("#F9D6A5", "#E39D6B")   # 橙
}

plt.figure(figsize=(10, 5))

for i, exp in enumerate(["Ga", "Ju", "Si"]):
    co_counts = [groups[exp]["Co"].count(b) for b in bins]
    pt_counts = [groups[exp]["Pt"].count(b) for b in bins]
    offset = i * (width * 3 + gap)

    # 绘制Control
    bars1 = plt.bar(bins + offset - width/2, co_counts, width=width,
                    color=colors[exp][0], edgecolor="gray",
                    label=f"{exp} Control")
    # 绘制Parkinson
    bars2 = plt.bar(bins + offset + width/2, pt_counts, width=width,
                    color=colors[exp][1], edgecolor="gray",
                    label=f"{exp} Parkinson")

    # 添加数值标签
    for bar in bars1 + bars2:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2, height + 0.8,
                     f"{int(height)}", ha='center', va='bottom', fontsize=8)

# === 坐标与装饰 ===
plt.xlabel("Number of Trials per Subject", fontsize=12)
plt.ylabel("Number of Subjects", fontsize=12)
plt.title("Distribution of Trials per Subject across Experiments and Groups", fontsize=13)
plt.xticks(bins + width, [str(b) for b in bins])
plt.grid(axis='y', linestyle='--', alpha=0.4)

# === 图例（六种颜色） ===
plt.legend(loc="upper right", ncol=2, fontsize=9, frameon=True)

plt.tight_layout()
plt.show()

