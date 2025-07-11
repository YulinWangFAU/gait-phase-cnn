# -*- coding: utf-8 -*-
"""
Created on 2025/6/30 19:39

@author: Yulin Wang
@email: yulin.wang@fau.de
"""

import os
import pandas as pd

data_dir = "/Users/wangyulin/Time Series/629/gait-phase-cnn/data/raw"
output_csv = "/Users/wangyulin/Time Series/629/gait-phase-cnn/data/index_si.csv"

records = []

for fname in os.listdir(data_dir):
    if fname.startswith("Si") and fname.endswith(".txt"):
        parts = fname.replace(".txt", "").split("_")
        subject_id = parts[0]
        trial = parts[1]

        if trial != "01":
            continue  # ✅ 仅保留 _01

        if "Co" in subject_id:
            group = "Co"
            label = 0
        elif "Pt" in subject_id:
            group = "Pt"
            label = 1
        else:
            continue

        records.append({
            "filename": fname,
            "subject": subject_id,
            "group": group,
            "trial": "_" + trial,
            "label": label
        })

df = pd.DataFrame(records)
df.to_csv(output_csv, index=False)
print(f"✅ index_si.csv saved to: {output_csv}")
