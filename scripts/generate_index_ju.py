# -*- coding: utf-8 -*-
"""
Created on 2025/6/30 19:38

@author: Yulin Wang
@email: yulin.wang@fau.de
"""

import os
import pandas as pd

data_dir = "/Users/wangyulin/Time Series/629/gait-phase-cnn/data/raw"
output_csv = "/Users/wangyulin/Time Series/629/gait-phase-cnn/data/index_ju.csv"

records = []

for fname in os.listdir(data_dir):
    if fname.startswith("Ju") and fname.endswith(".txt"):
        parts = fname.replace(".txt", "").split("_")
        subject_id = parts[0]  # JuPt01 / JuCo01
        trial = parts[1]       # "01", "02", etc.

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
print(f"✅ index_ju.csv saved to: {output_csv}")
