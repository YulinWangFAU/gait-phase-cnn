# -*- coding: utf-8 -*-
"""
Created on 2025/6/30 19:54
把原始 txt 数据整理成 CSV 索引文件，供后面 Dataset 使用
写入 CSV 文件（比如 index_ga.csv）
@author: Yulin Wang
@email: yulin.wang@fau.de
"""

import os
import pandas as pd

data_dir = "/Users/wangyulin/Time Series/629/gait-phase-cnn/data/raw"
output_dir = "/Users/wangyulin/Time Series/629/gait-phase-cnn/data"

def generate_index(prefix, output_filename, keep_trials):
    records = []

    for fname in os.listdir(data_dir):
        if fname.startswith(prefix) and fname.endswith(".txt"):
            parts = fname.replace(".txt", "").split("_")
            subject_id = parts[0]
            trial = parts[1]

            if trial not in keep_trials:
                continue

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
    output_path = os.path.join(output_dir, output_filename)
    df.to_csv(output_path, index=False)
    print(f"✅ {output_filename} saved to: {output_path}")


if __name__ == "__main__":
    generate_index("Ga", "index_ga.csv", keep_trials=["01", "02"])
    generate_index("Ju", "index_ju.csv", keep_trials=["01"])
    generate_index("Si", "index_si.csv", keep_trials=["01"])
