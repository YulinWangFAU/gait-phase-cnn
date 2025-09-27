import os
import pandas as pd

# 项目路径
data_dir = "/Users/wangyulin/Time Series/629/gait-phase-cnn/data/raw"
output_csv = "/Users/wangyulin/Time Series/629/gait-phase-cnn/data/index_ga.csv"

records = []

for fname in os.listdir(data_dir):
    if fname.startswith("Ga") and fname.endswith(".txt"):
        parts = fname.replace(".txt", "").split("_")
        subject_id = parts[0]       # GaCo01 / GaPt01
        trial = parts[1]            # "01", "02", or "10"

        #  只保留 _01 和 _02 试次
        if trial not in ["01", "02"]:
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

# 保存 CSV
df = pd.DataFrame(records)
df.to_csv(output_csv, index=False)
print(f"✅ index_ga.csv saved to: {output_csv}")
