# -*- coding: utf-8 -*-
"""
Created on 2025/11/6 21:55

@author: Yulin Wang
@email: yulin.wang@fau.de
"""
import os
import re
import pandas as pd

# ======== CONFIGURATION ========
BASE_DIR = "/home/hpc/iwi5/iwi5325h/gait-phase-cnn/results_all"
OUT_PATH = os.path.join(BASE_DIR, "results_summary.csv")

rows = []

# ======== MAIN LOOP ========
for root, dirs, files in os.walk(BASE_DIR):
    # 跳过 logs 文件夹
    if "logs" in root:
        continue

    # 只处理有 report.txt 的目录
    if "report.txt" not in files:
        continue

    report_path = os.path.join(root, "report.txt")

    # 自动识别 experiment 名称和模式（baseline / balanced）
    parts = root.split("/")
    exp_name = parts[-2] if parts[-1] in ["balanced", "baseline"] else parts[-1]
    mode = parts[-1] if parts[-1] in ["balanced", "baseline"] else "unknown"

    metrics = {
        "experiment": exp_name,
        "mode": mode,
        "acc": None,
        "auc": None,
        "prec_Co": None,
        "rec_Co": None,
        "f1_Co": None,
        "prec_Pt": None,
        "rec_Pt": None,
        "f1_Pt": None
    }

    # ======== Parse report.txt ========
    with open(report_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        # Co 行
        if re.match(r"\s*Co\s+", line):
            parts = re.split(r"\s+", line.strip())
            if len(parts) >= 4:
                metrics["prec_Co"] = float(parts[1])
                metrics["rec_Co"] = float(parts[2])
                metrics["f1_Co"] = float(parts[3])

        # Pt 行
        elif re.match(r"\s*Pt\s+", line):
            parts = re.split(r"\s+", line.strip())
            if len(parts) >= 4:
                metrics["prec_Pt"] = float(parts[1])
                metrics["rec_Pt"] = float(parts[2])
                metrics["f1_Pt"] = float(parts[3])

        # Accuracy 行
        elif re.search(r"accuracy", line):
            try:
                # 一般格式: "accuracy                           0.23        13"
                metrics["acc"] = float(line.strip().split()[1])
            except Exception:
                pass

        # AUC 行
        elif "AUC" in line:
            try:
                metrics["auc"] = float(line.strip().split(":")[-1])
            except Exception:
                pass

    rows.append(metrics)

# ======== Save Results ========
if not rows:
    print("⚠️ No valid report.txt files found!")
else:
    df = pd.DataFrame(rows)
    df = df.sort_values(["experiment", "mode"])
    df.to_csv(OUT_PATH, index=False)

    print(f"\n✅ Summary saved to: {OUT_PATH}\n")
    print(df.round(3))
