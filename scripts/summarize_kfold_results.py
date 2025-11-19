# -*- coding: utf-8 -*-
"""
Created on 2025/11/19 15:21

@author: Yulin Wang
@email: yulin.wang@fau.de
"""

# -*- coding: utf-8 -*-
"""
summarize_kfold_results.py
----------------------------------------------
Automatically summarize all 10-fold results:
- For each (sigma, method, signal, mode, fc_dim)
- Compute mean ACC, mean AUC, std ACC, std AUC
- Export a summary CSV for easy use in a paper

Directory expected:
results_kfold/
    sigmaX_iY_kfold/
        rawphase_left/
            fold0/baseline_fc128/report.txt
"""

import os
import re
import pandas as pd
import numpy as np


RESULT_ROOT = "/home/hpc/iwi5/iwi5325h/gait-phase-cnn/results_kfold"

OUT_CSV = os.path.join(RESULT_ROOT, "summary_kfold_results.csv")

# Regex to extract metrics from report.txt
ACC_RE = re.compile(r"accuracy\s*=\s*([0-9.]+)", re.IGNORECASE)
AUC_RE = re.compile(r"AUC[:=]\s*([0-9.]+)", re.IGNORECASE)


def extract_metrics(report_path):
    """Parse report.txt to find ACC and AUC."""
    if not os.path.exists(report_path):
        return None, None

    with open(report_path, "r") as f:
        txt = f.read()

    # Find accuracy (from classification report)
    acc = None
    m_acc = re.search(r"accuracy\s*[\s:]*([0-9.]+)", txt, flags=re.I)
    if m_acc:
        acc = float(m_acc.group(1))

    # Find AUC
    auc = None
    m_auc = AUC_RE.search(txt)
    if m_auc:
        auc = float(m_auc.group(1))

    return acc, auc


def main():
    rows = []

    for sigma_dir in sorted(os.listdir(RESULT_ROOT)):
        sigma_path = os.path.join(RESULT_ROOT, sigma_dir)
        if not os.path.isdir(sigma_path):
            continue

        for ms_dir in sorted(os.listdir(sigma_path)):
            ms_path = os.path.join(sigma_path, ms_dir)
            if not os.path.isdir(ms_path):
                continue

            method, signal = ms_dir.split("_")[0], ms_dir.split("_")[1]

            for fold_dir in sorted(os.listdir(ms_path)):
                fold_path = os.path.join(ms_path, fold_dir)
                if not fold_path.endswith("fold0") and not fold_path.endswith("fold1") \
                   and not fold_path.endswith("fold2") and not fold_path.endswith("fold3") \
                   and not fold_path.endswith("fold4") and not fold_path.endswith("fold5") \
                   and not fold_path.endswith("fold6") and not fold_path.endswith("fold7") \
                   and not fold_path.endswith("fold8") and not fold_path.endswith("fold9"):
                    continue

                for mode_fc in sorted(os.listdir(fold_path)):
                    exp_path = os.path.join(fold_path, mode_fc)
                    report_path = os.path.join(exp_path, "report.txt")

                    mode = mode_fc.split("_")[0]
                    fc_dim = mode_fc.split("_")[1].replace("fc", "")

                    acc, auc = extract_metrics(report_path)
                    if acc is None or auc is None:
                        print(f"⚠ Missing metrics: {report_path}")
                        continue

                    rows.append([
                        sigma_dir,
                        method,
                        signal,
                        mode,
                        fc_dim,
                        fold_dir,
                        acc,
                        auc
                    ])

    df = pd.DataFrame(rows, columns=[
        "sigma_group",
        "method",
        "signal",
        "mode",
        "fc_dim",
        "fold",
        "acc",
        "auc"
    ])

    # Aggregate for final summary
    summary = df.groupby(
        ["sigma_group", "method", "signal", "mode", "fc_dim"]
    ).agg({
        "acc": ["mean", "std"],
        "auc": ["mean", "std"]
    })

    summary.columns = ["acc_mean", "acc_std", "auc_mean", "auc_std"]
    summary = summary.reset_index()

    summary.to_csv(OUT_CSV, index=False)

    print("\n🎉 Summary complete!")
    print(f"📄 Saved to: {OUT_CSV}\n")
    print(summary.head())


if __name__ == "__main__":
    main()
