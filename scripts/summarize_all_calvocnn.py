# -*- coding: utf-8 -*-
"""
summarize_all_calvocnn.py
Batch summarize all CalvoCNN experiment folders (e.g. g8_i2000, g10_i4000, g12_i5000)
into individual CSVs and one combined summary.
"""

import os
import re
import pandas as pd

# ======== CONFIGURATION ========
ROOT_DIR = "/Users/wangyulin/Time Series/results_calvocnn_multi"
SUMMARY_ALL = os.path.join(ROOT_DIR, "results_calvocnn_all_summary.csv")


# ======== FUNCTION ========
def summarize_one(base_dir):
    """
    Summarize one experiment directory (e.g. results_calvocnn_g10_i4000)
    """
    rows = []

    for root, dirs, files in os.walk(base_dir):
        if "logs" in root:
            continue
        if "report.txt" not in files:
            continue

        report_path = os.path.join(root, "report.txt")

        # ---- Extract experiment identifiers ----
        parts = root.split("/")
        exp_name = parts[-2] if re.match(r"(balanced|baseline)_fc\d+", parts[-1]) else parts[-1]
        fc_block = parts[-1]
        match = re.search(r"(balanced|baseline)_fc(\d+)", fc_block)
        if match:
            mode, fc_dim = match.groups()
        else:
            mode, fc_dim = "unknown", "?"

        metrics = {
            "main_dir": os.path.basename(base_dir),
            "experiment": exp_name,
            "mode": mode,
            "fc_dim": fc_dim,
            "acc": None,
            "auc": None,
            "prec_Co": None,
            "rec_Co": None,
            "f1_Co": None,
            "prec_Pt": None,
            "rec_Pt": None,
            "f1_Pt": None
        }

        # ---- Parse report.txt ----
        with open(report_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            if re.match(r"\s*Co\s+", line):
                parts = re.split(r"\s+", line.strip())
                if len(parts) >= 4:
                    metrics["prec_Co"] = float(parts[1])
                    metrics["rec_Co"] = float(parts[2])
                    metrics["f1_Co"] = float(parts[3])

            elif re.match(r"\s*Pt\s+", line):
                parts = re.split(r"\s+", line.strip())
                if len(parts) >= 4:
                    metrics["prec_Pt"] = float(parts[1])
                    metrics["rec_Pt"] = float(parts[2])
                    metrics["f1_Pt"] = float(parts[3])

            elif re.search(r"accuracy", line, re.IGNORECASE):
                tokens = re.split(r"\s+", line.strip())
                for tok in tokens:
                    try:
                        val = float(tok)
                        if 0.0 <= val <= 1.0:
                            metrics["acc"] = val
                            break
                    except ValueError:
                        continue

            elif "AUC" in line:
                try:
                    metrics["auc"] = float(line.strip().split(":")[-1])
                except Exception:
                    pass

        rows.append(metrics)

    # ---- Save each experiment’s summary ----
    if not rows:
        print(f"⚠️ No valid report.txt files found in: {base_dir}")
        return None

    df = pd.DataFrame(rows)
    df = df.sort_values(["experiment", "mode", "fc_dim"])
    out_path = os.path.join(base_dir, f"{os.path.basename(base_dir)}_summary.csv")
    df.to_csv(out_path, index=False)
    print(f"✅ Saved summary: {out_path} ({len(df)} entries)")
    return df


# ======== MAIN LOOP ========
all_dfs = []
for d in sorted(os.listdir(ROOT_DIR)):
    subdir = os.path.join(ROOT_DIR, d)
    if os.path.isdir(subdir) and d.startswith("results_calvocnn_"):
        df = summarize_one(subdir)
        if df is not None:
            all_dfs.append(df)

# ======== Combine All ========
if all_dfs:
    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all.to_csv(SUMMARY_ALL, index=False)
    print(f"\n🎯 Combined summary saved to: {SUMMARY_ALL}\n")
    print(df_all.round(3))
else:
    print("⚠️ No experiment folders found under:", ROOT_DIR)
