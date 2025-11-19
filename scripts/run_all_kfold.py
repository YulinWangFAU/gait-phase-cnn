# -*- coding: utf-8 -*-
"""
Created on 2025/11/19 15:19

@author: Yulin Wang
@email: yulin.wang@fau.de
"""

# -*- coding: utf-8 -*-
"""
run_all_kfold.py
--------------------------------------------
Batch runner for full K-fold experiments
"""

import os
import subprocess

ROOT = "/home/woody/iwi5/iwi5325h/gaitphasecnn_middle_data_kfold"
SCRIPT = "/home/hpc/iwi5/iwi5325h/gait-phase-cnn/scripts/cnn_phaseplot_model_kfold.py"
RESULT_ROOT = "/home/hpc/iwi5/iwi5325h/gait-phase-cnn/results_kfold"

PARAM_GROUPS = [
    ("sigma8_i2000_kfold"),
    ("sigma10_i4000_kfold"),
    ("sigma12_i5000_kfold"),
]

METHOD_SIG = [
    ("rawphase_left"),
    ("rawphase_right"),
    ("rawphase_both"),
    ("tfs_left"),
    ("tfs_right"),
    ("tfs_both"),
]

FC_SIZES = [128, 256, 512]
MODES = ["baseline", "balanced"]

K = 10

os.makedirs(RESULT_ROOT, exist_ok=True)


def run_cmd(cmd, log_file):
    res = subprocess.run(cmd, capture_output=True, text=True)
    with open(log_file, "w") as f:
        f.write(res.stdout)
        f.write("\n--- STDERR ---\n")
        f.write(res.stderr)


for group in PARAM_GROUPS:
    group_path = os.path.join(ROOT, group)

    for ms in METHOD_SIG:
        ms_path = os.path.join(group_path, ms)

        if not os.path.exists(ms_path):
            continue

        for fold in range(K):
            fold_dir = os.path.join(ms_path, f"fold{fold}")

            train_csv = os.path.join(fold_dir, "train.csv")
            val_csv   = os.path.join(fold_dir, "val.csv")
            test_csv  = os.path.join(fold_dir, "test.csv")

            for fc in FC_SIZES:
                for mode in MODES:

                    out_dir = os.path.join(
                        RESULT_ROOT,
                        group,
                        ms,
                        f"fold{fold}",
                        f"{mode}_fc{fc}"
                    )
                    os.makedirs(out_dir, exist_ok=True)

                    log_file = os.path.join(out_dir, "log.txt")

                    if os.path.exists(os.path.join(out_dir, "best_model.pt")):
                        print(f"✔ Skip {out_dir} (already done)")
                        continue

                    cmd = [
                        "python", SCRIPT,
                        "--train_csv", train_csv,
                        "--val_csv", val_csv,
                        "--test_csv", test_csv,
                        "--out", out_dir,
                        "--mode", mode,
                        "--fc_dim", str(fc),
                    ]

                    print(f"🚀 Running: {out_dir}")
                    run_cmd(cmd, log_file)

print("🎉 ALL K-fold experiments completed.")
