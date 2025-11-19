# -*- coding: utf-8 -*-
"""
run_all_kfold_train.py
--------------------------------------------
Master launcher for:
- 4 experiments: GaNormal, GaDual, JuNormal, SiNormal
- 3 parameter groups: (8,2000), (10,4000), (12,5000)
- 6 methods: rawphase/tfs × left/right/both
- 10 folds per method
- 3 FC sizes: 128, 256, 512

Uses cnn_phaseplot_model_kfold.py
"""

import os
import subprocess

BASE = "/home/woody/iwi5/iwi5325h/gaitphasecnn_middle_data_kfold_experiments"
SCRIPT = "/home/hpc/iwi5/iwi5325h/gait-phase-cnn/scripts/cnn_phaseplot_model_kfold.py"

EXPERIMENTS = ["GaNormal", "GaDual", "JuNormal", "SiNormal"]

PARAM_GROUPS = [
    ("sigma8_i2000_kfold", 8, 2000),
    ("sigma10_i4000_kfold", 10, 4000),
    ("sigma12_i5000_kfold", 12, 5000),
]

METHODS = [
    "rawphase_left",
    "rawphase_right",
    "rawphase_both",
    "tfs_left",
    "tfs_right",
    "tfs_both"
]

FC_SIZES = [128, 256, 512]

def run_cmd(cmd):
    print("\n============================================")
    print("EXEC:", " ".join(cmd))
    print("============================================\n")
    subprocess.run(cmd)


def main():
    for exp in EXPERIMENTS:
        print(f"\n\n#################################################")
        print(f"🔥 Running experiment: {exp}")
        print("#################################################\n")

        exp_root = os.path.join(BASE, exp)

        for (folder_name, SIGMA, INTERP) in PARAM_GROUPS:
            param_root = os.path.join(exp_root, folder_name)

            print(f"\n===== PARAM GROUP {folder_name} =====")

            for method in METHODS:
                method_dir = os.path.join(param_root, method)
                folds_dir = os.path.join(method_dir, "folds")

                if not os.path.exists(folds_dir):
                    print(f"❌ Skip {method_dir} (no folds/ directory)")
                    continue

                print(f"\n----- Method: {method} -----")

                for fold in range(10):
                    fold_dir = os.path.join(folds_dir, f"fold{fold}")
                    train_csv = os.path.join(fold_dir, "train.csv")
                    val_csv   = os.path.join(fold_dir, "val.csv")
                    test_csv  = os.path.join(fold_dir, "test.csv")

                    if not (os.path.exists(train_csv) and os.path.exists(val_csv) and os.path.exists(test_csv)):
                        print(f"⚠️  Missing CSVs in fold {fold}, skipping...")
                        continue

                    for fc_dim in FC_SIZES:
                        out_dir = os.path.join(
                            method_dir,
                            f"results_fc{fc_dim}",
                            f"fold{fold}"
                        )
                        os.makedirs(out_dir, exist_ok=True)

                        cmd = [
                            "python", SCRIPT,
                            "--train_csv", train_csv,
                            "--val_csv", val_csv,
                            "--test_csv", test_csv,
                            "--out", out_dir,
                            "--mode", "baseline",
                            "--fc_dim", str(fc_dim)
                        ]

                        run_cmd(cmd)


if __name__ == "__main__":
    main()
