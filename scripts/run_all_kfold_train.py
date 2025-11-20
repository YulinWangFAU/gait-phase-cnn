# -*- coding: utf-8 -*-
"""
run_all_kfold_train.py (Final Version)
--------------------------------------------
Separates input heatmap folders from output result folders.

INPUT:
  /home/woody/iwi5/iwi5325h/gaitphasecnn_middle_data_kfold_experiments/

OUTPUT:
  /home/woody/iwi5/iwi5325h/gaitphasecnn_results_kfold/
"""

import os
import subprocess

# -------------------------------
# PATHS
# -------------------------------
INPUT_BASE = "/home/woody/iwi5/iwi5325h/gaitphasecnn_middle_data_kfold_experiments"
OUTPUT_BASE = "/home/woody/iwi5/iwi5325h/gaitphasecnn_results_kfold"
SCRIPT = "/home/hpc/iwi5/iwi5325h/gait-phase-cnn/scripts/cnn_phaseplot_model_kfold.py"

os.makedirs(OUTPUT_BASE, exist_ok=True)

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


def run_cmd(cmd, out_dir):
    print("\n============================================")
    print("EXEC:", " ".join(cmd))
    print("============================================\n")

    log_file = os.path.join(out_dir, "stdout_stderr.log")

    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    with open(log_file, "w") as f:
        for line in p.stdout:
            print(line, end="", flush=True)
            f.write(line)

    p.wait()




def main():

    for exp in EXPERIMENTS:
        print(f"\n\n#################################################")
        print(f"🔥 Running experiment: {exp}")
        print("#################################################\n")

        exp_in = os.path.join(INPUT_BASE, exp)
        exp_out = os.path.join(OUTPUT_BASE, exp)
        os.makedirs(exp_out, exist_ok=True)

        for (folder_name, SIGMA, INTERP) in PARAM_GROUPS:

            param_in = os.path.join(exp_in, folder_name)
            param_out = os.path.join(exp_out, folder_name.replace("_kfold", ""))
            os.makedirs(param_out, exist_ok=True)

            print(f"\n===== PARAM GROUP {folder_name} =====")

            for method in METHODS:

                method_in = os.path.join(param_in, method)
                folds_dir = os.path.join(method_in, "folds")

                if not os.path.exists(folds_dir):
                    print(f"❌ Skip {method_in} (no folds directory)")
                    continue

                print(f"\n----- Method: {method} -----")

                for fold in range(10):
                    fold_dir = os.path.join(folds_dir, f"fold{fold}")
                    train_csv = os.path.join(fold_dir, "train.csv")
                    val_csv = os.path.join(fold_dir, "val.csv")
                    test_csv = os.path.join(fold_dir, "test.csv")

                    if not (os.path.exists(train_csv) and os.path.exists(val_csv) and os.path.exists(test_csv)):
                        print(f"⚠️ Missing CSVs in fold {fold}, skipping")
                        continue

                    for fc_dim in FC_SIZES:

                        out_dir = os.path.join(
                            param_out,
                            method,
                            f"fc{fc_dim}",
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
