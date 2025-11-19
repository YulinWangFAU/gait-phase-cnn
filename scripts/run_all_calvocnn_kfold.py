# -*- coding: utf-8 -*-
"""
Created on 2025/11/19 15:12

@author: Yulin Wang
@email: yulin.wang@fau.de
"""
# -*- coding: utf-8 -*-
"""
run_all_calvocnn_kfold.py
---------------------------------
Run CalvoCNN on K-fold (subject-wise) splits.
Supports:
    - 3 parameter groups: (σ8,i2000), (σ10,i4000), (σ12,i5000)
    - 6 heatmap types: rawphase/tfs × left/right/both
    - 10 folds per experiment
    - FC sizes: 128, 256, 512
    - modes: baseline / balanced
"""

import os
import subprocess

# ======== CONFIG ========
ROOT = "/home/woody/iwi5/iwi5325h/gaitphasecnn_middle_data_kfold"
SCRIPT = "/home/hpc/iwi5/iwi5325h/gait-phase-cnn/scripts/cnn_phaseplot_model_kfold.py"
RESULTS = "/home/hpc/iwi5/iwi5325h/gait-phase-cnn/results_kfold_all"
os.makedirs(RESULTS, exist_ok=True)

PARAM_GROUPS = [
    ("sigma8_i2000_kfold",  8,  2000),
    ("sigma10_i4000_kfold", 10, 4000),
    ("sigma12_i5000_kfold", 12, 5000),
]

METHODS = ["rawphase", "tfs"]
SIGNALS = ["left", "right", "both"]
FC_SIZES = [128, 256, 512]
MODES = ["baseline", "balanced"]
KFOLD = 10


# ======== START RUNNING ========
success, fail = [], []

print("\n==========================================")
print("      🚀 Running ALL K-FOLD EXPERIMENTS")
print("==========================================\n")

for group_name, sigma, interp in PARAM_GROUPS:

    GROUP_PATH = os.path.join(ROOT, group_name)
    META_PATH = os.path.join(GROUP_PATH, f"meta_{group_name}.csv")

    print(f"\n===== PARAM GROUP: {group_name} =====")

    for method in METHODS:
        for sig in SIGNALS:

            EXP_NAME = f"{method}_{sig}"
            EXP_ROOT = os.path.join(GROUP_PATH, EXP_NAME)

            # check existence
            if not os.path.exists(EXP_ROOT):
                print(f"⚠️ Skip (missing folder): {EXP_ROOT}")
                continue

            # iterate folds
            for fold in range(KFOLD):

                fold_df = os.path.join(
                    GROUP_PATH,
                    f"{group_name}_{EXP_NAME}_fold{fold}.csv"
                )
                if not os.path.exists(fold_df):
                    print(f"⚠️ Missing CSV: {fold_df}")
                    continue

                for fc in FC_SIZES:
                    for mode in MODES:

                        OUT_DIR = os.path.join(
                            RESULTS,
                            group_name,
                            EXP_NAME,
                            f"fold{fold}",
                            f"{mode}_fc{fc}"
                        )
                        os.makedirs(OUT_DIR, exist_ok=True)

                        # Skip if already trained
                        if os.path.exists(os.path.join(OUT_DIR, "best_model.pt")):
                            print(f"✔ Already done: {EXP_NAME} fold{fold} [{mode}, fc={fc}]")
                            continue

                        print(f"\n▶️ Running {EXP_NAME} | fold={fold} | {mode} | fc={fc}")

                        cmd = [
                            "python", SCRIPT,
                            "--csv", fold_df,
                            "--img", GROUP_PATH,
                            "--out", OUT_DIR,
                            "--mode", mode,
                            "--fc_dim", str(fc)
                        ]

                        result = subprocess.run(cmd, capture_output=True, text=True)

                        # log
                        LOG_DIR = os.path.join(OUT_DIR, "log.txt")
                        with open(LOG_DIR, "w") as f:
                            f.write(result.stdout)
                            f.write("\n--- stderr ---\n")
                            f.write(result.stderr)

                        # check
                        if result.returncode == 0 and os.path.exists(os.path.join(OUT_DIR, "report.txt")):
                            success.append(f"{EXP_NAME}_fold{fold}_{mode}_fc{fc}")
                        else:
                            fail.append(f"{EXP_NAME}_fold{fold}_{mode}_fc{fc}")


# ======== SUMMARY ========
print("\n==========================================")
print("                SUMMARY")
print("==========================================")
print(f"✅ Success ({len(success)})")
for s in success:
    print("   -", s)

print(f"\n❌ Failed ({len(fail)})")
for f in fail:
    print("   -", f)

print("==========================================")
