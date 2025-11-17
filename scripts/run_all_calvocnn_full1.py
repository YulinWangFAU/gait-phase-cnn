# -*- coding: utf-8 -*-
"""
Created on 2025/11/17 16:08

@author: Yulin Wang
@email: yulin.wang@fau.de
"""
# -*- coding: utf-8 -*-
"""
run_all_calvocnn_full.py
Batch CalvoCNN runner using FIXED split.json.
"""

import os, subprocess

BASE_ROOT = "/home/woody/iwi5/iwi5325h"
SCRIPT_PATH = "/home/hpc/iwi5/iwi5325h/gait-phase-cnn/scripts/cnn_phaseplot_model.py"
RESULTS_ROOT = "/home/hpc/iwi5/iwi5325h/gait-phase-cnn/results_calvocnn_multi"
os.makedirs(RESULTS_ROOT, exist_ok=True)

EXPERIMENT_GROUPS = [
    ("g8", 2000),
    ("g10", 4000),
    ("g12", 5000),
]

FC_SIZES = [128, 256, 512]
MODES = ["baseline", "balanced"]

success, fail = [], []

for g, interp in EXPERIMENT_GROUPS:

    BASE_DIR = f"{BASE_ROOT}/gaitphasecnn_middle_data_balanced_multiset"
    SPLIT_JSON = os.path.join(BASE_DIR, "subject_split.json")

    if not os.path.exists(SPLIT_JSON):
        print(f"❌ No split.json found in {BASE_DIR}")
        continue

    # match group by σ and i
    csv_files = sorted([
        f for f in os.listdir(BASE_DIR)
        if f.endswith(".csv") and f"_σ{g[1:]}_i{interp}_" in f
    ])

    for csv_name in csv_files:
        csv_path = os.path.join(BASE_DIR, csv_name)
        img_dir = os.path.join(
            BASE_DIR,
            csv_name.replace("labels_", "heatmaps_").replace(".csv", "")
        )

        exp_name = os.path.basename(img_dir)
        RESULTS_BASE = os.path.join(RESULTS_ROOT, f"results_{exp_name}")
        LOG_DIR = os.path.join(RESULTS_BASE, "logs")
        os.makedirs(LOG_DIR, exist_ok=True)

        for fc_dim in FC_SIZES:
            for mode in MODES:

                out_dir = os.path.join(RESULTS_BASE, f"{mode}_fc{fc_dim}")
                os.makedirs(out_dir, exist_ok=True)

                if os.path.exists(os.path.join(out_dir, "best_model.pt")):
                    print(f"✔ Already done: {exp_name} ({mode}, fc={fc_dim})")
                    continue

                print(f"\n▶ Training {exp_name} | mode={mode} | fc={fc_dim}")

                log_path = os.path.join(LOG_DIR, f"{exp_name}_{mode}_fc{fc_dim}.log")

                result = subprocess.run(
                    ["python", SCRIPT_PATH,
                     "--csv", csv_path,
                     "--img", img_dir,
                     "--out", out_dir,
                     "--split_json", SPLIT_JSON,
                     "--mode", mode,
                     "--fc_dim", str(fc_dim)],
                    capture_output=True, text=True
                )

                with open(log_path, "w") as f:
                    f.write(result.stdout)
                    f.write("\n\n--- STDERR ---\n")
                    f.write(result.stderr)

                if result.returncode == 0 and \
                   os.path.exists(os.path.join(out_dir, "report.txt")):
                    success.append(f"{exp_name}/{mode}_fc{fc_dim}")
                else:
                    fail.append(f"{exp_name}/{mode}_fc{fc_dim}")

# summary
print("SUMMARY")
print("SUCCESS:", success)
print("FAIL:", fail)
