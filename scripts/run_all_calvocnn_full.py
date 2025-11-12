# -*- coding: utf-8 -*-
"""
run_all_calvocnn_full.py
---------------------------------
Batch run Calvo-Ariza CNN for multiple σ (Gaussian) and interpolation (i) configurations.
Each subdirectory will contain all heatmap experiments (rawphase/tfs, left/right/both, Ga/Ju/Si, dual/normal)
Test both modes (baseline/balanced) and FC layer sizes (128/256/512).
"""

import os, subprocess

# ======== CONFIGURABLE PATHS ========
BASE_ROOT = "/home/woody/iwi5/iwi5325h"
SCRIPT_PATH = "/home/hpc/iwi5/iwi5325h/gait-phase-cnn/scripts/cnn_phaseplot_model.py"
RESULTS_ROOT = "/home/hpc/iwi5/iwi5325h/gait-phase-cnn/results_calvocnn_multi"
os.makedirs(RESULTS_ROOT, exist_ok=True)

# ======== EXPERIMENT CONFIG ========
EXPERIMENT_GROUPS = [
    ("g8", 2000),
    ("g10", 4000),
    ("g12", 5000),
]

# ======== TRAINING CONFIG ========
FC_SIZES = [128, 256, 512]
MODES = ["baseline", "balanced"]

# ======== MAIN LOOP ========
success, fail = [], []

for g, interp in EXPERIMENT_GROUPS:
    BASE_DIR = f"{BASE_ROOT}/gaitphasecnn_middle_data_balanced_{g}_i{interp}"
    if not os.path.exists(BASE_DIR):
        print(f"⚠️ Skip {BASE_DIR} (not found)")
        continue

    print(f"\n🚀 Running group: σ={g}, i={interp}")
    csv_files = sorted([f for f in os.listdir(BASE_DIR) if f.endswith(".csv")])
    if not csv_files:
        print(f"⚠️ No CSV files in {BASE_DIR}")
        continue

    RESULTS_BASE = os.path.join(RESULTS_ROOT, f"results_calvocnn_{g}_i{interp}")
    LOG_DIR = os.path.join(RESULTS_BASE, "logs")
    os.makedirs(LOG_DIR, exist_ok=True)

    for csv_name in csv_files:
        csv_path = os.path.join(BASE_DIR, csv_name)
        img_dir = os.path.join(BASE_DIR, csv_name.replace("labels_", "heatmaps_").replace(".csv", ""))
        if not os.path.exists(img_dir):
            print(f"⚠️ Skip {csv_name} — no matching heatmap dir.")
            continue

        exp_name = img_dir.split("/")[-1]
        out_base = os.path.join(RESULTS_BASE, exp_name)
        os.makedirs(out_base, exist_ok=True)
        print(f"\n▶️ Experiment: {exp_name}")

        for fc_dim in FC_SIZES:
            for mode in MODES:
                out_dir = os.path.join(out_base, f"{mode}_fc{fc_dim}")
                os.makedirs(out_dir, exist_ok=True)
                log_path = os.path.join(LOG_DIR, f"{exp_name}_{mode}_fc{fc_dim}.log")

                if os.path.exists(os.path.join(out_dir, "best_model.pt")):
                    print(f"✅ Skip ({mode}, fc={fc_dim}) — already done.")
                    continue

                print(f"⏳ [{mode.upper()} | FC={fc_dim}] Training...")

                result = subprocess.run(
                    ["python", SCRIPT_PATH,
                     "--csv", csv_path,
                     "--img", img_dir,
                     "--out", out_dir,
                     "--mode", mode,
                     "--fc_dim", str(fc_dim)],
                    capture_output=True, text=True
                )

                with open(log_path, "w") as f:
                    f.write(result.stdout)
                    f.write("\n\n--- STDERR ---\n")
                    f.write(result.stderr)

                report_path = os.path.join(out_dir, "report.txt")
                if result.returncode == 0 and os.path.exists(report_path):
                    print(f"✅ {exp_name} ({mode}, fc={fc_dim}) done.")
                    success.append(f"{g}_{interp}/{exp_name}/{mode}_fc{fc_dim}")
                else:
                    print(f"❌ {exp_name} ({mode}, fc={fc_dim}) failed.")
                    fail.append(f"{g}_{interp}/{exp_name}/{mode}_fc{fc_dim}")

# ======== SUMMARY ========
print("\n" + "=" * 60)
print("📊 SUMMARY:")
print(f"✅ Success ({len(success)}):")
for s in success: print("   -", s)
print(f"\n❌ Failed ({len(fail)}):")
for fitem in fail: print("   -", fitem)
print("=" * 60)
print("All CalvoCNN experiments completed across σ and i configurations.")
