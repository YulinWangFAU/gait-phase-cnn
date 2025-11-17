# -*- coding: utf-8 -*-
"""
Created on 2025/11/6 12:37

@author: Yulin Wang
@email: yulin.wang@fau.de
"""
# -*- coding: utf-8 -*-
"""
run_all_experiments_v2_balanced.py
---------------------------------
Batch run CNN training on *balanced* dataset (Co/Pt subject-wise split).
Results will be stored separately under results_all_balanced.
"""

import os, subprocess

# ======== PATH CONFIGURATION ========
BASE_DIR = "/home/woody/iwi5/iwi5325h/gaitphasecnn_middle_data_balanced_g10_i4000"
SCRIPT_PATH = "/home/hpc/iwi5/iwi5325h/gait-phase-cnn/scripts/cnn_phaseplot_model_old.py"
RESULTS_BASE = "/home/hpc/iwi5/iwi5325h/gait-phase-cnn/results_all_balanced_g10_i4000"
LOG_DIR = os.path.join(RESULTS_BASE, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

success, fail = [], []
csv_files = sorted([f for f in os.listdir(BASE_DIR) if f.endswith(".csv")])

for csv_name in csv_files:
    csv_path = os.path.join(BASE_DIR, csv_name)
    img_dir = os.path.join(BASE_DIR, csv_name.replace("labels_", "heatmaps_").replace(".csv", ""))
    if not os.path.exists(img_dir):
        print(f"⚠️ Skip {csv_name} — no matching heatmap dir.")
        continue

    exp_name = img_dir.split("/")[-1]
    print(f"\n🚀 Running experiment: {exp_name}")
    out_base = os.path.join(RESULTS_BASE, exp_name)
    os.makedirs(out_base, exist_ok=True)

    for mode in ["baseline", "balanced"]:
        out_dir = os.path.join(out_base, mode)
        log_path = os.path.join(LOG_DIR, f"{exp_name}_{mode}.log")

        # 如果 baseline 已完成且有 best_model.pt，跳过
        if mode == "baseline" and os.path.exists(os.path.join(out_dir, "best_model.pt")):
            print(f"✅ Skip baseline ({exp_name}) — already done.")
            continue

        os.makedirs(out_dir, exist_ok=True)
        print(f"▶️  [{mode.upper()}] Running...")

        result = subprocess.run(
            ["python", SCRIPT_PATH, "--csv", csv_path, "--img", img_dir, "--out", out_dir, "--mode", mode],
            capture_output=True, text=True
        )

        # 保存日志文件
        with open(log_path, "w") as f:
            f.write(result.stdout)
            f.write("\n\n--- STDERR ---\n")
            f.write(result.stderr)

        # 检查是否出错
        if result.returncode == 0 and os.path.exists(os.path.join(out_dir, "report.txt")):
            print(f"✅ {exp_name} ({mode}) finished successfully.")
            success.append(f"{exp_name}/{mode}")
        else:
            print(f"❌ {exp_name} ({mode}) failed. Check log: {log_path}")
            fail.append(f"{exp_name}/{mode}")

# Summary
print("\n" + "=" * 50)
print("📊 SUMMARY:")
print(f"✅ Success ({len(success)}):")
for s in success: print("   -", s)
print(f"\n❌ Failed ({len(fail)}):")
for fitem in fail: print("   -", fitem)
print("=" * 50)
print("All experiments completed.")
