# -*- coding: utf-8 -*-
"""
run_all_calvocnn_full.py
---------------------------------
Batch run Calvo-Ariza CNN on *all gait heatmap datasets*.
For each dataset:
  - Run both modes: baseline / balanced
  - Test FC layer sizes: 128 / 256 / 512
All results saved under results_calvocnn_all/.
"""

import os, subprocess

# ======== PATH CONFIGURATION ========
BASE_DIR = "/home/woody/iwi5/iwi5325h/gaitphasecnn_middle_data_balanced_g8_i2000"
SCRIPT_PATH = "/home/hpc/iwi5/iwi5325h/gait-phase-cnn/scripts/cnn_phaseplot_model.py"
RESULTS_BASE = "/home/hpc/iwi5/iwi5325h/gait-phase-cnn/results_calvocnn_all_g8_i2000"
LOG_DIR = os.path.join(RESULTS_BASE, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

success, fail = [], []

# 搜索所有 CSV 文件
csv_files = sorted([f for f in os.listdir(BASE_DIR) if f.endswith(".csv")])
print(f"🔍 Found {len(csv_files)} CSV label files.")

# ======== LOOP THROUGH ALL EXPERIMENTS ========
for csv_name in csv_files:
    csv_path = os.path.join(BASE_DIR, csv_name)
    img_dir = os.path.join(BASE_DIR, csv_name.replace("labels_", "heatmaps_").replace(".csv", ""))
    if not os.path.exists(img_dir):
        print(f"⚠️ Skip {csv_name} — no matching heatmap dir.")
        continue

    exp_name = img_dir.split("/")[-1]
    out_base = os.path.join(RESULTS_BASE, exp_name)
    os.makedirs(out_base, exist_ok=True)
    print(f"\n🚀 Running experiment: {exp_name}")

    for fc_dim in [128, 256, 512]:
        for mode in ["baseline", "balanced"]:
            out_dir = os.path.join(out_base, f"{mode}_fc{fc_dim}")
            os.makedirs(out_dir, exist_ok=True)
            log_path = os.path.join(LOG_DIR, f"{exp_name}_{mode}_fc{fc_dim}.log")

            # 如果结果已存在（例如 best_model.pt 存在），则跳过
            if os.path.exists(os.path.join(out_dir, "best_model.pt")):
                print(f"✅ Skip {exp_name} ({mode}, fc={fc_dim}) — already done.")
                continue

            print(f"▶️  [{mode.upper()} | FC={fc_dim}] Running...")

            result = subprocess.run(
                ["python", SCRIPT_PATH,
                 "--csv", csv_path,
                 "--img", img_dir,
                 "--out", out_dir,
                 "--mode", mode,
                 "--fc_dim", str(fc_dim)],
                capture_output=True, text=True
            )

            # 写入日志
            with open(log_path, "w") as f:
                f.write(result.stdout)
                f.write("\n\n--- STDERR ---\n")
                f.write(result.stderr)

            # 检查结果文件
            report_path = os.path.join(out_dir, "report.txt")
            if result.returncode == 0 and os.path.exists(report_path):
                print(f"✅ {exp_name} ({mode}, fc={fc_dim}) finished successfully.")
                success.append(f"{exp_name}/{mode}_fc{fc_dim}")
            else:
                print(f"❌ {exp_name} ({mode}, fc={fc_dim}) failed. Check log: {log_path}")
                fail.append(f"{exp_name}/{mode}_fc{fc_dim}")

# ======== SUMMARY ========
print("\n" + "=" * 50)
print("📊 SUMMARY:")
print(f"✅ Success ({len(success)}):")
for s in success: print("   -", s)
print(f"\n❌ Failed ({len(fail)}):")
for fitem in fail: print("   -", fitem)
print("=" * 50)
print("All CalvoCNN experiments completed.")
