import os, subprocess

BASE_DIR = "/home/woody/iwi5/iwi5325h/gaitphasecnn_middle_data"
SCRIPT_PATH = "/home/hpc/iwi5/iwi5325h/gait-phase-cnn/scripts/cnn_training_full_eval.py"
RESULTS_BASE = "/home/hpc/iwi5/iwi5325h/gait-phase-cnn/results_all"
os.makedirs(RESULTS_BASE, exist_ok=True)

csv_files = sorted([f for f in os.listdir(BASE_DIR) if f.endswith(".csv")])

for csv_name in csv_files:
    csv_path = os.path.join(BASE_DIR, csv_name)
    img_dir = os.path.join(BASE_DIR, csv_name.replace("labels_", "heatmaps_").replace(".csv", ""))
    if not os.path.exists(img_dir): continue

    exp_name = img_dir.split("/")[-1]
    print(f"\n🚀 Running {exp_name}")
    out_base = os.path.join(RESULTS_BASE, exp_name)
    os.makedirs(out_base, exist_ok=True)

    # 版本1：Baseline
    out1 = os.path.join(out_base, "baseline")
    subprocess.run(["python", SCRIPT_PATH, "--csv", csv_path, "--img", img_dir, "--out", out1, "--mode", "baseline"])

    # 版本2：Balanced
    out2 = os.path.join(out_base, "balanced")
    subprocess.run(["python", SCRIPT_PATH, "--csv", csv_path, "--img", img_dir, "--out", out2, "--mode", "balanced"])

print("\n✅ All experiments completed!")
