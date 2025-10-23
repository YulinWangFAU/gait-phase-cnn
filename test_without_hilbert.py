# -*- coding: utf-8 -*-
"""
Evaluate combined CNN models (All_01)
Version: WITHOUT Hilbert transform
All paths, files, and outputs are marked with "_nohilbert"
"""

import os
import csv
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from models.cnn_model_paper import CNNModel
from datasets.heatmap_dataset import HeatmapDataset
from config import Config

# === 参数设置 ===
conditions = ["_01"]     # 当前只训练了 _01
fc_sizes = [128, 256, 512]
group = "All"            # 合并组（Ga + Ju + Si）

# === 当前版本信息 ===
print(f"\n🧭 Evaluating models (NO Hilbert) for version: {Config.VERSION_TAG}_nohilbert")
#CHECKPOINT_DIR = Config.CHECKPOINT_DIR + "_nohilbert" 需要修改
CHECKPOINT_DIR = "/home/woody/iwi5/iwi5325h/gaitphasecnn_raw_data/checkpoints/hilbert_tfs_cnn_i1500_s6_fullsignal_20251023_2245/cnn_model_paper_nohilbert"

print(f"📁 Base dir: {Config.BASE_DIR}")
print(f"📁 Checkpoints dir: {CHECKPOINT_DIR}")

# === 结果文件 ===
results_path = os.path.join(CHECKPOINT_DIR, f"test_results_summary_combined_{Config.VERSION_TAG}_nohilbert.csv")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
with open(results_path, "w", newline="") as f:
    csv.writer(f).writerow(["Group", "Condition", "fc_size", "Test Acc", "Test Loss"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.CrossEntropyLoss()

# === 主循环 ===
for c in conditions:
    for fc_size in fc_sizes:
        # === 模型路径 ===
        model_path = os.path.join(
            CHECKPOINT_DIR,
            f"best_{group}{c}_fc{fc_size}_{Config.VERSION_TAG}_nohilbert.pth"
        )

        # === 标签路径 ===
        label_csv = os.path.join(
            Config.BASE_DIR,
            f"labels_{group}{c}_fullsignal_{Config.VERSION_TAG}_nohilbert.csv"
        )

        # === 文件存在检查 ===
        if not os.path.exists(model_path):
            print(f"⚠️ Model not found: {model_path}")
            continue
        if not os.path.exists(label_csv):
            print(f"⚠️ Label file not found: {label_csv}")
            continue

        print(f"\n🚀 Testing model (NO Hilbert): {os.path.basename(model_path)}")
        print(f"📄 Using label file: {os.path.basename(label_csv)}")

        # === 加载数据 ===
        dataset = HeatmapDataset(label_csv)
        total_size = len(dataset)

        # === 加载与训练一致的 test split ===
        test_idx_path = os.path.join(CHECKPOINT_DIR, f"test_indices_{Config.VERSION_TAG}_nohilbert.pt")
        if not os.path.exists(test_idx_path):
            print(f"⚠️ Test indices file not found: {test_idx_path}")
            continue
        test_indices = torch.load(test_idx_path)
        test_ds = Subset(dataset, test_indices)
        test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

        # === 加载模型 ===
        model = CNNModel(fc_size=fc_size).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # === 测试循环 ===
        test_loss, test_correct = 0.0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = criterion(output, y)
                test_loss += loss.item()
                test_correct += (output.argmax(dim=1) == y).sum().item()

        test_acc = test_correct / len(test_ds)
        test_loss_avg = test_loss / len(test_loader)

        print(f"✅ {group}{c} fc={fc_size} | Test Acc: {test_acc:.4f} | Test Loss: {test_loss_avg:.4f}")

        # === 写入结果 ===
        with open(results_path, "a", newline="") as f:
            csv.writer(f).writerow([group, c, fc_size, f"{test_acc:.4f}", f"{test_loss_avg:.6f}"])

# === 绘制测试准确率柱状图 ===
df = pd.read_csv(results_path)
plt.figure(figsize=(7, 5))
for c in conditions:
    subset = df[df["Condition"] == c]
    plt.bar(subset["fc_size"].astype(str) + c, subset["Test Acc"], label=f"All{c}")

plt.ylabel("Test Accuracy")
plt.xlabel("Model (fc_size_condition)")
plt.title(f"Combined Models Test Accuracy ({Config.VERSION_TAG}_nohilbert)")
plt.legend()
plt.tight_layout()

plot_path = os.path.join(CHECKPOINT_DIR, f"test_accuracy_barplot_combined_{Config.VERSION_TAG}_nohilbert.png")
plt.savefig(plot_path)
plt.close()

print("\n🎯 Evaluation completed (NO Hilbert).")
print(f"📊 Results saved to: {results_path}")
print(f"📈 Plot saved to:   {plot_path}\n")
