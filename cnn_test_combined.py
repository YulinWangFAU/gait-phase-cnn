# -*- coding: utf-8 -*-
"""
Created on 2025/10/16 16:19

@author: Yulin Wang
@email: yulin.wang@fau.de
"""

# -*- coding: utf-8 -*-
"""
Evaluate combined CNN models (All_01, All_02)
Generates accuracy summary and barplot
"""

import os
import csv
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from models.cnn_model_paper import CNNModel
from datasets.heatmap_dataset import HeatmapDataset
from config import Config

# === 参数设置 ===
conditions = ["_01", "_02"]     # 可扩展
fc_sizes = [128, 256, 512]
group = "All"                   # ✅ 固定为合并组

# === 结果文件 ===
results_path = os.path.join(Config.CHECKPOINT_DIR, "test_results_summary_combined.csv")
with open(results_path, "w", newline="") as f:
    csv.writer(f).writerow(["Group", "Condition", "fc_size", "Test Acc", "Test Loss"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.CrossEntropyLoss()

# === 主循环 ===
for c in conditions:
    for fc_size in fc_sizes:
        model_path = os.path.join(Config.CHECKPOINT_DIR, f"best_{group}{c}_fc{fc_size}.pth")
        label_csv = os.path.join(Config.BASE_DIR, f"labels_{group}{c}.csv")

        if not os.path.exists(model_path) or not os.path.exists(label_csv):
            print(f"⚠️ Skipping {group}{c} fc={fc_size}: model or labels not found.")
            continue

        print(f"\n🚀 Testing model: {os.path.basename(model_path)}")

        dataset = HeatmapDataset(label_csv)
        total_size = len(dataset)
        test_ratio = 0.15
        val_ratio = 0.15
        test_size = int(total_size * test_ratio)
        val_size = int(total_size * val_ratio)
        train_size = total_size - val_size - test_size
        _, _, test_ds = random_split(dataset, [train_size, val_size, test_size])
        test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

        model = CNNModel(fc_size=fc_size).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

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
plt.title("Combined Models Test Accuracy across FC Sizes")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(Config.CHECKPOINT_DIR, "test_accuracy_barplot_combined.png"))
plt.close()

print("\n🎯 Evaluation completed. Results saved to:")
print(results_path)
