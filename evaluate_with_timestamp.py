# -*- coding: utf-8 -*-
"""
Evaluate trained CNN model on gait-phase heatmaps (PD vs HC)
Automatically loads latest or specified timestamp model
and evaluates on the held-out test set.

Author: Yulin Wang
Date: 2025-10-27
"""

import argparse
import os
import glob
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve, accuracy_score

from datasets.heatmap_dataset import HeatmapDataset
from config_preproc_train import Config
from models.cnn_model_paper import CNNModel

# === argparse 参数 ===
parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, required=True,
                    help='Preprocessing version: baseline | pd_sensitive | smooth')
parser.add_argument('--exp_mode', type=str, default="All",
                    choices=["Ga", "Ju", "Si", "All"],
                    help="Experiment mode: Ga_01 | Ju_01 | Si_01 | All_01 (combined)")
parser.add_argument('--fc', type=int, default=256, help='Fully-connected layer size')
parser.add_argument('--timestamp', type=str, default=None,
                    help='Optional timestamp to load specific model (e.g., 20251027_1720)')
args = parser.parse_args()

# === 初始化配置 ===
Config.initialize()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 搜索模型文件 ===
pattern = f"best_{args.exp_mode}_01_fc{args.fc}_{args.version}_*.pth"
model_files = sorted(glob.glob(os.path.join(Config.CHECKPOINT_DIR, pattern)), key=os.path.getmtime)

if not model_files:
    raise FileNotFoundError(f"❌ 未找到匹配的模型文件: {pattern}")

if args.timestamp:
    matched = [f for f in model_files if args.timestamp in f]
    if not matched:
        raise FileNotFoundError(f"❌ 未找到带时间戳 {args.timestamp} 的模型文件。")
    model_path = matched[-1]
else:
    model_path = model_files[-1]  # 加载最新的
print(f"\n🧭 Loading model: {model_path}")

# === 加载模型 ===
model = CNNModel(fc_size=args.fc).to(device)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# === 确定对应 CSV（自动匹配同时间戳）
csv_pattern = f"labels_{args.exp_mode}_01_{args.version}_*.csv"
csv_files = sorted(glob.glob(os.path.join(Config.BASE_DIR, csv_pattern)), key=os.path.getmtime)
if not csv_files:
    raise FileNotFoundError(f"❌ 未找到匹配的标签文件: {csv_pattern}")
csv_path = csv_files[-1]
print(f"📄 Using label file: {csv_path}")

# === 构建数据集（完整，然后取测试集）
dataset = HeatmapDataset(csv_path)
total_size = len(dataset)
val_size = int(total_size * Config.VAL_SPLIT)
test_size = int(total_size * Config.TEST_SPLIT)
train_size = total_size - val_size - test_size

# 读取之前保存的 test indices
test_idx_path = os.path.join(Config.CHECKPOINT_DIR, f"test_indices_{args.exp_mode}_{args.version}.pt")
if not os.path.exists(test_idx_path):
    raise FileNotFoundError(f"❌ 找不到测试集索引文件: {test_idx_path}\n请先运行训练脚本生成。")

test_indices = torch.load(test_idx_path)
test_ds = torch.utils.data.Subset(dataset, test_indices)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

# === 评估 ===
y_true, y_pred, y_prob = [], [], []

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = outputs.argmax(dim=1)
        y_true.extend(y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_prob.extend(probs.cpu().numpy())

acc = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)
auc = roc_auc_score(y_true, y_prob)

print(f"\n✅ Test Accuracy: {acc:.4f}")
print(f"✅ ROC-AUC: {auc:.4f}")
print("✅ Confusion Matrix:\n", cm)

# === 保存结果 ===
save_dir = os.path.join(Config.CHECKPOINT_DIR, "evaluation_results")
os.makedirs(save_dir, exist_ok=True)

base_name = os.path.basename(model_path).replace(".pth", "")
cm_path = os.path.join(save_dir, f"{base_name}_cm.png")
roc_path = os.path.join(save_dir, f"{base_name}_roc.png")
txt_path = os.path.join(save_dir, f"{base_name}_metrics.txt")

# 混淆矩阵图
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["HC", "PD"])
disp.plot(cmap="Blues", values_format='d')
plt.title(f"Confusion Matrix ({args.exp_mode}_01)")
plt.savefig(cm_path)
plt.close()

# ROC 曲线
fpr, tpr, _ = roc_curve(y_true, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve ({args.exp_mode}_01)")
plt.legend()
plt.savefig(roc_path)
plt.close()

# 文本结果
with open(txt_path, 'w') as f:
    f.write(f"Model: {model_path}\n")
    f.write(f"Label file: {csv_path}\n")
    f.write(f"Test Accuracy: {acc:.4f}\n")
    f.write(f"ROC-AUC: {auc:.4f}\n")
    f.write(f"Confusion Matrix:\n{cm}\n")

print(f"\n📊 Evaluation results saved to:\n{save_dir}")
print(f"  ├─ Accuracy / AUC: {txt_path}")
print(f"  ├─ Confusion Matrix: {cm_path}")
print(f"  └─ ROC Curve: {roc_path}")
