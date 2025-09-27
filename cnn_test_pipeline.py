# -*- coding: utf-8 -*-
"""
CNN Test Pipeline
评估训练好的 CNN 模型在测试集上的表现
输出指标: Accuracy, Loss, AUC, ROC 曲线
@author: Yulin Wang
@email: yulin.wang@fau.de
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
import os
import importlib  # ✅ 新增

from datasets.heatmap_dataset import HeatmapDataset
from config import Config

# === 动态加载模型 ✅ ===
model_module = importlib.import_module(f"models.{Config.MODEL_NAME}")
CNNModel = getattr(model_module, "CNNModel")

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Dataset ===
dataset = HeatmapDataset(Config.LABEL_CSV_PATH)

# 使用相同的比例划分 (70/15/15)
test_size = int(len(dataset) * Config.TEST_SPLIT)
val_size = int(len(dataset) * Config.VAL_SPLIT)
train_size = len(dataset) - val_size - test_size
_, _, test_ds = random_split(dataset, [train_size, val_size, test_size])

test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

# === Load Model ===
model = CNNModel().to(device)
model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
model.eval()

criterion = nn.CrossEntropyLoss()

# === Evaluation ===
all_labels = []
all_preds = []
all_probs = []
test_loss = 0.0

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        loss = criterion(outputs, y)
        test_loss += loss.item()

        probs = torch.softmax(outputs, dim=1)[:, 1]  # 取正类概率
        preds = outputs.argmax(dim=1)

        all_labels.extend(y.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# === Metrics ===
test_loss_avg = test_loss / len(test_loader)
test_acc = accuracy_score(all_labels, all_preds)
test_auc = roc_auc_score(all_labels, all_probs)

print(f"\n🎯 Test Results:")
print(f"Accuracy: {test_acc:.4f}")
print(f"Loss: {test_loss_avg:.6f}")
print(f"AUC: {test_auc:.4f}")

# === ROC Curve ===
fpr, tpr, thresholds = roc_curve(all_labels, all_probs)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AUC = {test_auc:.4f}")
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Test Set)")
plt.legend(loc="lower right")
plt.grid(True)

roc_path = os.path.join(Config.CHECKPOINT_DIR, "roc_curve.png")
plt.savefig(roc_path)
plt.show()

print(f"✅ ROC curve saved to {roc_path}")
