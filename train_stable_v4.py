# -*- coding: utf-8 -*-
"""
5-Fold Cross Validation Training for CNNModelStable
Author: Yulin Wang
Date: 2025-10-27
"""

import argparse
import csv
import os
import glob
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from datasets.heatmap_dataset import HeatmapDataset
from utils.early_stopping import EarlyStopping
from config_preproc_train import Config
from models.cnn_model_stable import CNNModel

# -------------------------------
#  固定随机种子，保证可复现
# -------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# -------------------------------
#  参数配置
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, required=True,
                    help='Preprocessing version: baseline | pd_sensitive | smooth')
parser.add_argument('--exp_mode', type=str, default="All",
                    choices=["Ga", "Ju", "Si", "All"],
                    help="Experiment mode: Ga | Ju | Si | All")
parser.add_argument('--side', type=str, default='both',
                    choices=['left', 'right', 'both'],
                    help="Foot side: left | right | both")
args = parser.parse_args()

Config.initialize()
timestamp = time.strftime("%Y%m%d_%H%M", time.localtime())

# -------------------------------
#  读取最新标签文件
# -------------------------------
label_pattern = os.path.join(Config.BASE_DIR, f"labels_{args.version}_*.csv")
label_files = sorted(glob.glob(label_pattern), key=os.path.getmtime, reverse=True)
if not label_files:
    raise FileNotFoundError(f"❌ 未找到标签文件: {label_pattern}")

LABEL_CSV_PATH = label_files[0]
print(f"\n🧭 Using label file: {LABEL_CSV_PATH}")
df = pd.read_csv(LABEL_CSV_PATH)
df = df[df['filename'].str.contains('_01')]

# 模式筛选
if args.exp_mode != "All":
    df = df[df['filename'].str.contains(args.exp_mode)]
if args.side in ["left", "right"]:
    df = df[df['filename'].str.contains(f"_{args.side}_")]

# 输出样本数
n_co = df['filename'].str.contains('Co').sum()
n_pt = df['filename'].str.contains('Pt').sum()
print(f"✅ 模式={args.exp_mode} | 样本数: {len(df)} | Control={n_co} | Parkinson={n_pt}")

subset_csv = os.path.join(Config.BASE_DIR,
                          f"labels_{args.exp_mode}_{args.side}_{args.version}_{timestamp}.csv")
df.to_csv(subset_csv, index=False)

# -------------------------------
#  构建数据集
# -------------------------------
dataset = HeatmapDataset(subset_csv)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
#  5-Fold 交叉验证
# -------------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []
fold_idx = 1

for train_idx, val_idx in kf.split(np.arange(len(dataset))):
    print(f"\n===== Fold {fold_idx}/5 =====")

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=Config.BATCH_SIZE, shuffle=False)

    model = CNNModel(fc_size=256, dropout_p=0.3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    early_stopper = EarlyStopping(patience=Config.EARLY_STOPPING_PATIENCE, mode='max')

    best_val_acc = 0.0
    train_accs, val_accs = [], []
    train_losses, val_losses = [], []

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss, train_correct = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += (out.argmax(1) == y).sum().item()

        train_acc = train_correct / len(train_subset)
        train_loss /= len(train_loader)

        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item()
                val_correct += (out.argmax(1) == y).sum().item()

        val_acc = val_correct / len(val_subset)
        val_loss /= len(val_loader)

        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{Config.EPOCHS} | "
              f"Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(Config.CHECKPOINT_DIR, f"best_fold{fold_idx}.pth"))

        early_stopper(val_acc, model)
        if early_stopper.early_stop:
            print(f"🛑 Early stopping at epoch {epoch+1}")
            break

    fold_results.append(best_val_acc)
    plt.figure()
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.legend(); plt.title(f'Fold {fold_idx} Accuracy')
    plt.savefig(os.path.join(Config.CHECKPOINT_DIR, f"fold{fold_idx}_acc.png"))
    plt.close()

    fold_idx += 1

# -------------------------------
#  输出最终结果
# -------------------------------
mean_acc = np.mean(fold_results)
std_acc = np.std(fold_results)
print(f"\n✅ 5-Fold 平均验证准确率: {mean_acc:.4f} ± {std_acc:.4f}")
