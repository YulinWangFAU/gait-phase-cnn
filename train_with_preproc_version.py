# -*- coding: utf-8 -*-
"""
Train CNN model on gait-phase heatmaps (PD vs HC)
Supports:
  - preprocessing version (baseline / pd_sensitive / smooth)
  - side-specific training (left / right / both)
  - experiment mode (Ga / Ju / Si / All)
Author: Yulin Wang
Date: 2025-10-27
"""

import argparse
import csv
import os
import glob
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from datasets.heatmap_dataset import HeatmapDataset
from utils.early_stopping import EarlyStopping
from config_preproc_train import Config
from models.cnn_model_paper import CNNModel

# === argparse 参数 ===
parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, required=True,
                    help='Preprocessing version: baseline | pd_sensitive | smooth')
parser.add_argument('--side', type=str, default='both',
                    choices=['left', 'right', 'both'],
                    help='Foot side: left | right | both')
parser.add_argument('--exp_mode', type=str, default="All",
                    choices=["Ga", "Ju", "Si", "All"],
                    help="Experiment mode: Ga_01 | Ju_01 | Si_01 | All_01 (combined)")
parser.add_argument('--win', type=int, default=0)
parser.add_argument('--step', type=int, default=0)
args = parser.parse_args()

# === 初始化配置 ===
Config.initialize()

# === 时间戳标识 ===
timestamp = time.strftime("%Y%m%d_%H%M", time.localtime())

# === 搜索匹配的标签文件 ===
label_pattern = os.path.join(Config.BASE_DIR, f"labels_{args.version}_*.csv")
label_files = sorted(glob.glob(label_pattern), key=os.path.getmtime, reverse=True)
if not label_files:
    raise FileNotFoundError(f"❌ 未找到匹配的标签文件: {label_pattern}\n请先运行 generate_heatmaps.py 生成该版本热图。")

LABEL_CSV_PATH = label_files[0]
print(f"\n🧭 Using label file: {LABEL_CSV_PATH}")

# === 加载标签文件 ===
full_df = pd.read_csv(LABEL_CSV_PATH)

# === 筛选 _01 条件 ===
df = full_df[full_df['filename'].str.contains(r'_01\.')]

# === 实验模式筛选（Ga / Ju / Si / All）===
mode = args.exp_mode
if mode == "Ga":
    df = df[df['filename'].str.contains(r'/Ga')]
elif mode == "Ju":
    df = df[df['filename'].str.contains(r'/Ju')]
elif mode == "Si":
    df = df[df['filename'].str.contains(r'/Si')]
elif mode == "All":
    df = df[df['filename'].str.contains(r'/(Ga|Ju|Si)')]

# === 左右脚筛选 ===
if args.side in ["left", "right"]:
    df = df[df['filename'].str.contains(f"_{args.side}_")]
    print(f"🦶 Training on {args.side.upper()} foot samples only.")
elif args.side == "both":
    print("🦶 Training on both feet combined dataset.")

if len(df) == 0:
    raise ValueError(f"❌ No samples found for mode={mode}, side={args.side}.")

# 🆕 === 统计 Co vs Pt 数量 ===
n_co = df['filename'].str.contains('Co').sum()
n_pt = df['filename'].str.contains('Pt').sum()
total = len(df)
print(f"✅ 当前模式: {mode}_01 | side={args.side} | 样本总数: {total}")
print(f"   ├── Control (Co): {n_co}")
print(f"   └── Parkinson (Pt): {n_pt}")

# === 自动版本标签 ===
Config.VERSION_TAG = f"{mode}_{args.side}_{args.version}"

# === 输出路径 ===
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(Config.TENSORBOARD_LOG_DIR, exist_ok=True)

# === 保存筛选后的临时 CSV ===
subset_csv = os.path.join(Config.BASE_DIR,
                          f"labels_{mode}_01_{args.side}_{args.version}_{timestamp}.csv")
df.to_csv(subset_csv, index=False)

# === 构建数据集与划分 ===
dataset = HeatmapDataset(subset_csv)
total_size = len(dataset)
val_size = int(total_size * Config.VAL_SPLIT)
test_size = int(total_size * Config.TEST_SPLIT)
train_size = total_size - val_size - test_size

train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

# === 模型、优化器等 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
fc_sizes = [128, 256, 512]

# === 主训练循环 ===
for fc_size in fc_sizes:
    print(f"\n🚀 Training mode={mode}_01 | side={args.side} | version={args.version} | fc_size={fc_size}")

    model = CNNModel(fc_size=fc_size).to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=Config.LEARNING_RATE,
                           weight_decay=Config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    # === 保存路径 ===
    model_tag = f"{mode}_01_fc{fc_size}_{args.side}_{args.version}_{timestamp}"
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, f"best_{model_tag}.pth")

    early_stopper = EarlyStopping(
        patience=Config.EARLY_STOPPING_PATIENCE,
        min_delta=Config.EARLY_STOPPING_DELTA,
        mode=Config.EARLY_STOPPING_MODE,
        path=best_model_path
    )

    # === 日志路径 ===
    log_dir = os.path.join(Config.TENSORBOARD_LOG_DIR,
                           f"{model_tag}_win{args.win}_step{args.step}")
    os.makedirs(log_dir, exist_ok=True)
    log_csv_path = os.path.join(log_dir, "training_log.csv")
    f_csv = open(log_csv_path, 'w', newline='', encoding='utf-8')
    writer_csv = csv.writer(f_csv)
    writer_csv.writerow(['epoch', 'train_acc', 'val_acc', 'train_loss', 'val_loss', 'lr'])

    # === 训练循环 ===
    best_val_acc = 0.0
    train_acc_list, val_acc_list, train_loss_list, val_loss_list = [], [], [], []

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss, train_correct = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += (output.argmax(dim=1) == y).sum().item()

        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = criterion(output, y)
                val_loss += loss.item()
                val_correct += (output.argmax(dim=1) == y).sum().item()

        train_acc = train_correct / len(train_ds)
        val_acc = val_correct / len(val_ds)
        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)

        print(f"Epoch {epoch + 1}/{Config.EPOCHS} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
              f"Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f}")

        writer_csv.writerow([
            f"{epoch + 1}",
            f"{train_acc:.6f}",
            f"{val_acc:.6f}",
            f"{train_loss_avg:.6f}",
            f"{val_loss_avg:.6f}",
            f"{optimizer.param_groups[0]['lr']:.6f}"
        ])
        f_csv.flush()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"🌟 New best model saved (Val Acc = {val_acc:.4f})")

        scheduler.step(val_acc)
        early_stopper(val_acc, model)
        if early_stopper.early_stop:
            print(f"🛑 Early stopping at epoch {epoch + 1}")
            break

    # === 保存曲线 ===
    plt.figure()
    plt.plot(train_acc_list, label='Train Acc')
    plt.plot(val_acc_list, label='Val Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    plt.title(f'Accuracy ({model_tag})')
    plt.savefig(os.path.join(log_dir, f"{model_tag}_acc.png"))
    plt.close()

    plt.figure()
    plt.plot(train_loss_list, label='Train Loss')
    plt.plot(val_loss_list, label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.title(f'Loss ({model_tag})')
    plt.savefig(os.path.join(log_dir, f"{model_tag}_loss.png"))
    plt.close()

    f_csv.close()
    print(f"✅ Training finished for {model_tag}. Best Val Acc: {best_val_acc:.4f}")
    print(f"✅ Best model saved to {best_model_path}")
