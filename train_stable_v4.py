# -*- coding: utf-8 -*-
"""
Train CNNModelStable with 5-Fold Cross Validation
Supports fc_size = [128, 256, 512]
Author: Yulin Wang
Date: 2025-10-28
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from datasets.heatmap_dataset import HeatmapDataset
from utils.early_stopping import EarlyStopping
from config_preproc_train import Config
from models.cnn_model_stable import CNNModel


# === 固定随机种子 ===
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


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
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(Config.TENSORBOARD_LOG_DIR, exist_ok=True)
timestamp = time.strftime("%Y%m%d_%H%M", time.localtime())


# === 搜索匹配的标签文件 ===
label_pattern = os.path.join(Config.BASE_DIR, f"labels_{args.version}_*.csv")
label_files = sorted(glob.glob(label_pattern), key=os.path.getmtime, reverse=True)
if not label_files:
    raise FileNotFoundError(f"❌ 未找到匹配的标签文件: {label_pattern}\n请先运行 generate_heatmaps.py。")

LABEL_CSV_PATH = label_files[0]
print(f"\n🧭 Using label file: {LABEL_CSV_PATH}")

# === 加载标签文件 ===
df = pd.read_csv(LABEL_CSV_PATH)
df = df[df['filename'].str.contains('_01')]

# === 模式筛选 ===
mode = args.exp_mode
if mode == "Ga":
    df = df[df['filename'].str.contains(r'/Ga')]
elif mode == "Ju":
    df = df[df['filename'].str.contains(r'/Ju')]
elif mode == "Si":
    df = df[df['filename'].str.contains(r'/Si')]
elif mode == "All":
    df = df[df['filename'].str.contains('Ga') | df['filename'].str.contains('Ju') | df['filename'].str.contains('Si')]

# === 左右脚筛选 ===
if args.side in ["left", "right"]:
    df = df[df['filename'].str.contains(f"_{args.side}_")]
    print(f"🦶 Training on {args.side.upper()} foot samples only.")
else:
    print("🦶 Training on both feet combined dataset.")

if len(df) == 0:
    raise ValueError(f"❌ No samples found for mode={mode}, side={args.side}.")

# === 样本统计 ===
n_co = df['filename'].str.contains('Co').sum()
n_pt = df['filename'].str.contains('Pt').sum()
print(f"✅ 当前模式: {mode}_01 | side={args.side} | 样本总数: {len(df)} | Control={n_co} | Parkinson={n_pt}")

# === 保存临时标签文件 ===
subset_csv = os.path.join(Config.BASE_DIR,
                          f"labels_{mode}_01_{args.side}_{args.version}_{timestamp}.csv")
df.to_csv(subset_csv, index=False)

# === 构建数据集 ===
dataset = HeatmapDataset(subset_csv)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 三种全连接层大小 ===
fc_sizes = [128, 256, 512]

# === 主循环 ===
for fc_size in fc_sizes:
    print(f"\n🚀 Training with fc_size={fc_size}")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    fold_idx = 1

    for train_idx, val_idx in kf.split(np.arange(len(dataset))):
        print(f"\n===== Fold {fold_idx}/5 (fc_size={fc_size}) =====")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=Config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=Config.BATCH_SIZE, shuffle=False)

        # === 初始化模型 ===
        model = CNNModel(fc_size=fc_size, dropout_p=0.3).to(device)
        optimizer = optim.Adam(model.parameters(),
                               lr=Config.LEARNING_RATE,
                               weight_decay=Config.WEIGHT_DECAY)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        criterion = nn.CrossEntropyLoss()
        early_stopper = EarlyStopping(
            patience=Config.EARLY_STOPPING_PATIENCE,
            min_delta=Config.EARLY_STOPPING_DELTA,
            mode=Config.EARLY_STOPPING_MODE
        )

        # === 日志路径 ===
        model_tag = f"{mode}_fold{fold_idx}_fc{fc_size}_{args.side}_{args.version}_{timestamp}"
        log_dir = os.path.join(Config.TENSORBOARD_LOG_DIR, model_tag)
        os.makedirs(log_dir, exist_ok=True)
        log_csv_path = os.path.join(log_dir, "training_log.csv")
        f_csv = open(log_csv_path, 'w', newline='', encoding='utf-8')
        writer_csv = csv.writer(f_csv)
        writer_csv.writerow(['epoch', 'train_acc', 'val_acc', 'train_loss', 'val_loss', 'lr'])

        best_val_acc = 0.0
        train_acc_list, val_acc_list, train_loss_list, val_loss_list = [], [], [], []

        # === 每折训练 ===
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

            # === 验证 ===
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

            # === 记录日志 ===
            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)

            print(f"Epoch {epoch+1}/{Config.EPOCHS} | "
                  f"Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            writer_csv.writerow([
                f"{epoch + 1}",
                f"{train_acc:.6f}",
                f"{val_acc:.6f}",
                f"{train_loss:.6f}",
                f"{val_loss:.6f}",
                f"{optimizer.param_groups[0]['lr']:.6f}"
            ])
            f_csv.flush()

            # === 保存最优模型 ===
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_path = os.path.join(Config.CHECKPOINT_DIR, f"best_fold{fold_idx}_fc{fc_size}.pth")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f"🌟 New best model saved (Val Acc = {val_acc:.4f})")

            scheduler.step(val_acc)
            early_stopper(val_acc, model)
            if early_stopper.early_stop:
                print(f"🛑 Early stopping at epoch {epoch + 1}")
                break

        f_csv.close()

        # === 绘制曲线 ===
        plt.figure()
        plt.plot(train_acc_list, label='Train Acc')
        plt.plot(val_acc_list, label='Val Acc')
        plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
        plt.title(f'Accuracy (Fold {fold_idx}, fc={fc_size})')
        plt.savefig(os.path.join(log_dir, f"fold{fold_idx}_fc{fc_size}_acc.png"))
        plt.close()

        plt.figure()
        plt.plot(train_loss_list, label='Train Loss')
        plt.plot(val_loss_list, label='Val Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
        plt.title(f'Loss (Fold {fold_idx}, fc={fc_size})')
        plt.savefig(os.path.join(log_dir, f"fold{fold_idx}_fc{fc_size}_loss.png"))
        plt.close()

        print(f"✅ Fold {fold_idx} best Val Acc = {best_val_acc:.4f}")
        fold_results.append(best_val_acc)
        fold_idx += 1

    # === 汇总结果（当前 fc_size） ===
    mean_acc = np.mean(fold_results)
    std_acc = np.std(fold_results)
    print(f"\n✅ fc_size={fc_size} 平均验证准确率: {mean_acc:.4f} ± {std_acc:.4f}")
