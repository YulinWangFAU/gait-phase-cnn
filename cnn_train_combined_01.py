# -*- coding: utf-8 -*-
"""
Train combined CNN model on Ga_01 + Ju_01 + Si_01
Now includes dynamic label path from Config (I_POINTS, GAUSS_SMOOTH)
and fixed train/val/test split (70/15/15)
"""

import argparse
import csv
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

from datasets.heatmap_dataset import HeatmapDataset
from utils.early_stopping import EarlyStopping
from config import Config
from models.cnn_model_paper import CNNModel

# === argparse 参数 ===
parser = argparse.ArgumentParser()
parser.add_argument('--win', type=int, required=True, help='Window size (use 0 for full signal)')
parser.add_argument('--step', type=int, required=True, help='Step size (use 0 for full signal)')
args = parser.parse_args()

# === 打印当前配置参数 ===
print(f"\n🧭 Current Config: I_POINTS={Config.I_POINTS}, GAUSS_SMOOTH={Config.GAUSS_SMOOTH}")
print(f"📂 Output folder: {Config.TENSORBOARD_LOG_DIR}")

# === 自动更新标签路径 ===
config_name = f"fullsignal_i{Config.I_POINTS}_s{Config.GAUSS_SMOOTH}"
LABEL_CSV_PATH = os.path.join(Config.BASE_DIR, f"labels_{config_name}.csv")

if not os.path.exists(LABEL_CSV_PATH):
    raise FileNotFoundError(f"❌ Label file not found: {LABEL_CSV_PATH}\n请先运行生成热力图脚本。")

# === 全局目录 ===
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(Config.TENSORBOARD_LOG_DIR, exist_ok=True)

# === 加载完整标签文件 ===
full_df = pd.read_csv(LABEL_CSV_PATH)

# === 超参数组合 ===
fc_sizes = [128, 256, 512]
condition = "_01"  # 只训练 _01 条件的合并数据

# === 主训练循环 ===
for fc_size in fc_sizes:
    print(f"\n🚀 Training combined dataset: All groups {condition} with fc_size={fc_size}")
    print(f"🔧 Using heatmaps from {config_name}")

    # === 合并 Ga_01 + Ju_01 + Si_01 ===
    subset_df = full_df[full_df["filename"].str.contains(condition)]
    if len(subset_df) == 0:
        print(f"⚠️ No samples found for {condition}, skipping...")
        continue

    print(f"✅ Found {len(subset_df)} samples for {condition}")

    # 保存合并后的标签文件
    subset_csv = os.path.join(Config.BASE_DIR, f"labels_All{condition}_{config_name}.csv")
    subset_df.to_csv(subset_csv, index=False)

    # === 数据加载 ===
    dataset = HeatmapDataset(subset_csv)
    total_size = len(dataset)

    val_ratio = Config.VAL_SPLIT  # 0.15
    test_ratio = Config.TEST_SPLIT  # 0.15
    val_size = int(total_size * val_ratio)
    test_size = int(total_size * test_ratio)
    train_size = total_size - val_size - test_size

    # === 测试集索引文件路径 ===
    test_idx_path = os.path.join(Config.CHECKPOINT_DIR, f"test_indices_{config_name}.pt")

    if os.path.exists(test_idx_path):
        print(f"📂 Loading existing test split: {test_idx_path}")
        test_indices = torch.load(test_idx_path)
        remaining_indices = [i for i in range(total_size) if i not in test_indices]
        remaining_ds = Subset(dataset, remaining_indices)
        val_size_new = int(len(remaining_indices) * val_ratio / (1 - test_ratio))
        train_size_new = len(remaining_indices) - val_size_new
        train_ds, val_ds = random_split(remaining_ds, [train_size_new, val_size_new])
        test_ds = Subset(dataset, test_indices)
    else:
        print(f"🧩 Creating new train/val/test split (70/15/15)")
        train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
        torch.save(test_ds.indices, test_idx_path)
        print(f"💾 Saved test indices to {test_idx_path}")

    # === DataLoaders ===
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

    # === 模型 & 优化器 ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNModel(fc_size=fc_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    # === 模型保存路径 ===
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, f"best_All{condition}_fc{fc_size}_{config_name}.pth")

    early_stopper = EarlyStopping(
        patience=Config.EARLY_STOPPING_PATIENCE,
        min_delta=Config.EARLY_STOPPING_DELTA,
        mode=Config.EARLY_STOPPING_MODE,
        path=best_model_path
    )

    # === 日志目录 ===
    log_dir = os.path.join(Config.TENSORBOARD_LOG_DIR, f"All{condition}_fc{fc_size}_win{args.win}_step{args.step}")
    os.makedirs(log_dir, exist_ok=True)
    log_csv_path = os.path.join(log_dir, "training_log.csv")
    f_csv = open(log_csv_path, 'w', newline='')
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

        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        train_loss_list.append(train_loss_avg)
        val_loss_list.append(val_loss_avg)

        print(f"Epoch {epoch + 1}/{Config.EPOCHS} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
              f"Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f}")

        writer_csv.writerow([epoch + 1, train_acc, val_acc, train_loss_avg, val_loss_avg, optimizer.param_groups[0]['lr']])
        f_csv.flush()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"🌟 Saved new best model at epoch {epoch + 1} (Val Acc = {val_acc:.4f})")

        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(Config.CHECKPOINT_DIR, f"epoch{epoch+1:03d}_All{condition}_fc{fc_size}_{config_name}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"💾 Saved checkpoint: {ckpt_path}")

        scheduler.step(val_acc)
        early_stopper(val_acc, model)
        if early_stopper.early_stop:
            print(f"🛑 Early stopping at epoch {epoch + 1} (no improvement).")
            break

    # === 保存训练曲线 ===
    plt.figure()
    plt.plot(train_acc_list, label='Train Acc')
    plt.plot(val_acc_list, label='Val Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    plt.title(f'All{condition} fc={fc_size} Accuracy ({config_name})')
    plt.savefig(os.path.join(log_dir, f"All{condition}_fc{fc_size}_acc.png"))
    plt.close()

    plt.figure()
    plt.plot(train_loss_list, label='Train Loss')
    plt.plot(val_loss_list, label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.title(f'All{condition} fc={fc_size} Loss ({config_name})')
    plt.savefig(os.path.join(log_dir, f"All{condition}_fc{fc_size}_loss.png"))
    plt.close()

    f_csv.close()
    print(f"✅ Training finished for fc_size={fc_size}. Best Val Acc: {best_val_acc:.4f}")
    print(f"✅ Best model saved to {best_model_path}")
