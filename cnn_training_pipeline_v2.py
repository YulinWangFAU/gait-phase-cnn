# -*- coding: utf-8 -*-
"""
Train CNN models for Ga/Ju/Si groups with versioned dataset paths (Config.VERSION_TAG)
Each group & condition trained separately with multiple fc sizes.
"""

import argparse
import csv
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import matplotlib.pyplot as plt
from datasets.heatmap_dataset import HeatmapDataset
from utils.early_stopping import EarlyStopping
from config import Config
from models.cnn_model_paper import CNNModel

from tqdm import tqdm
tqdm.disable = True  # ✅ 禁用 tqdm 进度条，防止 CSV 出现换行错位

# === argparse 参数 ===
parser = argparse.ArgumentParser()
parser.add_argument('--win', type=int, required=True, help='Window size (use 0 for full signal)')
parser.add_argument('--step', type=int, required=True, help='Step size (use 0 for full signal)')
args = parser.parse_args()

# === 打印当前版本信息 ===
print(f"\n🧭 Training with version: {Config.VERSION_TAG}")
print(f"📁 Base dir: {Config.BASE_DIR}")
print(f"📂 Checkpoints dir: {Config.CHECKPOINT_DIR}")
print(f"📂 TensorBoard dir: {Config.TENSORBOARD_LOG_DIR}\n")

# === 确保输出目录存在 ===
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(Config.TENSORBOARD_LOG_DIR, exist_ok=True)

# === 加载主标签文件 ===
config_name = f"fullsignal_{Config.VERSION_TAG}"
LABEL_CSV_PATH = os.path.join(Config.BASE_DIR, f"labels_{config_name}.csv")

if not os.path.exists(LABEL_CSV_PATH):
    raise FileNotFoundError(f"❌ Label file not found: {LABEL_CSV_PATH}\n请先运行生成热力图脚本。")

full_df = pd.read_csv(LABEL_CSV_PATH)

groups = ["Ga", "Ju", "Si"]
conditions = ["_01", "_02"]
fc_sizes = [128, 256, 512]

# === 主训练循环 ===
for g in groups:
    for c in conditions:
        for fc_size in fc_sizes:
            print(f"\n🚀 Training subset: {g}{c} | fc_size={fc_size} | version={Config.VERSION_TAG}")

            # === 筛选数据子集 ===
            subset_df = full_df[full_df["filename"].str.contains(f"{g}") & full_df["filename"].str.contains(f"{c}")]
            if len(subset_df) == 0:
                print(f"⚠️ No samples found for {g}{c}, skipping...")
                continue

            subset_csv = os.path.join(Config.BASE_DIR, f"labels_{g}{c}_{config_name}.csv")
            subset_df.to_csv(subset_csv, index=False)

            # === 数据划分 ===
            dataset = HeatmapDataset(subset_csv)
            total_size = len(dataset)
            val_ratio = Config.VAL_SPLIT
            val_size = int(total_size * val_ratio)
            train_size = total_size - val_size
            train_ds, val_ds = random_split(dataset, [train_size, val_size])

            train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

            # === 模型定义 ===
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = CNNModel(fc_size=fc_size).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

            model_save_path = os.path.join(
                Config.CHECKPOINT_DIR,
                f"best_{g}{c}_fc{fc_size}_{Config.VERSION_TAG}.pth"
            )

            early_stopper = EarlyStopping(
                patience=Config.EARLY_STOPPING_PATIENCE,
                min_delta=Config.EARLY_STOPPING_DELTA,
                mode=Config.EARLY_STOPPING_MODE,
                path=model_save_path
            )

            # === 日志路径 ===
            log_dir = os.path.join(
                Config.TENSORBOARD_LOG_DIR,
                f"{g}{c}_fc{fc_size}_win{args.win}_step{args.step}"
            )
            os.makedirs(log_dir, exist_ok=True)
            log_csv_path = os.path.join(log_dir, "training_log.csv")
            f_csv = open(log_csv_path, 'w', newline='', encoding='utf-8')  # ✅ 指定编码
            writer_csv = csv.writer(f_csv)
            writer_csv.writerow(['epoch', 'train_acc', 'val_acc', 'train_loss', 'val_loss', 'lr'])
            f_csv.flush()
            os.fsync(f_csv.fileno())  # ✅ 防止缓冲问题

            # === 训练循环 ===
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

                train_acc = train_correct / train_size
                val_acc = val_correct / val_size
                train_loss_avg = train_loss / len(train_loader)
                val_loss_avg = val_loss / len(val_loader)

                train_acc_list.append(train_acc)
                val_acc_list.append(val_acc)
                train_loss_list.append(train_loss_avg)
                val_loss_list.append(val_loss_avg)

                print(f"Epoch {epoch + 1}/{Config.EPOCHS} | "
                      f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

                writer_csv.writerow([
                    f"{epoch + 1}",
                    f"{train_acc:.6f}",
                    f"{val_acc:.6f}",
                    f"{train_loss_avg:.6f}",
                    f"{val_loss_avg:.6f}",
                    f"{optimizer.param_groups[0]['lr']:.6f}"
                ])
                f_csv.flush()
                os.fsync(f_csv.fileno())  # ✅ 确保立即写入磁盘

                scheduler.step(val_acc)
                early_stopper(val_acc, model)
                if early_stopper.early_stop:
                    print(f"🛑 Early stopping at epoch {epoch + 1} for {g}{c}_fc{fc_size}")
                    break

            # === 保存训练曲线 ===
            plt.figure()
            plt.plot(train_acc_list, label='Train Acc')
            plt.plot(val_acc_list, label='Val Acc')
            plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
            plt.title(f'{g}{c} fc={fc_size} Accuracy ({Config.VERSION_TAG})')
            plt.savefig(os.path.join(log_dir, f"{g}{c}_fc{fc_size}_acc.png"))
            plt.close()

            plt.figure()
            plt.plot(train_loss_list, label='Train Loss')
            plt.plot(val_loss_list, label='Val Loss')
            plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
            plt.title(f'{g}{c} fc={fc_size} Loss ({Config.VERSION_TAG})')
            plt.savefig(os.path.join(log_dir, f"{g}{c}_fc{fc_size}_loss.png"))
            plt.close()

            f_csv.close()
            print(f"✅ Saved model & curves for {g}{c}_fc{fc_size}_{Config.VERSION_TAG}")
