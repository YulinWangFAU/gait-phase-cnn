import argparse
import csv
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt  # 🟢 新增
from datasets.heatmap_dataset import HeatmapDataset
from utils.early_stopping import EarlyStopping
from config import Config
from models.cnn_model_paper import CNNModel  # ✏️ 修改：使用你新的模型

# === argparse 参数 ===
parser = argparse.ArgumentParser()
parser.add_argument('--win', type=int, required=True, help='Window size (use 0 for full signal)')
parser.add_argument('--step', type=int, required=True, help='Step size (use 0 for full signal)')
args = parser.parse_args()

# === 全局目录 ===
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(Config.TENSORBOARD_LOG_DIR, exist_ok=True)

# === 加载完整标签文件 ===
full_df = pd.read_csv(Config.LABEL_CSV_PATH)

groups = ["Ga", "Ju", "Si"]
conditions = ["_01", "_02"]
fc_sizes = [128, 256, 512]  # 🟢 新增：三种全连接层规模

# === 主循环 ===
for g in groups:
    for c in conditions:
        for fc_size in fc_sizes:  # 🟢 新增
            print(f"\n🚀 Training subset: {g}{c} with fc_size={fc_size}")

            subset_df = full_df[full_df["filename"].str.contains(f"{g}") & full_df["filename"].str.contains(f"{c}")]
            if len(subset_df) == 0:
                print(f"⚠️ No samples found for {g}{c}, skipping...")
                continue

            subset_csv = os.path.join(Config.BASE_DIR, f"labels_{g}{c}.csv")
            subset_df.to_csv(subset_csv, index=False)

            # === 数据划分 ===
            dataset = HeatmapDataset(subset_csv)
            total_size = len(dataset)
            val_ratio = 0.15
            val_size = int(total_size * val_ratio)
            train_size = total_size - val_size
            train_ds, val_ds = random_split(dataset, [train_size, val_size])

            train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

            # === 模型 ===
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = CNNModel(fc_size=fc_size).to(device)  # ✏️ 修改
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
            model_save_path = os.path.join(Config.CHECKPOINT_DIR, f"best_{g}{c}_fc{fc_size}.pth")

            early_stopper = EarlyStopping(
                patience=Config.EARLY_STOPPING_PATIENCE,
                min_delta=Config.EARLY_STOPPING_DELTA,
                mode=Config.EARLY_STOPPING_MODE,
                path=model_save_path
            )

            # === 日志路径 ===
            log_dir = os.path.join(Config.TENSORBOARD_LOG_DIR,
                                   f"{g}{c}_fc{fc_size}_win{args.win}_step{args.step}")
            os.makedirs(log_dir, exist_ok=True)
            log_csv_path = os.path.join(log_dir, "training_log.csv")
            f_csv = open(log_csv_path, 'w', newline='')
            writer_csv = csv.writer(f_csv)
            writer_csv.writerow(['epoch', 'train_acc', 'val_acc', 'train_loss', 'val_loss', 'lr'])

            # === 训练循环 ===
            train_acc_list, val_acc_list, train_loss_list, val_loss_list = [], [], [], []  # 🟢 新增
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

                print(f"Epoch {epoch + 1}/{Config.EPOCHS} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
                writer_csv.writerow([epoch + 1, train_acc, val_acc, train_loss_avg, val_loss_avg, optimizer.param_groups[0]['lr']])
                f_csv.flush()

                scheduler.step(val_acc)
                early_stopper(val_acc, model)
                if early_stopper.early_stop:
                    print(f"🛑 Early stopping at epoch {epoch + 1} for {g}{c}_fc{fc_size}")
                    break

            # === 保存训练曲线图 === 🟢 新增
            plt.figure()
            plt.plot(train_acc_list, label='Train Acc')
            plt.plot(val_acc_list, label='Val Acc')
            plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
            plt.title(f'{g}{c} fc={fc_size} Accuracy')
            plt.savefig(os.path.join(log_dir, f"{g}{c}_fc{fc_size}_acc.png"))
            plt.close()

            plt.figure()
            plt.plot(train_loss_list, label='Train Loss')
            plt.plot(val_loss_list, label='Val Loss')
            plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
            plt.title(f'{g}{c} fc={fc_size} Loss')
            plt.savefig(os.path.join(log_dir, f"{g}{c}_fc{fc_size}_loss.png"))
            plt.close()

            f_csv.close()
            print(f"✅ Saved model & curves for {g}{c}_fc{fc_size}")
