# -*- coding: utf-8 -*-
"""
Created on 2025/7/14 10:22

@author: Yulin Wang
@email: yulin.wang@fau.de
"""
# cnn_training_pipeline.py
import os
import csv
import importlib
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import Config
from datasets.heatmap_dataset import HeatmapDataset
from utils.early_stopping import EarlyStopping

# === 创建输出目录 ===
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(Config.TENSORBOARD_LOG_DIR, exist_ok=True)

# === 设置 TensorBoard 和 CSV 日志 ===
log_dir = Config.TENSORBOARD_LOG_DIR
writer = SummaryWriter(log_dir=log_dir)
log_csv_path = os.path.join(log_dir, "training_log.csv")
with open(log_csv_path, mode='w', newline='') as f:
    writer_csv = csv.writer(f)
    writer_csv.writerow(['epoch', 'train_acc', 'val_acc', 'train_loss', 'val_loss', 'lr'])

# === 数据集准备 ===
dataset = HeatmapDataset(Config.LABEL_CSV_PATH)
val_size = int(len(dataset) * Config.VAL_SPLIT)
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

# === 动态导入模型 ===
model_module = importlib.import_module(f"models.{Config.MODEL_NAME}")
CNNModel = getattr(model_module, "CNNModel")
model = CNNModel().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# === 损失、优化器、调度器、EarlyStopping
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, threshold=0.001)

early_stopper = EarlyStopping(
    patience=Config.EARLY_STOPPING_PATIENCE,
    min_delta=Config.EARLY_STOPPING_DELTA,
    mode=Config.EARLY_STOPPING_MODE,
    path=Config.MODEL_SAVE_PATH
)

# === 训练循环 ===
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

    # === 记录日志
    writer.add_scalars('Loss', {
        'Train': train_loss / len(train_loader),
        'Validation': val_loss / len(val_loader)
    }, epoch + 1)
    writer.add_scalars('Accuracy', {
        'Train': train_acc,
        'Validation': val_acc
    }, epoch + 1)

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch + 1}/{Config.EPOCHS} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.6f}")
    with open(log_csv_path, mode='a', newline='') as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow([
            epoch + 1,
            f"{train_acc:.4f}",
            f"{val_acc:.4f}",
            f"{train_loss / len(train_loader):.6f}",
            f"{val_loss / len(val_loader):.6f}",
            f"{current_lr:.6f}"
        ])

    scheduler.step(val_acc)
    early_stopper(val_acc, model)
    if early_stopper.early_stop:
        print("\n🛑 Early stopping triggered.")
        break

writer.close()
print(f"\n✅ Best model saved to: {Config.MODEL_SAVE_PATH}")
