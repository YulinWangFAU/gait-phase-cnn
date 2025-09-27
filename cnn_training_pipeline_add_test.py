# -*- coding: utf-8 -*-
"""
Created on 2025/9/26 00:39

@author: Yulin Wang
@email: yulin.wang@fau.de
"""

import argparse
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from datasets.heatmap_dataset import HeatmapDataset
from utils.early_stopping import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from config import Config
from torch.optim.lr_scheduler import ReduceLROnPlateau
import importlib   # ✅ 新增

# === argparse 参数 ===
parser = argparse.ArgumentParser()
parser.add_argument('--win', type=int, required=True, help='Window size (use 0 for full signal)')
parser.add_argument('--step', type=int, required=True, help='Step size (use 0 for full signal)')
args = parser.parse_args()

# === 动态加载模型 ✅ ===
model_module = importlib.import_module(f"models.{Config.MODEL_NAME}")
CNNModel = getattr(model_module, "CNNModel")

# === 日志与目录设置 ===
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
log_dir = os.path.join(
    Config.TENSORBOARD_LOG_DIR,
    f"win{args.win}_step{args.step}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)
writer = SummaryWriter(log_dir=log_dir)

# CSV 日志
log_csv_path = os.path.join(log_dir, "training_log.csv")
f_csv = open(log_csv_path, mode='w', newline='')
writer_csv = csv.writer(f_csv)
writer_csv.writerow(['epoch', 'train_acc', 'val_acc', 'train_loss', 'val_loss', 'lr'])

# === Dataset ===
dataset = HeatmapDataset(Config.LABEL_CSV_PATH)
test_size = int(len(dataset) * Config.TEST_SPLIT)
val_size = int(len(dataset) * Config.VAL_SPLIT)
train_size = len(dataset) - val_size - test_size
train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

# === Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel().to(device)   # ✅ 动态模型
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)

scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, threshold=0.001)
early_stopper = EarlyStopping(
    patience=Config.EARLY_STOPPING_PATIENCE,
    min_delta=Config.EARLY_STOPPING_DELTA,
    mode=Config.EARLY_STOPPING_MODE,
    path=Config.MODEL_SAVE_PATH
)

# === Training Loop ===
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
    current_lr = optimizer.param_groups[0]['lr']

    writer.add_scalars('Loss', {'Train': train_loss_avg, 'Validation': val_loss_avg}, epoch + 1)
    writer.add_scalars('Accuracy', {'Train': train_acc, 'Validation': val_acc}, epoch + 1)

    print(f"Epoch {epoch + 1}/{Config.EPOCHS} | "
          f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.6f}")

    writer_csv.writerow([
        epoch + 1,
        f"{train_acc:.4f}",
        f"{val_acc:.4f}",
        f"{train_loss_avg:.6f}",
        f"{val_loss_avg:.6f}",
        f"{current_lr:.6f}"
    ])
    f_csv.flush()

    scheduler.step(val_acc)
    early_stopper(val_acc, model)
    if early_stopper.early_stop:
        print("\n🛑 Early stopping triggered.")
        break

# === 测试集评估 ===
print("\n🎯 Evaluating on test set...")
model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
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

print(f"✅ Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss_avg:.6f}")

writer.add_scalar("Test/Accuracy", test_acc)
writer.add_scalar("Test/Loss", test_loss_avg)
writer_csv.writerow(["TEST", "", f"{test_acc:.4f}", "", f"{test_loss_avg:.6f}", ""])
f_csv.flush()
f_csv.close()
writer.close()
print(f"\n🏆 Best model saved to: {Config.MODEL_SAVE_PATH}")
