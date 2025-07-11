# -*- coding: utf-8 -*-
"""
Created on 2025/6/30 23:38

@author: Yulin Wang
@email: yulin.wang@fau.de
"""

# cnn_training_pipeline.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from datasets.heatmap_dataset import HeatmapDataset
# from models.cnn_model import CNNModel
from utils.early_stopping import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from config import Config
# === 1. 选择模型架构 ===
USE_PAPER_MODEL = True  # ← 改成 False 就可以切换回旧模型

if USE_PAPER_MODEL:
    from models.cnn_model_paper import CNNModel
else:
    from models.cnn_model import CNNModel
# === Setup ===
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
log_dir = os.path.join(Config.TENSORBOARD_LOG_DIR, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
writer = SummaryWriter(log_dir=log_dir)

# === Dataset ===
dataset = HeatmapDataset(Config.LABEL_CSV_PATH)
val_size = int(len(dataset) * Config.VAL_SPLIT)
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

# === Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
early_stopper = EarlyStopping(patience=Config.EARLY_STOPPING_PATIENCE, path=Config.MODEL_SAVE_PATH)

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

    writer.add_scalars('Loss', {'Train': train_loss, 'Validation': val_loss}, epoch + 1)
    writer.add_scalars('Accuracy', {'Train': train_acc, 'Validation': val_acc}, epoch + 1)

    print(f"Epoch {epoch+1}/{Config.EPOCHS} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    early_stopper(val_loss, model)
    if early_stopper.early_stop:
        print("\n🛑 Early stopping triggered.")
        break

writer.close()
print(f"\n✅ Best model saved to: {Config.MODEL_SAVE_PATH}")
