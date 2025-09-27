# -*- coding: utf-8 -*-
"""
Created on 2025/6/30 23:49
快速可视化一小批样本，显示模型预测和真实标签，帮助直观评估模型的分类效果。
@author: Yulin Wang
@email: yulin.wang@fau.de
"""

# scripts/visualize_predictions.py

import torch
import matplotlib.pyplot as plt
import numpy as np
from models.cnn_model import CNNModel
from utils.dataset import HeatmapDataset
from config import config
from torchvision import transforms
from torch.utils.data import DataLoader

# 加载模型
model = CNNModel()
model.load_state_dict(torch.load("checkpoints/cnn_best.pt"))
model.eval()

# 加载数据（只选一小批验证图）
dataset = HeatmapDataset(csv_file=config.label_csv_path, transform=transforms.ToTensor())
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# 可视化一批
images, labels = next(iter(loader))
with torch.no_grad():
    outputs = model(images)
    preds = torch.argmax(outputs, dim=1)

# 绘图
plt.figure(figsize=(12, 6))
for i in range(len(images)):
    img = images[i].squeeze().numpy()
    plt.subplot(2, 4, i + 1)
    plt.imshow(img, cmap="hot")
    plt.title(f"GT: {labels[i].item()} / Pred: {preds[i].item()}")
    plt.axis("off")
plt.tight_layout()
plt.show()
