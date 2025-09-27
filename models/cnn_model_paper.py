# -*- coding: utf-8 -*-
"""
Created on 2025/9/27 16:30

@author: Yulin Wang
@email: yulin.wang@fau.de
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, fc_size=256):
        super(CNNModel, self).__init__()

        # Conv1: 输入 1 通道 → 输出 64 通道
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2)   # (B, 64, 128, 128)

        # Conv2: 64 → 32
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)   # (B, 32, 64, 64)

        # Conv3: 32 → 16
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool3 = nn.MaxPool2d(2)   # (B, 16, 32, 32)

        # Dropout 防止过拟合
        self.dropout = nn.Dropout(0.5)

        # 全连接层
        self.fc1 = nn.Linear(16 * 32 * 32, fc_size)  # 16×32×32=16384
        self.fc2 = nn.Linear(fc_size, 2)  # 二分类：0=Healthy, 1=Disordered

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # (B, 64, 128, 128)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))  # (B, 32, 64, 64)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))  # (B, 16, 32, 32)

        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # 输出 (B, 2)
        return x
