# -*- coding: utf-8 -*-
"""
Created on 2025/7/14 23:28

@author: Yulin Wang
@email: yulin.wang@fau.de
"""
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)  # → (B, 16, 128, 128)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)  # → (B, 32, 64, 64)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2)  # → (B, 64, 32, 32)

        self.dropout = nn.Dropout(0.5)  # 👈 更强的正则化

        self.fc1 = nn.Linear(64 * 32 * 32, 64)  # 👈 更小的隐藏层
        self.fc2 = nn.Linear(64, 2)  # Pt vs Co

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
