# -*- coding: utf-8 -*-
"""
Created on 2025/7/11 16:20

@author: Yulin Wang
@email: yulin.wang@fau.de
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModelPaper(nn.Module):
    def __init__(self):
        super(CNNModelPaper, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # (B, 1, 256, 256) → (B, 32, 256, 256)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)                              # → (B, 32, 128, 128)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)                              # → (B, 64, 64, 64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)                              # → (B, 128, 32, 32)

        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(128 * 32 * 32, 256)
        self.fc2 = nn.Linear(256, 2)  # 二分类：0=Control, 1=Patient

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
