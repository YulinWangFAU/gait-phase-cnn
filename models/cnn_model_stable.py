# -*- coding: utf-8 -*-
"""
Created on 2025/10/27 20:01

@author: Yulin Wang
@email: yulin.wang@fau.de
"""
# -*- coding: utf-8 -*-
"""
Stable CNN model for gait-phase heatmaps (no BatchNorm)
Author: Yulin Wang
Date: 2025-10-27
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, fc_size=256, dropout_p=0.3):
        super(CNNModel, self).__init__()

        # --- Convolutional Blocks ---
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)  # (B, 64, 128, 128)

        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)  # (B, 32, 64, 64)

        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2)  # (B, 16, 32, 32)

        # --- Regularization ---
        self.dropout = nn.Dropout(dropout_p)

        # --- Dimensionality reduction ---
        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))  # reduce feature map to 8×8

        # --- Fully connected layers ---
        self.fc1 = nn.Linear(16 * 8 * 8, fc_size)
        self.fc2 = nn.Linear(fc_size, 2)  # Binary classification: HC=0, PD=1

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
