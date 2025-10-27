# -*- coding: utf-8 -*-
"""
Created on 2025/09/27 16:30
Updated on 2025/10/27
Author: Yulin Wang
Email: yulin.wang@fau.de

Description:
CNN model for gait-phase heatmaps (PD vs HC)
with BatchNorm, Dropout, and He (Kaiming) initialization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    def __init__(self, fc_size=256):
        super(CNNModel, self).__init__()

        # --- Convolutional layers ---
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2)   # (B, 64, 128, 128)

        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)   # (B, 32, 64, 64)

        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool3 = nn.MaxPool2d(2)   # (B, 16, 32, 32)

        # --- Dropout ---
        self.dropout = nn.Dropout(0.5)

        # --- Fully connected layers ---
        self.fc1 = nn.Linear(16 * 32 * 32, fc_size)
        self.fc2 = nn.Linear(fc_size, 2)  # Binary classification: 0=HC, 1=PD

        # --- Initialize weights (He initialization) ---
        self._init_weights()

    def _init_weights(self):
        """
        Apply He (Kaiming) initialization for Conv2d and Linear layers.
        This improves gradient flow for ReLU activations.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # --- Conv Blocks ---
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # (B, 64, 128, 128)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))  # (B, 32, 64, 64)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))  # (B, 16, 32, 32)

        # --- Flatten + Fully Connected ---
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
