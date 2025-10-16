# model/cnn_model.py

import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # (B, 1, 256, 256) → (B, 16, 256, 256)
        self.pool1 = nn.MaxPool2d(2)                             # → (B, 16, 128, 128)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)                             # → (B, 32, 64, 64)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2)                             # → (B, 64, 32, 32)

        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.dropout = nn.Dropout(0.7)  # NEW: Dropout 层
        self.fc2 = nn.Linear(128, 2)  # 2类：Pt 和 Co

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  #  NEW: Dropout 应用在 fc1 后
        x = self.fc2(x)
        return x
