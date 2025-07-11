# -*- coding: utf-8 -*-
"""
Created on 2025/6/30 23:23

@author: Yulin Wang
@email: yulin.wang@fau.de
"""

# datasets/heatmap_dataset.py

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as T

class HeatmapDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform or T.Compose([
            T.Grayscale(),                    # 保证是单通道
            T.Resize((256, 256)),             # 统一尺寸
            T.ToTensor(),                     # 转为 tensor，shape=[1, 256, 256]
            T.Normalize(mean=[0.5], std=[0.5])  # 标准化至 [-1, 1]
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['filename']
        label = int(self.df.iloc[idx]['label'])

        image = Image.open(img_path).convert('L')  # 转灰度图
        image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)
