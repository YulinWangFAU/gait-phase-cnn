# -*- coding: utf-8 -*-
"""
Created on 2025/6/30 23:25

@author: Yulin Wang
@email: yulin.wang@fau.de
"""

# scripts/test_dataset_loader.py

from datasets.heatmap_dataset import HeatmapDataset
from torch.utils.data import DataLoader

dataset = HeatmapDataset(csv_path="/data/labels.csv")
loader = DataLoader(dataset, batch_size=8, shuffle=True)

for i, (x, y) in enumerate(loader):
    print(f"Batch {i}: X shape = {x.shape}, y = {y}")
    if i == 2:
        break
