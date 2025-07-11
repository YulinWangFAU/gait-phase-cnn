# -*- coding: utf-8 -*-
"""
Created on 2025/6/30 20:16

@author: Yulin Wang
@email: yulin.wang@fau.de
"""
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class GaitDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file with image paths and labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            raise ValueError("Slicing not supported")

        img_path = self.data_frame.iloc[idx, 0]
        label = int(self.data_frame.iloc[idx, 1])

        # Ensure path is absolute
        if not os.path.isabs(img_path):
            img_path = os.path.join(os.path.dirname(self.data_frame.columns[0]), img_path)

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
