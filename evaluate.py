# -*- coding: utf-8 -*-
"""
Created on 2025/6/30 09:53

@author: Yulin Wang
@email: yulin.wang@fau.de
"""
# scripts/evaluate.py

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from models.cnn_model import SimpleCNN

DATA_DIR = "data/heatmaps"
LABEL_CSV = "data/train_labels.csv"
MODEL_PATH = "models/best_model.pt"

class GaitDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.df = pd.read_csv(label_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(os.path.join(self.image_dir, row['filename'])).convert('L')
        if self.transform:
            image = self.transform(image)
        label = int(row['label'])
        return image, label

transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

dataset = GaitDataset(DATA_DIR, LABEL_CSV, transform)
loader = DataLoader(dataset, batch_size=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
        preds = out.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_probs.extend(probs)
        all_labels.extend(y.numpy())

print("Classification Report:")
print(classification_report(all_labels, all_preds))
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

# ROC Curve
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("models/roc_curve.png")
plt.show()
