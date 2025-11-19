# -*- coding: utf-8 -*-
"""
Created on 2025/11/19 15:09

@author: Yulin Wang
@email: yulin.wang@fau.de
"""
# -*- coding: utf-8 -*-
"""
cnn_phaseplot_kfold.py
--------------------------------------------
Train CalvoCNN on ONE fold of K-fold CV.
Data structure comes from split_into_kfold.py.
"""

import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
np.random.seed(42)


# ------------------------ Dataset ------------------------
class HeatmapDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        df = pd.read_csv(csv_path)
        self.paths = df["path"].tolist()
        self.labels = df["label"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("L")
        label = self.labels[idx]
        return self.transform(img), label


# ---------------------- CalvoCNN ------------------------
class CalvoCNN(nn.Module):
    def __init__(self, fc_dim=128, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 32, 2, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, 248, 248)
            out = self.features(dummy)
            self.flatten_dim = out.numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, fc_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fc_dim, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# --------------------- Train + Eval ---------------------
def train_fold(train_csv, val_csv, test_csv, out_dir,
               epochs=50, batch=8, lr=1e-4, fc_dim=128):

    os.makedirs(out_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((248, 248)),
        transforms.ToTensor()
    ])

    train_ds = HeatmapDataset(train_csv, transform)
    val_ds = HeatmapDataset(val_csv, transform)
    test_ds = HeatmapDataset(test_csv, transform)

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch)
    test_loader = DataLoader(test_ds, batch_size=batch)

    model = CalvoCNN(fc_dim=fc_dim).to(DEVICE)
    optimiz = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiz, 'min', factor=0.5, patience=3)

    best_acc = 0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_correct, train_loss = 0, 0

        for img, label in train_loader:
            img, label = img.to(DEVICE), label.to(DEVICE)
            optimiz.zero_grad()
            out = model(img)
            loss = criterion(out, label)
            loss.backward()
            optimiz.step()
            train_loss += loss.item() * img.size(0)
            train_correct += (out.argmax(1) == label).sum().item()

        train_acc = train_correct / len(train_ds)

        # ---- Validation ----
        model.eval()
        val_correct, val_loss = 0, 0
        with torch.no_grad():
            for img, label in val_loader:
                img, label = img.to(DEVICE), label.to(DEVICE)
                out = model(img)
                loss = criterion(out, label)
                val_loss += loss.item() * img.size(0)
                val_correct += (out.argmax(1) == label).sum().item()

        val_acc = val_correct / len(val_ds)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1:02d} | Train Acc={train_acc:.3f} | Val Acc={val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))
        else:
            patience_counter += 1

        if patience_counter >= 7:
            print("Early stopping.")
            break

    # ---------------- Test ----------------
    model.load_state_dict(torch.load(os.path.join(out_dir, "best_model.pt")))
    y_true, y_pred, y_prob = [], [], []

    model.eval()
    with torch.no_grad():
        for img, label in test_loader:
            img = img.to(DEVICE)
            out = model(img)
            prob = torch.softmax(out, 1)[:, 1].cpu().numpy()
            pred = out.argmax(1).cpu().numpy()
            y_prob.extend(prob)
            y_pred.extend(pred)
            y_true.extend(label.numpy())

    # Save results
    df = pd.DataFrame({"true": y_true, "pred": y_pred, "prob": y_prob})
    df.to_csv(os.path.join(out_dir, "test_pred.csv"), index=False)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=["Co","Pt"], yticklabels=["Co","Pt"])
    plt.savefig(os.path.join(out_dir, "cm.png")); plt.close()

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    AUC = auc(fpr, tpr)
    plt.plot(fpr, tpr)
    plt.title(f"ROC AUC={AUC:.3f}")
    plt.savefig(os.path.join(out_dir, "roc.png"))
    plt.close()

    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write(classification_report(y_true, y_pred))
        f.write(f"\nAUC={AUC:.3f}\n")

    print(f"Test AUC = {AUC:.3f}")


# ----------------------------- MAIN -----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fc_dim", type=int, default=128)
    args = ap.parse_args()

    train_fold(args.train_csv, args.val_csv, args.test_csv, args.out,
               fc_dim=args.fc_dim)
