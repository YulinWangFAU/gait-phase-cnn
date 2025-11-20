# -*- coding: utf-8 -*-
"""
cnn_phaseplot_model_kfold.py (FINAL)
--------------------------------------------
Compatible with directory:
    .../<experiment>/<sigma>_kfold/<method>_<signal>/folds/foldX/train.csv

Supports:
    - baseline / balanced
    - fc_dim choices
    - patience=5
    - batch=4
"""

import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
np.random.seed(42)


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------
class HeatmapDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        df = pd.read_csv(csv_path)

        # path column exists
        self.paths = df["path"].tolist()

        # convert group → label
        if "label" in df.columns:
            self.labels = df["label"].astype(int).tolist()
        else:
            # group column: Co / Pt
            self.labels = df["group"].apply(lambda g: 0 if g=="Co" else 1).tolist()

        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("L")
        return self.transform(img), int(self.labels[idx])



# ---------------------------------------------------------
# Calvo-Ariza CNN
# ---------------------------------------------------------
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

        # compute flatten dim
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
        x = self.features(x)
        return self.classifier(x)


# ---------------------------------------------------------
# Train one fold
# ---------------------------------------------------------
def train_fold(train_csv, val_csv, test_csv, out_dir,
               fc_dim=128, mode="baseline",
               epochs=50, batch=4, lr=1e-4, patience=5):

    os.makedirs(out_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((248, 248)),
        transforms.ToTensor()
    ])

    train_ds = HeatmapDataset(train_csv, transform)
    val_ds   = HeatmapDataset(val_csv, transform)
    test_ds  = HeatmapDataset(test_csv, transform)

    # -----------------------------------------------------
    # balanced mode
    # -----------------------------------------------------
    if mode == "balanced":
        labels = pd.read_csv(train_csv)["label"].values
        cls_weights = compute_class_weight("balanced", classes=np.array([0,1]), y=labels)
        cls_weights = torch.tensor(cls_weights, dtype=torch.float, device=DEVICE)

        sample_weights = np.array([cls_weights[l].item() for l in labels])
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

        train_loader = DataLoader(train_ds, batch_size=batch, sampler=sampler)
        criterion = nn.CrossEntropyLoss(weight=cls_weights)

    else:
        train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
        criterion = nn.CrossEntropyLoss()

    val_loader  = DataLoader(val_ds, batch_size=batch)
    test_loader = DataLoader(test_ds, batch_size=batch)

    # -----------------------------------------------------
    # Model / optimizer
    # -----------------------------------------------------
    model = CalvoCNN(fc_dim=fc_dim).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=3)

    best_acc = -1
    patience_counter = 0

    train_acc_hist, val_acc_hist = [], []
    train_loss_hist, val_loss_hist = [], []

    print(f"\n🔥 Training fold at {out_dir}")
    print(f"Mode={mode}, FC={fc_dim}, batch={batch}, patience={patience}\n")

    # -----------------------------------------------------
    # Training loop
    # -----------------------------------------------------
    for ep in range(epochs):

        # --------------- Train ---------------
        model.train()
        t_loss, t_correct = 0, 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()

            t_loss += loss.item() * x.size(0)
            t_correct += (out.argmax(1) == y).sum().item()

        train_loss = t_loss / len(train_ds)
        train_acc  = t_correct / len(train_ds)

        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)

        # --------------- Validate ---------------
        model.eval()
        v_loss, v_correct = 0, 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                loss = criterion(out, y)
                v_loss += loss.item() * x.size(0)
                v_correct += (out.argmax(1) == y).sum().item()

        val_loss = v_loss / len(val_ds)
        val_acc  = v_correct / len(val_ds)

        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)

        sched.step(val_loss)

        print(f"Epoch {ep+1:02d} | Train={train_acc:.3f} | Val={val_acc:.3f}")

        # -------- Early stopping --------
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"⏳ Early stopping triggered at epoch {ep+1}.")
            break

    # ---------------------------------------------------------
    # Save training curves
    # ---------------------------------------------------------
    plt.figure()
    plt.plot(train_acc_hist, label="Train")
    plt.plot(val_acc_hist, label="Val")
    plt.title("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "acc_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(train_loss_hist, label="Train")
    plt.plot(val_loss_hist, label="Val")
    plt.title("Loss")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"))
    plt.close()

    # ---------------------------------------------------------
    # Testing
    # ---------------------------------------------------------
    print("\n🎯 Testing...")

    model.load_state_dict(torch.load(os.path.join(out_dir, "best_model.pt")))

    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            out = model(x)
            prob = torch.softmax(out, 1)[:,1].cpu().numpy()
            pred = out.argmax(1).cpu().numpy()

            y_prob.extend(prob)
            y_pred.extend(pred)
            y_true.extend(y.numpy())

    # save predictions
    pd.DataFrame({
        "true": y_true,
        "pred": y_pred,
        "prob": y_prob
    }).to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=["Co","Pt"], yticklabels=["Co","Pt"])
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
    plt.close()

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr)
    plt.title(f"ROC AUC={roc_auc:.3f}")
    plt.savefig(os.path.join(out_dir, "roc_curve.png"))
    plt.close()

    # write report
    with open(os.path.join(out_dir, "report.txt"), "w") as f:
        f.write(f"Results for {out_dir}\n\n")
        f.write(classification_report(y_true, y_pred))
        f.write(f"\nAUC={roc_auc:.4f}\n")

    print(f"\n✅ DONE: AUC={roc_auc:.4f}")
    print(f"Results saved in: {out_dir}\n")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["baseline","balanced"], default="baseline")
    ap.add_argument("--fc_dim", type=int, default=128)
    args = ap.parse_args()

    train_fold(
        args.train_csv,
        args.val_csv,
        args.test_csv,
        args.out,
        fc_dim=args.fc_dim,
        mode=args.mode
    )
