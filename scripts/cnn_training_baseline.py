# -*- coding: utf-8 -*-
"""
cnn_training_baseline.py
-------------------------
CNN baseline training (no class weighting or sampling)
Author: Yulin Wang, Nov 2025
"""

import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from PIL import Image
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns

# ============ CONFIG ============
CSV_PATH = ""
IMG_DIR = ""
OUT_DIR = ""
BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 100
PATIENCE = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =================================


# ============ Dataset ============
class GaitHeatmapDataset(Dataset):
    def __init__(self, df, img_dir, split, transform=None):
        self.data = df[df["split"] == split].reset_index(drop=True)
        self.img_dir = os.path.join(img_dir, split)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row["filename"])
        img = Image.open(img_path).convert("L")
        label = 1 if row["group"] == "Pt" else 0
        if self.transform:
            img = self.transform(img)
        return img, label


# ============ Model ============
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 31 * 31, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ============ Training + Eval ============
def evaluate_model(model, dataloader, out_dir):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    df_pred = pd.DataFrame({"true": y_true, "pred": y_pred, "prob": y_prob})
    df_pred.to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Co", "Pt"], yticklabels=["Co", "Pt"])
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png")); plt.close()

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(); plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0,1],[0,1],'--'); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve")
    plt.legend(); plt.savefig(os.path.join(out_dir, "roc_curve.png")); plt.close()

    with open(os.path.join(out_dir, "report.txt"), "w") as f:
        f.write(classification_report(y_true, y_pred, target_names=["Co", "Pt"]))
        f.write(f"\nAUC: {roc_auc:.4f}")

    print(f"✅ Test AUC: {roc_auc:.4f}")
    return roc_auc


def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs, out_dir):
    best_acc = 0.0
    best_model_wts = model.state_dict()
    train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist = [], [], [], []
    patience_counter = 0

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss, running_corrects = 0.0, 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward(); optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            if phase == 'train':
                train_loss_hist.append(epoch_loss)
                train_acc_hist.append(epoch_acc.item())
            else:
                val_loss_hist.append(epoch_loss)
                val_acc_hist.append(epoch_acc.item())
                scheduler.step(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1

        if patience_counter >= PATIENCE: break

    # save curves
    plt.figure(); plt.plot(train_loss_hist, label='Train'); plt.plot(val_loss_hist, label='Val')
    plt.legend(); plt.title('Loss'); plt.savefig(os.path.join(out_dir, "loss_curve.png"))
    plt.figure(); plt.plot(train_acc_hist, label='Train'); plt.plot(val_acc_hist, label='Val')
    plt.legend(); plt.title('Accuracy'); plt.savefig(os.path.join(out_dir, "acc_curve.png"))
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))
    return model


def run_training(csv_path, img_dir, out_dir):
    df = pd.read_csv(csv_path)
    transform = transforms.Compose([transforms.Resize((248, 248)), transforms.ToTensor()])
    train_ds = GaitHeatmapDataset(df, img_dir, "train", transform)
    val_ds = GaitHeatmapDataset(df, img_dir, "val", transform)
    test_ds = GaitHeatmapDataset(df, img_dir, "test", transform)

    dataloaders = {
        "train": DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
        "val": DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4),
        "test": DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    }

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    model = SimpleCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    os.makedirs(out_dir, exist_ok=True)
    trained_model = train_model(model, criterion, optimizer, scheduler, dataloaders, EPOCHS, out_dir)
    evaluate_model(trained_model, dataloaders["test"], out_dir)
