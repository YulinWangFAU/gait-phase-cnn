# -*- coding: utf-8 -*-
"""
Created on 2025/11/5 17:07

@author: Yulin Wang
@email: yulin.wang@fau.de
"""
# -*- coding: utf-8 -*-
"""
cnn_training_full_eval.py
Train and evaluate a CNN on gait heatmaps (baseline or balanced).
"""

import os, argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from PIL import Image
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- Dataset ----------------
class GaitHeatmapDataset(Dataset):
    def __init__(self, df, img_dir, split, transform=None):
        self.data = df[df["split"] == split].reset_index(drop=True)
        self.img_dir = os.path.join(img_dir, split)
        self.transform = transform

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = Image.open(os.path.join(self.img_dir, row["filename"])).convert("L")
        label = 1 if row["group"] == "Pt" else 0
        return self.transform(img), label


# ---------------- Model ----------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 31 * 31, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# ---------------- Train & Eval ----------------
def train_one(csv_path, img_dir, out_dir, balance=False, epochs=100, batch_size=16, lr=1e-4, patience=10):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    transform = transforms.Compose([transforms.Resize((248, 248)), transforms.ToTensor()])
    train_ds = GaitHeatmapDataset(df, img_dir, "train", transform)
    val_ds = GaitHeatmapDataset(df, img_dir, "val", transform)
    test_ds = GaitHeatmapDataset(df, img_dir, "test", transform)

    if balance:
        train_labels = np.array([1 if g == "Pt" else 0 for g in df[df["split"] == "train"]["group"]])
        class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=train_labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
        sample_weights = np.array([class_weights[int(l)] for l in train_labels])
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = SimpleCNN().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=5)

    best_acc, patience_counter = 0, 0
    train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist = [], [], [], []

    for epoch in range(epochs):
        for phase, loader in [("train", train_loader), ("val", val_loader)]:
            model.train() if phase == "train" else model.eval()
            running_loss, correct = 0.0, 0
            for imgs, labels in loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                opt.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    out = model(imgs)
                    loss = criterion(out, labels)
                    if phase == "train": loss.backward(); opt.step()
                preds = out.argmax(1)
                running_loss += loss.item() * imgs.size(0)
                correct += torch.sum(preds == labels)
            epoch_loss = running_loss / len(loader.dataset)
            epoch_acc = correct.double() / len(loader.dataset)
            if phase == "train":
                train_loss_hist.append(epoch_loss)
                train_acc_hist.append(epoch_acc.item())
            else:
                val_loss_hist.append(epoch_loss)
                val_acc_hist.append(epoch_acc.item())
                scheduler.step(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))
                    patience_counter = 0
                else:
                    patience_counter += 1
        if patience_counter >= patience: break

    # save curves
    plt.figure();
    plt.plot(train_loss_hist, label='Train');
    plt.plot(val_loss_hist, label='Val');
    plt.legend();
    plt.title('Loss');
    plt.savefig(os.path.join(out_dir, "loss_curve.png"))
    plt.figure();
    plt.plot(train_acc_hist, label='Train');
    plt.plot(val_acc_hist, label='Val');
    plt.legend();
    plt.title('Accuracy');
    plt.savefig(os.path.join(out_dir, "acc_curve.png"))

    model.load_state_dict(torch.load(os.path.join(out_dir, "best_model.pt")))
    test_eval(model, test_loader, out_dir)


def test_eval(model, loader, out_dir):
    model.eval();
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = model(imgs)
            prob = torch.softmax(out, 1)[:, 1]
            pred = torch.argmax(out, 1)
            y_true += labels.cpu().numpy().tolist()
            y_pred += pred.cpu().numpy().tolist()
            y_prob += prob.cpu().numpy().tolist()
    pd.DataFrame({"true": y_true, "pred": y_pred, "prob": y_prob}).to_csv(os.path.join(out_dir, "test_predictions.csv"),
                                                                          index=False)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=["Co", "Pt"], yticklabels=["Co", "Pt"]);
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"));
    plt.close()
    fpr, tpr, _ = roc_curve(y_true, y_prob);
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}");
    plt.plot([0, 1], [0, 1], '--');
    plt.legend();
    plt.savefig(os.path.join(out_dir, "roc_curve.png"));
    plt.close()
    with open(os.path.join(out_dir, "report.txt"), "w") as f:
        f.write(classification_report(y_true, y_pred, target_names=["Co", "Pt"]))
        f.write(f"\nAUC: {roc_auc:.4f}")
    print(f"✅ Test AUC: {roc_auc:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True);
    p.add_argument("--img", required=True)
    p.add_argument("--out", required=True);
    p.add_argument("--mode", choices=["baseline", "balanced"], default="baseline")
    args = p.parse_args()
    train_one(args.csv, args.img, args.out, balance=(args.mode == "balanced"))
