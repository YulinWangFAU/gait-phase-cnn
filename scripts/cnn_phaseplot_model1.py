# -*- coding: utf-8 -*-
"""
Created on 2025/11/17 16:05

@author: Yulin Wang
@email: yulin.wang@fau.de
"""
# -*- coding: utf-8 -*-
"""
cnn_phaseplot_model.py
Calvo-Ariza (2020) CNN for Phase Plot Classification
Updated:
    ✔ Uses subject_split.json for split (reproducible split)
    ✔ Ignores CSV split column
    ✔ Determines split based on subject ID via JSON
"""

import os, argparse, random, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from PIL import Image
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc


# ---------------- Global Setup ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# ---------------- Load split.json ----------------
def load_split_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"❌ subject_split.json not found: {path}\n"
            f"You must run the heatmap generation script FIRST."
        )
    with open(path, "r") as f:
        split_dict = json.load(f)
    print(f"📁 Loaded fixed subject split: {path}")
    return split_dict


# ---------------- Dataset ----------------
class GaitHeatmapDataset(Dataset):
    def __init__(self, df, img_dir, split_name, split_json, transform=None):
        """
        df: CSV data
        img_dir: directory of heatmaps
        split_name: 'train' / 'val' / 'test'
        split_json: loaded subject_split.json
        """
        self.img_dir = os.path.join(img_dir, split_name)
        self.transform = transform

        # auto detect EXP & TASK from folder name
        # e.g. heatmaps_rawphase_left_σ8_i2000_Ga_normal_balanced
        folder = os.path.basename(img_dir)
        parts = folder.split("_")
        EXP, TASK = parts[-3], parts[-2]
        key = f"{EXP}_{TASK}"

        allowed_subjects = set(split_json[key][split_name])

        rows = []
        for i in range(len(df)):
            fname = df.iloc[i]["filename"]
            sid = fname[:6]  # GaCo01
            if sid in allowed_subjects:
                rows.append(df.iloc[i])

        self.data = pd.DataFrame(rows).reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = Image.open(os.path.join(self.img_dir, row["filename"])).convert("L")
        label = 1 if row["group"] == "Pt" else 0
        return self.transform(img), label


# ---------------- Calvo-Ariza CNN Model ----------------
class CalvoCNN(nn.Module):
    def __init__(self, fc_dim=128, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 32, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # auto compute shape
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 248, 248)
            out = self.features(dummy)
            self.flatten_dim = out.numel()
        print(f"[CalvoCNN] Flatten dim = {self.flatten_dim}")

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, fc_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fc_dim, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------------- Train & Eval ----------------
def train_one(csv_path, img_dir, out_dir,
              split_json_path,
              balance=False, epochs=100,
              batch_size=4, lr=1e-4, patience=5, fc_dim=128):

    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    # -------------- Load fixed split --------------
    split_json = load_split_json(split_json_path)

    # -------------- transforms --------------
    transform = transforms.Compose([
        transforms.Resize((248, 248)),
        transforms.ToTensor()
    ])

    # -------------- dataset --------------
    train_ds = GaitHeatmapDataset(df, img_dir, "train", split_json, transform)
    val_ds   = GaitHeatmapDataset(df, img_dir, "val",   split_json, transform)
    test_ds  = GaitHeatmapDataset(df, img_dir, "test",  split_json, transform)

    # -------------- loader --------------
    if balance:
        train_labels = np.array([1 if g=="Pt" else 0 for g in train_ds.data["group"]])
        class_weight = compute_class_weight("balanced", [0,1], train_labels)
        class_weight = torch.tensor(class_weight, dtype=torch.float).to(DEVICE)
        sample_weights = np.array([class_weight[int(l)].item() for l in train_labels])
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
        criterion = nn.CrossEntropyLoss(weight=class_weight)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()

    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # -------------- model --------------
    model = CalvoCNN(fc_dim=fc_dim).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5, patience=3)

    # -------------- training loop --------------
    best_acc, patience_counter = 0, 0
    train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist = [], [], [], []

    print(f"\n🚀 Training start ... FC={fc_dim}, balance={balance}\n")

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
                    if phase == "train":
                        loss.backward()
                        opt.step()
                preds = out.argmax(1)
                running_loss += loss.item() * imgs.size(0)
                correct += (preds == labels).sum().item()

            epoch_loss = running_loss / len(loader.dataset)
            epoch_acc = correct / len(loader.dataset)

            if phase == "train":
                train_loss_hist.append(epoch_loss)
                train_acc_hist.append(epoch_acc)
            else:
                val_loss_hist.append(epoch_loss)
                val_acc_hist.append(epoch_acc)
                scheduler.step(epoch_loss)

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))
                    patience_counter = 0
                else:
                    patience_counter += 1

        print(f"Epoch {epoch+1:03d} | Train Acc={train_acc_hist[-1]:.3f} | Val Acc={val_acc_hist[-1]:.3f}")

        if patience_counter >= patience:
            print("⏳ Early stopping.")
            break

    # -------- curves --------
    plt.figure(); plt.plot(train_loss_hist); plt.plot(val_loss_hist); plt.title("Loss")
    plt.savefig(os.path.join(out_dir, "loss_curve.png"))

    plt.figure(); plt.plot(train_acc_hist); plt.plot(val_acc_hist); plt.title("Accuracy")
    plt.savefig(os.path.join(out_dir, "acc_curve.png"))

    # ============ Evaluate =============
    model.load_state_dict(torch.load(os.path.join(out_dir, "best_model.pt")))
    test_eval(model, test_loader, out_dir)


# ---------------- Evaluation ----------------
def test_eval(model, loader, out_dir):
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = model(imgs)
            prob = torch.softmax(out, 1)[:, 1]
            pred = out.argmax(1)

            y_true += labels.cpu().numpy().tolist()
            y_pred += pred.cpu().numpy().tolist()
            y_prob += prob.cpu().numpy().tolist()

    # Save predictions
    pd.DataFrame({"true": y_true, "pred": y_pred, "prob": y_prob}).to_csv(
        os.path.join(out_dir, "test_predictions.csv"), index=False
    )

    # ============ Confusion Matrix ===============
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5,4))        # <--- new: new canvas
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=["Co","Pt"], yticklabels=["Co","Pt"],
                cmap="rocket_r")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
    plt.close()                      # <--- new: close figure


    # ============ ROC Curve ===============
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5,4))        # <--- new
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_curve.png"))
    plt.close()                      # <--- new


    # ============ Classification Report ===============
    with open(os.path.join(out_dir, "report.txt"), "w") as f:
        f.write(classification_report(y_true, y_pred, target_names=["Co","Pt"]))
        f.write(f"\nAUC: {roc_auc:.4f}\n")

    print(f"🎯 Test AUC = {roc_auc:.4f}")
    print(f"Results saved in: {out_dir}")


# ---------------- Main ----------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--img", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split_json", required=True)   # <-- NEW ARGUMENT
    p.add_argument("--mode", choices=["baseline","balanced"], default="baseline")
    p.add_argument("--fc_dim", type=int, default=128)
    args = p.parse_args()

    train_one(
        args.csv, args.img, args.out,
        split_json_path=args.split_json,
        balance=(args.mode == "balanced"),
        fc_dim=args.fc_dim
    )
