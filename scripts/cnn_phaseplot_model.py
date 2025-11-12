# -*- coding: utf-8 -*-
"""
cnn_phaseplot_model.py
Calvo-Ariza (2020) CNN for Phase Plot Classification
Trains and evaluates a CNN following the paper structure:
3 conv layers (64, 32, 16 filters), FC layer with 128/256/512 neurons.
"""

import os, argparse, random, numpy as np, pandas as pd
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


# ---------------- Dataset ----------------
class GaitHeatmapDataset(Dataset):
    def __init__(self, df, img_dir, split, transform=None):
        self.data = df[df["split"] == split].reset_index(drop=True)
        self.img_dir = os.path.join(img_dir, split)
        self.transform = transform

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
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 32, kernel_size=2, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # 🔹自动计算 flatten_dim
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
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------------- Train & Eval ----------------
def train_one(csv_path, img_dir, out_dir, balance=False,
              epochs=100, batch_size=4, lr=1e-4, patience=5, fc_dim=128):

    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    transform = transforms.Compose([
        transforms.Resize((248, 248)),
        transforms.ToTensor()
    ])

    train_ds = GaitHeatmapDataset(df, img_dir, "train", transform)
    val_ds = GaitHeatmapDataset(df, img_dir, "val", transform)
    test_ds = GaitHeatmapDataset(df, img_dir, "test", transform)

    # ----- Sampler or Balanced Training -----
    if balance:
        train_labels = np.array([1 if g == "Pt" else 0 for g in df[df["split"] == "train"]["group"]])
        class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=train_labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float)
        sample_weights = np.array([class_weights[int(l)].item() for l in train_labels])
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        class_weights = class_weights.to(DEVICE)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # ----- Model, Optimizer, Scheduler -----
    model = CalvoCNN(fc_dim=fc_dim).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=3)

    best_acc, patience_counter = 0, 0
    train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist = [], [], [], []

    print(f"\n🚀 Training start on {DEVICE.upper()} for {epochs} epochs ...\n")

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

        print(f"Epoch {epoch+1:03d}/{epochs} "
              f"| Train Acc: {train_acc_hist[-1]:.3f} "
              f"| Val Acc: {val_acc_hist[-1]:.3f} "
              f"| Val Loss: {val_loss_hist[-1]:.4f}")

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # ----- Plot Curves -----
    plt.figure(); plt.plot(train_loss_hist, label='Train'); plt.plot(val_loss_hist, label='Val');
    plt.legend(); plt.title('Loss'); plt.savefig(os.path.join(out_dir, "loss_curve.png"))
    plt.figure(); plt.plot(train_acc_hist, label='Train'); plt.plot(val_acc_hist, label='Val');
    plt.legend(); plt.title('Accuracy'); plt.savefig(os.path.join(out_dir, "acc_curve.png"))

    # ----- Final Evaluation -----
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
            pred = torch.argmax(out, 1)
            y_true += labels.cpu().numpy().tolist()
            y_pred += pred.cpu().numpy().tolist()
            y_prob += prob.cpu().numpy().tolist()

    pd.DataFrame({"true": y_true, "pred": y_pred, "prob": y_prob}).to_csv(
        os.path.join(out_dir, "test_predictions.csv"), index=False
    )

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=["Co", "Pt"], yticklabels=["Co", "Pt"])
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
    plt.close()

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], '--')
    plt.legend()
    plt.title("ROC Curve")
    plt.savefig(os.path.join(out_dir, "roc_curve.png"))
    plt.close()

    with open(os.path.join(out_dir, "report.txt"), "w") as f:
        f.write(classification_report(y_true, y_pred, target_names=["Co", "Pt"]))
        f.write(f"\nAUC: {roc_auc:.4f}\n")

    print(f"\n✅ Test AUC: {roc_auc:.4f}")
    print(f"Results saved in: {out_dir}\n")


# ---------------- Main ----------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--img", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=["baseline", "balanced"], default="baseline")
    p.add_argument("--fc_dim", type=int, default=128,
                   help="Fully connected layer size (128/256/512)")
    args = p.parse_args()

    train_one(args.csv, args.img, args.out,
              balance=(args.mode == "balanced"), fc_dim=args.fc_dim)
