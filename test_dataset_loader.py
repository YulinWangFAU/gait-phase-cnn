# cnn_test_pipeline.py
#如何运行python cnn_test_pipeline.py --log_dir data/runs/hilbert_tfs_cnn_i3000_s8_win800_step400/cnn_model_bn
import os
import csv
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from datasets.heatmap_dataset import HeatmapDataset
from models.cnn_model_bn import CNNModel  # ← 根据你使用的模型修改
from config import Config

# === Argument Parser ===
parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', type=str, required=True, help='Path to training log directory')
args = parser.parse_args()

# === Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_dir = args.log_dir
model_path = Config.MODEL_SAVE_PATH  # 由 Config 控制存储模型路径

# === Dataset ===
dataset = HeatmapDataset(Config.LABEL_CSV_PATH)
val_size = int(len(dataset) * Config.VAL_SPLIT)
test_size = val_size  # 20%
train_size = len(dataset) - val_size - test_size
train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

# === Load Model ===
model = CNNModel().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# === Evaluate on Test Set ===
criterion = nn.CrossEntropyLoss()
test_loss, test_correct = 0.0, 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = criterion(output, y)
        test_loss += loss.item()
        test_correct += (output.argmax(dim=1) == y).sum().item()

test_acc = test_correct / len(test_ds)
test_loss = test_loss / len(test_loader)

# === Save Metrics ===
os.makedirs(log_dir, exist_ok=True)
test_csv_path = os.path.join(log_dir, "test_metrics.csv")
with open(test_csv_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['metric', 'value'])
    writer.writerow(['test_accuracy', f"{test_acc:.4f}"])
    writer.writerow(['test_loss', f"{test_loss:.6f}"])

print(f"\n✅ Test Accuracy: {test_acc:.4f}")
print(f"📄 Saved test results to: {test_csv_path}")
