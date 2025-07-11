import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models.cnn_model import SimpleCNN
from datasets.gait_dataset import GaitDataset
from sklearn.model_selection import train_test_split
import pandas as pd

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.001
PATIENCE = 5

# Paths
LABELS_PATH = 'data/labels.csv'
MODEL_SAVE_PATH = 'models/best_model.pth'

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Load dataset
full_df = pd.read_csv(LABELS_PATH)
train_df, val_df = train_test_split(full_df, test_size=0.2, stratify=full_df['label'], random_state=42)
train_df.to_csv("data/train_labels.csv", index=False)
val_df.to_csv("data/val_labels.csv", index=False)

train_dataset = GaitDataset("data/train_labels.csv", transform=transform)
val_dataset = GaitDataset("data/val_labels.csv", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Model
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Early stopping
best_loss = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)

    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(val_loader.dataset)
    val_acc = correct / total

    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        patience_counter = 0
        print("  Saved new best model!")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("  Early stopping triggered!")
            break
