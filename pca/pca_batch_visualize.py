# -*- coding: utf-8 -*-
"""
PCA batch visualization for multiple heatmap datasets
@author: Yulin Wang
@email: yulin.wang@fau.de
"""

import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from PIL import Image

# === Config ===
LABEL_FILES = [
    "labels_win800_step200.csv",
    "labels_win800_step400.csv",
    "labels_win200_step100.csv",
    "labels_win200_step50.csv",
    "labels_win400_step200.csv",
    "labels_win100_step25.csv",
    "labels_fullsignal.csv"
]

BASE_DIR = "/home/woody/rlvl/rlvl144v/gaitphasecnn_raw_data"
OUTPUT_DIR = "/home/woody/rlvl/rlvl144v/gait-phase-cnn/logs/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = 64  # resize all images to 64x64

# === Main Function ===
def run_pca(csv_path, output_path, img_size=64, max_samples=1000):
    df = pd.read_csv(csv_path)
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)

    X, y = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=os.path.basename(csv_path)):
        try:
            img = Image.open(row['filename']).convert('L').resize((img_size, img_size))
            X.append(np.array(img).flatten())
            y.append(int(row['label']))
        except Exception as e:
            print(f"⚠️  Skipped {row['filename']} due to error: {e}")
            continue

    X = np.array(X)
    y = np.array(y)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    for label in np.unique(y):
        idx = y == label
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=f'Class {label}', alpha=0.6, s=10)
    plt.legend()
    plt.title(f'PCA Projection: {os.path.basename(csv_path)}')
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Saved PCA to: {output_path}")

# === Run Batch ===
if __name__ == "__main__":
    for csv_file in LABEL_FILES:
        csv_path = os.path.join(BASE_DIR, csv_file)
        output_file = f"pca_{csv_file.replace('labels_', '').replace('.csv', '')}.png"
        output_path = os.path.join(OUTPUT_DIR, output_file)
        run_pca(csv_path, output_path, img_size=IMG_SIZE)
