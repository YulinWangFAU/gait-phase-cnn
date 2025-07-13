# -*- coding: utf-8 -*-
"""
Created on 2025/7/13 16:19

@author: Yulin Wang
@email: yulin.wang@fau.de
"""

# pca_visualize.py

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image


def load_images_and_labels(csv_path, img_size=(64, 64), max_samples=None):
    df = pd.read_csv(csv_path)
    if max_samples:
        df = df.sample(n=max_samples, random_state=42)
    X, y = [], []

    for _, row in df.iterrows():
        try:
            img = Image.open(row["filename"]).convert("L").resize(img_size)
            arr = np.array(img).flatten() / 255.0
            X.append(arr)
            y.append(row["label"])
        except Exception as e:
            print(f"Error loading {row['filename']}: {e}")
    return np.array(X), np.array(y)


def run_pca_visualization(csv_path, img_size=(64, 64), max_samples=None, output="pca_plot.png"):
    print(f"📊 Loading data from: {csv_path}")
    X, y = load_images_and_labels(csv_path, img_size, max_samples)

    print(f"✅ Loaded {len(X)} samples, running PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    for label in np.unique(y):
        plt.scatter(X_pca[y == label, 0], X_pca[y == label, 1],
                    label=f"Class {label}", alpha=0.6)
    plt.title("PCA Projection of Heatmap Dataset")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output)
    print(f"📎 PCA plot saved to: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to labels_*.csv file")
    parser.add_argument("--img_size", type=int, default=64, help="Resize images to this square size")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to load for PCA")
    parser.add_argument("--output", type=str, default="pca_plot.png", help="Output image filename")
    args = parser.parse_args()

    run_pca_visualization(
        csv_path=args.csv,
        img_size=(args.img_size, args.img_size),
        max_samples=args.max_samples,
        output=args.output
    )
