# -*- coding: utf-8 -*-
"""
Rawphase vs TFS Phase Plot Pipeline Comparison
Matches EXACTLY the implementation in generate_heatmaps_auto_split_all_balanced.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter

# ---------------------------------
# CONFIG
# ---------------------------------
RAW_FILE = "/Users/wangyulin/Time Series/629/gait-phase-cnn/data/raw/GaCo05_01.txt"
SIGMA = 10
INTERP_POINTS = 4000
BINS = 248
PAD = 8


# ---------------------------------
# Helpers (exact from your code)
# ---------------------------------
def read_gait_data(filename):
    data = np.loadtxt(filename)
    sensors = data[:, 1:17]
    sensors = sensors - np.mean(sensors, axis=0)
    sensors = sensors / np.max(np.abs(sensors), axis=0)
    return sensors


def get_gait_signal(sensors):
    signal = np.sqrt(np.sum(sensors ** 2, axis=1))
    signal = signal - np.mean(signal)
    signal = signal / np.max(np.abs(signal))
    return signal


def inter2D(points, n_points):
    dist = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
    dist = np.insert(dist, 0, 0)
    dist = dist / dist[-1]
    alpha = np.linspace(0, 1, n_points)
    interp = interp1d(dist, points, kind='cubic', axis=0)
    return interp(alpha)


def compute_phase(signal, method="rawphase"):
    if method == "tfs":
        analytic = hilbert(signal)
        env = np.abs(analytic)
        tfs = signal / env
        analytic = hilbert(tfs)
    else:
        analytic = hilbert(signal)
    return analytic.real, analytic.imag


def make_heatmap(x, y):
    pts = np.vstack([x, y]).T
    pts_i = inter2D(pts, INTERP_POINTS)
    x_i, y_i = pts_i[:, 0], pts_i[:, 1]

    hist, _, _ = np.histogram2d(x_i, y_i, bins=BINS)
    p = PAD // 2
    hmap = np.zeros((BINS + PAD, BINS + PAD))
    hmap[p:-p, p:-p] = hist
    hmap = gaussian_filter(hmap, sigma=SIGMA).T

    return pts, pts_i, hmap


# ---------------------------------
# RUN PIPELINE
# ---------------------------------
sensors = read_gait_data(RAW_FILE)
signal = get_gait_signal(sensors)

# rawphase
xr, yr = compute_phase(signal, method="rawphase")
pts_r, ptsi_r, heat_r = make_heatmap(xr, yr)

# TFS
xt, yt = compute_phase(signal, method="tfs")
pts_t, ptsi_t, heat_t = make_heatmap(xt, yt)

# ---------------------------------
# PLOT
# ---------------------------------
plt.figure(figsize=(13, 14))

# ----- Rawphase -----
plt.subplot(4, 2, 1)
plt.plot(signal[:1500], linewidth=0.7)
plt.title("Raw Gait Signal")

plt.subplot(4, 2, 3)
plt.plot(xr, yr, linewidth=0.3)
plt.title("Rawphase: Phase Plot")

plt.subplot(4, 2, 5)
plt.plot(ptsi_r[:, 0], ptsi_r[:, 1], linewidth=0.3)
plt.title("Rawphase: Interpolated Trajectory")

plt.subplot(4, 2, 7)
plt.imshow(heat_r, cmap="hot", origin="lower")
plt.title("Rawphase: Final Heatmap")
plt.colorbar()

# ----- TFS -----
plt.subplot(4, 2, 2)
plt.plot(signal[:1500], linewidth=0.7)
plt.title("Raw Gait Signal (same)")

plt.subplot(4, 2, 4)
plt.plot(xt, yt, linewidth=0.3)
plt.title("TFS: Phase Plot")

plt.subplot(4, 2, 6)
plt.plot(ptsi_t[:, 0], ptsi_t[:, 1], linewidth=0.3)
plt.title("TFS: Interpolated Trajectory")

plt.subplot(4, 2, 8)
plt.imshow(heat_t, cmap="hot", origin="lower")
plt.title("TFS: Final Heatmap")
plt.colorbar()

plt.tight_layout()
plt.show()
