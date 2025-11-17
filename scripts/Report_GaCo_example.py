# -*- coding: utf-8 -*-
"""
Phase-Plot Pipeline Visualization (Matches your heatmap generation EXACTLY)
Raw gait signal → Hilbert (rawphase) → Interpolation → 2D histogram + padding → Gaussian → Heatmap
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter

# ========= CONFIG =========
RAW_FILE = "/Users/wangyulin/Time Series/629/gait-phase-cnn/data/raw/GaCo05_01.txt"

SIGMA = 10
INTERP_POINTS = 4000
BINS = 248
PAD = 8
# ==========================


def read_gait_data(filename):
    data = np.loadtxt(filename)
    time = data[:, 0]
    sensors = data[:, 1:17]

    # EXACTLY as your code
    sensors = sensors - np.mean(sensors, axis=0)
    sensors = sensors / np.max(np.abs(sensors), axis=0)

    return time, sensors


def get_gait_signal(sensors):
    """Your 'both' signal: sqrt(sum(VGRF^2)) across all 16 sensors."""
    signal = np.sqrt(np.sum(sensors ** 2, axis=1))
    signal = signal - np.mean(signal)
    signal = signal / np.max(np.abs(signal))
    return signal


def inter2D(points, n_points=4000):
    """Your exact interpolation algorithm."""
    distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
    distance = np.insert(distance, 0, 0)
    distance = distance / distance[-1]

    alpha = np.linspace(0, 1, n_points)
    interp = interp1d(distance, points, kind='cubic', axis=0)
    return interp(alpha)


def get_heat(signal):
    """Your exact heatmap procedure (rawphase)."""
    analytic = hilbert(signal)

    x = analytic.real.copy()
    y = analytic.imag.copy()

    pts = np.vstack([x, y]).T
    pts_i = inter2D(pts, INTERP_POINTS)

    x_i, y_i = pts_i[:, 0], pts_i[:, 1]

    heat, _, _ = np.histogram2d(x_i, y_i, bins=BINS)

    p = PAD // 2
    hmap = np.zeros((BINS + PAD, BINS + PAD))
    hmap[p:-p, p:-p] = heat

    return gaussian_filter(hmap, sigma=SIGMA).T, (x, y), (x_i, y_i)


# ====== RUN PIPELINE ======
t, sensors = read_gait_data(RAW_FILE)
signal = get_gait_signal(sensors)

heatmap, (x, y), (x_i, y_i) = get_heat(signal)


# ====== PLOT FIGURE ======
plt.figure(figsize=(11, 12))

# (1) raw signal
plt.subplot(3, 2, 1)
plt.plot(signal[:1500], linewidth=0.7)
plt.title("Raw Gait Signal (VGRF magnitude)")
plt.xlabel("Samples"); plt.ylabel("Amplitude")

# (2) Hilbert phase plot (raw)
plt.subplot(3, 2, 2)
plt.plot(x, y, linewidth=0.3)
plt.title("Phase Plot Before Interpolation")
plt.xlabel("Real axis"); plt.ylabel("Imag axis")

# (3) interpolated trajectory
plt.subplot(3, 2, 3)
plt.plot(x_i, y_i, linewidth=0.3)
plt.title(f"Interpolated Trajectory ({INTERP_POINTS} points)")
plt.xlabel("Real axis"); plt.ylabel("Imag axis")

# (4) histogram (no smoothing)
plt.subplot(3, 2, 4)
plt.imshow(hmap := np.histogram2d(x_i, y_i, bins=BINS)[0],
           cmap="hot", origin="lower")
plt.title("2D Histogram (Before Gaussian)")
plt.colorbar()

# (5) final heatmap
plt.subplot(3, 1, 3)
plt.imshow(heatmap, cmap="hot", origin="lower")
plt.title(f"Final Heatmap (bins={BINS}, σ={SIGMA})")
plt.colorbar()

plt.tight_layout()
plt.show()
