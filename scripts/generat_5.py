# -*- coding: utf-8 -*-
"""
Generate all 5 figures required:
1. Acoustic signal
2. Temporal fine structure (TFS)
3. Phase plot (TFS)
4. Phase plot (rawphase)
5. Heatmap comparison: rawphase vs TFS

Parameters:
- Interpolation = 5000
- Gaussian sigma = 12
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
import os

# ========= PARAMETERS =========
#DATA_PATH = "/home/woody/iwi5/iwi5325h/gaitphasecnn_raw_data/raw/GaCo05_01.txt"
#DATA_PATH = "/Users/wangyulin/Time Series/629/gait-phase-cnn/data/raw/GaCo05_01.txt"
# -*- coding: utf-8 -*-
"""
Final fixed version of the 5-figure generation script
with correct Temporal Fine Structure (TFS) plot.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
import os

# ========= PARAMETERS =========
DATA_PATH = "/Users/wangyulin/Time Series/629/gait-phase-cnn/data/raw/GaCo05_01.txt"
SIGMA = 12
INTERP_POINTS = 5000
BINS = 248
PAD = 8

SAVE_DIR = "./fig_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

# ========= Helper functions =========
def read_gait_data(path):
    data = np.loadtxt(path)
    t = data[:, 0]
    sensors = data[:, 1:17]
    sensors = sensors - np.mean(sensors, axis=0)
    sensors = sensors / np.max(np.abs(sensors), axis=0)
    return t, sensors

def get_signal(sensors):
    s = np.sqrt(np.sum(sensors**2, axis=1))
    s = s - np.mean(s)
    s = s / np.max(np.abs(s))
    return s

def inter2D(points, n_points):
    d = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
    d = np.insert(d, 0, 0)
    d = d / d[-1]
    alpha = np.linspace(0, 1, n_points)
    f = interp1d(d, points, kind='cubic', axis=0)
    return f(alpha)

def compute_heatmap(signal):
    analytic = hilbert(signal)
    x, y = analytic.real, analytic.imag
    pts = np.vstack([x, y]).T

    interp_pts = inter2D(pts, INTERP_POINTS)
    xi, yi = interp_pts[:, 0], interp_pts[:, 1]

    H, _, _ = np.histogram2d(xi, yi, bins=BINS)

    p = int(PAD/2)
    final = np.zeros([BINS + PAD, BINS + PAD])
    final[p:-p, p:-p] = H
    return gaussian_filter(final, sigma=SIGMA).T


# ========= Load data =========
t, sensors = read_gait_data(DATA_PATH)
signal = get_signal(sensors)

# analytic signals
z = hilbert(signal)
env = np.abs(z)
tfs = signal / env            # <-- correct TFS
analytic_tfs = hilbert(tfs)   # for phase plot
analytic_raw = z


# heatmaps
heat_raw = compute_heatmap(signal)
heat_tfs = compute_heatmap(tfs)

vmin = min(heat_raw.min(), heat_tfs.min())
vmax = max(heat_raw.max(), heat_tfs.max())


# ============================================
# FIGURE 1: Acoustic signal
# ============================================
plt.figure(figsize=(6,4))
plt.plot(t, signal, color="black", linewidth=1)
plt.title("Acoustic signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/fig1_acoustic_signal.png", dpi=300)


# ============================================
# FIGURE 2: Correct Temporal Fine Structure
# ============================================
plt.figure(figsize=(6,4))
plt.plot(t, tfs, color="black", linewidth=1)
plt.title("Temporal fine structure")
plt.xlabel("Time [s]")
plt.ylabel("Normalized amplitude")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/fig2_tfs_correct.png", dpi=300)


# ============================================
# FIGURE 3: Phase plot (TFS)
# ============================================
plt.figure(figsize=(6,4))
plt.plot(analytic_tfs.imag, analytic_tfs.real, color="black", linewidth=0.7)
plt.title("Phase plot (TFS)")
plt.xlabel("Imaginary axis")
plt.ylabel("Real axis")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/fig3_phaseplot_tfs.png", dpi=300)


# ============================================
# FIGURE 4: Phase plot (rawphase)
# ============================================
plt.figure(figsize=(6,4))
plt.plot(analytic_raw.imag, analytic_raw.real, color="black", linewidth=0.7)
plt.title("Phase plot (rawphase)")
plt.xlabel("Imaginary axis")
plt.ylabel("Real axis")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/fig4_phaseplot_raw.png", dpi=300)


# ============================================
# FIGURE 5: Heatmap comparison (no shared vmin/vmax)
# ============================================
plt.figure(figsize=(10,4))

# rawphase
plt.subplot(1,2,1)
plt.imshow(heat_raw, cmap="hot", origin="lower")   # 自动使用自身 min/max
plt.title("Phase plot heatmap (rawphase)")
plt.axis("off")

# TFS
plt.subplot(1,2,2)
plt.imshow(heat_tfs, cmap="hot", origin="lower")   # 自动使用自身 min/max
plt.title("Phase plot heatmap (TFS)")
plt.axis("off")

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/fig5_heatmap_compare.png", dpi=300)


print("\n🎉 All 5 corrected figures saved to:", SAVE_DIR)
