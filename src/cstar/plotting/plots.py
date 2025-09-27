from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal


def knee_plot(df: pd.DataFrame):
    fig, ax = plt.subplots()
    ax.set_title("Fairness vs Throughput")
    for name, sub in df.groupby("scheduler"):
        ax.scatter(sub["throughput"], sub["fairness"], s=8, alpha=0.35, label=name)
    ax.set_xlabel("Throughput (Mb/s)")
    ax.set_ylabel("Jain Fairness")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncols=2, frameon=False)
    return fig


def radar_mean(df: pd.DataFrame):
    metrics = ["fairness", "throughput", "unmet", "cvar", "entropy", "stability", "se"]
    means = df.groupby("scheduler")[metrics].mean()
    th = np.linspace(0, 2 * np.pi, len(metrics) + 1)
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    for name, row in means.iterrows():
        vals = np.r_[row.values, row.values[0]]
        ax.plot(th, vals, label=name)
        ax.fill(th, vals, alpha=0.08)
    ax.set_xticks(th[:-1])
    ax.set_xticklabels(metrics, fontsize=8)
    ax.set_title("Mean metrics (radar)", pad=18)
    ax.legend(fontsize=8, bbox_to_anchor=(1.2, 1.05))
    return fig


def cdf_rates(df: pd.DataFrame):
    fig, ax = plt.subplots()
    ax.set_title("Distribution of snapshot throughput")
    for name, sub in df.groupby("scheduler"):
        x = np.sort(sub["throughput"].values)
        y = np.linspace(0, 1, len(x), endpoint=True)
        ax.plot(x, y, label=name)
    ax.set_xlabel("Throughput (Mb/s)")
    ax.set_ylabel("CDF")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncols=2, frameon=False)
    return fig


def spectrogram_demo(save_prefix: str = "figures/spec_demo"):
    """Synthesize IQ with a tone + noise + bursty interferer and produce PSD & spectrogram."""
    fs = 30_000  # Hz
    T = 0.5  # seconds
    t = np.arange(int(fs * T)) / fs
    # clean signal: QPSK-like tone + wide noise
    sig = 0.3 * np.sin(2 * np.pi * 2000 * t) + 0.2 * np.sin(2 * np.pi * 4500 * t)
    noise = 0.5 * np.random.default_rng(0).normal(size=t.size)
    x = sig + noise
    # add narrowband burst
    burst = (t > 0.18) & (t < 0.28)
    x[burst] += 1.0 * np.sin(2 * np.pi * 7000 * t[burst])
    # Welch PSD
    f, Pxx = signal.welch(x, fs=fs, nperseg=1024)
    fig1, ax1 = plt.subplots()
    ax1.semilogy(f, Pxx)
    ax1.set_xlabel("Hz")
    ax1.set_ylabel("PSD")
    ax1.set_title("Welch PSD (demo)")
    fig1.tight_layout()
    fig1.savefig(f"{save_prefix}_psd.png", dpi=150)
    # Spectrogram
    f2, t2, Sxx = signal.spectrogram(x, fs=fs, nperseg=256, noverlap=192)
    fig2, ax2 = plt.subplots()
    im = ax2.pcolormesh(t2, f2, 10 * np.log10(Sxx + 1e-12), shading="auto")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Hz")
    ax2.set_title("Spectrogram (demo)")
    fig2.colorbar(im, ax=ax2, label="dB")
    fig2.tight_layout()
    fig2.savefig(f"{save_prefix}_spectrogram.png", dpi=150)
    return (f"{save_prefix}_psd.png", f"{save_prefix}_spectrogram.png")
