"""Redraw two figures with publication axis titles. Does not retime the model."""
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(r"C:\projects\AstroVerse")
FIG = ROOT / "paper" / "scie" / "figures"
STEEL = "#1a5276"
OCHRE = "#9a7d0a"
GRID = "#d5d8dc"
INK = "#1c2833"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "axes.linewidth": 0.6,
    }
)


def lightcurves():
    picks = [
        ("HD-209458b", "HD 209458 b"),
        ("Kepler-10b", "Kepler-10 b"),
        ("TOI-700d", "TOI-700 d"),
        ("FP-Grazing-Binary-01", "Grazing binary"),
        ("FP-Stellar-Flare-Noise", "Flare-like series"),
        ("Giant-Star-Deep-False-Alarm", "Giant-host dip"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4))
    for ax, (key, title) in zip(axes.ravel(), picks):
        df = pd.read_csv(ROOT / "demo" / f"{key}.csv")
        x = df["time"].to_numpy(float)
        y = df["flux"].to_numpy(float)
        step = max(1, len(x) // 800)
        ax.plot(x[::step], y[::step], color=STEEL, lw=0.7)
        ax.axhline(float(np.median(y)), color=OCHRE, lw=0.6, ls="--")
        ax.ticklabel_format(useOffset=False, style="plain", axis="y")
        ax.set_title(title, loc="left", color=INK, fontsize=8)
        span = float(np.nanmax(x) - np.nanmin(x))
        ax.set_xlabel("phase" if span < 2 else "time (days)")
        ax.grid(True, color=GRID, lw=0.4)
    axes[0, 0].set_ylabel("flux")
    axes[1, 0].set_ylabel("flux")
    fig.tight_layout()
    fig.savefig(FIG / "fig_lightcurves.png", dpi=200)
    fig.savefig(FIG / "fig_lightcurves.pdf")
    plt.close(fig)


def cached():
    path = ROOT / "research" / "evonex" / "datasets" / "v2-curated-500" / "tess_cache.h5"
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4))
    with h5py.File(path, "r") as f:
        keys = sorted(f.keys(), key=int)
        for ax, key in zip(axes.ravel(), keys):
            y = np.asarray(f[key]["lc_flux"][:], dtype=float)
            x = np.linspace(0, 1, len(y), endpoint=False)
            step = max(1, len(y) // 800)
            ax.plot(x[::step], y[::step], color=STEEL, lw=0.6)
            ax.set_title(f"TIC {key}", loc="left", color=INK, fontsize=8)
            ax.ticklabel_format(useOffset=False, style="plain", axis="y")
            ax.grid(True, color=GRID, lw=0.4)
            ax.set_xlabel("sample / 2000")
    axes[0, 0].set_ylabel("flux")
    axes[1, 0].set_ylabel("flux")
    fig.tight_layout()
    fig.savefig(FIG / "fig_cached_curves.png", dpi=200)
    fig.savefig(FIG / "fig_cached_curves.pdf")
    plt.close(fig)


if __name__ == "__main__":
    lightcurves()
    cached()
    print("redrawn")
