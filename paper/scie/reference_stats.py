"""Signal properties of the 18 reference curves, plus a three-panel morphology figure."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(r"C:\projects\AstroVerse")
OUT = Path(r"C:\projects\AstroVerse\paper\scie")
FIG = OUT / "figures"
catalog = json.loads((ROOT / "demo" / "demo_catalog.json").read_text(encoding="utf-8"))

rows = []
curves = {}
for item in catalog:
    df = pd.read_csv(ROOT / "demo" / item["file"])
    x = df["time"].to_numpy(float)
    y = df["flux"].to_numpy(float)
    med = float(np.median(y))
    depth = med - float(np.min(y))
    # Noise from the brighter half of the samples, away from the deepest points.
    bright = y[y >= np.quantile(y, 0.55)]
    sigma = float(np.std(bright))
    snr = depth / sigma if sigma > 0 else float("nan")
    rows.append(
        {
            "id": item["id"],
            "host": item.get("stellar_type", ""),
            "category": item["category"],
            "period_days": item.get("period_days"),
            "depth_ppm": round(depth * 1e6, 1),
            "snr": round(snr, 2),
            "mode": item.get("data_mode"),
        }
    )
    curves[item["id"]] = (x, y, med)

(OUT / "reference_set.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "axes.linewidth": 0.6,
    }
)
picks = [
    ("Kepler-10b", "Kepler-10 b"),
    ("WASP-126b", "WASP-126 b"),
    ("FalsePositive-EB", "Contact eclipsing binary"),
]
fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.55))
for ax, (key, title) in zip(axes, picks):
    x, y, med = curves[key]
    step = max(1, len(x) // 700)
    ax.plot(x[::step], y[::step], color="#1a5276", lw=0.7)
    ax.axhline(med, color="#9a7d0a", lw=0.6, ls="--")
    ax.set_title(title, loc="left", fontsize=8.5)
    ax.set_xlabel("time")
    ax.ticklabel_format(useOffset=False, style="plain", axis="y")
    ax.grid(True, color="#d5d8dc", lw=0.4)
axes[0].set_ylabel("flux")
fig.tight_layout()
fig.savefig(FIG / "fig_cases.png", dpi=200)
fig.savefig(FIG / "fig_cases.pdf")
print("n", len(rows))
