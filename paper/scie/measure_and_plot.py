"""Measure the implemented EvoMoE model and the files on disk, then draw figures.

Numbers written here are computed in this run. They are not taken from earlier reports.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "research" / "evonex" / "src"
sys.path.insert(0, str(SRC))

from model_evonex import EvoMoE_Model  # noqa: E402

OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "axes.linewidth": 0.6,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 160,
        "savefig.dpi": 200,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

INK = "#1c2833"
STEEL = "#1a5276"
RUST = "#922b21"
GREEN = "#196f3d"
OCHRE = "#9a7d0a"
GRID = "#d5d8dc"


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def measure_model() -> dict:
    torch.manual_seed(42)
    model = EvoMoE_Model()
    model.eval()

    groups = {
        "Multi-scale CNN": model.cnn_expert,
        "Temporal Transformer": model.transformer_expert,
        "Physics MLP": model.physics_expert,
        "Shared classifier": model.classifier,
    }
    breakdown = {name: count_params(mod) for name, mod in groups.items()}
    total = count_params(model)

    lc = torch.randn(1, 2000)
    tic = torch.randn(1, 12)
    with torch.no_grad():
        logits, weights = model(lc, tic)

    # Latency on this CPU. Warm-up excluded. Batch size 1.
    with torch.no_grad():
        for _ in range(20):
            model(lc, tic)
        samples = []
        for _ in range(100):
            t0 = time.perf_counter()
            model(lc, tic)
            samples.append((time.perf_counter() - t0) * 1000.0)

    samples = np.array(samples)
    return {
        "seed": 42,
        "total_parameters": int(total),
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "breakdown": breakdown,
        "logit_shape": list(logits.shape),
        "weight_shape": list(weights.shape),
        "weight_sum": float(weights.sum().item()),
        "latency_ms": {
            "n": 100,
            "batch": 1,
            "device": "cpu",
            "median": float(np.median(samples)),
            "p05": float(np.percentile(samples, 5)),
            "p95": float(np.percentile(samples, 95)),
            "mean": float(np.mean(samples)),
        },
    }


def curve_stats(path: Path) -> dict:
    df = pd.read_csv(path)
    cols = list(df.columns)
    xcol = "time" if "time" in cols else ("phase" if "phase" in cols else cols[0])
    ycol = "flux" if "flux" in cols else cols[1]
    x = df[xcol].to_numpy(dtype=float)
    y = df[ycol].to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    baseline = float(np.median(y))
    ymin = float(np.min(y))
    depth = (baseline - ymin) * 1e6
    # Noise from the upper half of the flux distribution, away from the dip.
    upper = y[y >= np.quantile(y, 0.6)]
    noise = float(np.std(upper)) if len(upper) > 2 else float("nan")
    return {
        "n": int(len(y)),
        "columns": cols,
        "xcol": xcol,
        "ycol": ycol,
        "x_min": float(np.min(x)),
        "x_max": float(np.max(x)),
        "flux_min": ymin,
        "flux_max": float(np.max(y)),
        "flux_median": baseline,
        "measured_depth_ppm": float(depth),
        "upper_std": noise,
        "x": x,
        "y": y,
    }


def measure_library() -> tuple[list[dict], dict]:
    catalog = json.loads((ROOT / "demo" / "demo_catalog.json").read_text(encoding="utf-8"))
    rows = []
    curves = {}
    for item in catalog:
        path = ROOT / "demo" / item["file"]
        stats = curve_stats(path)
        curves[item["id"]] = stats
        row = {
            "id": item["id"],
            "name": item["name"],
            "file": item["file"],
            "category": item["category"],
            "data_mode": item["data_mode"],
            "period_days": item.get("period_days"),
            "catalog_depth_ppm": item.get("depth_ppm"),
            "catalog_snr": item.get("snr"),
            "n": stats["n"],
            "columns": stats["columns"],
            "x_min": stats["x_min"],
            "x_max": stats["x_max"],
            "measured_depth_ppm": stats["measured_depth_ppm"],
            "flux_median": stats["flux_median"],
        }
        if item.get("depth_ppm") not in (None, 0):
            row["depth_ratio"] = stats["measured_depth_ppm"] / float(item["depth_ppm"])
        else:
            row["depth_ratio"] = None
        rows.append(row)

    targets = json.loads((ROOT / "services" / "evonex-api" / "data" / "targets.json").read_text(encoding="utf-8"))
    status_counts: dict[str, int] = {}
    for t in targets:
        status_counts[t.get("status", "UNKNOWN")] = status_counts.get(t.get("status", "UNKNOWN"), 0) + 1

    labels_path = ROOT / "research" / "datasets" / "v2-curated-500" / "labels_raw.csv"
    labels = pd.read_csv(labels_path) if labels_path.exists() else None
    state_path = ROOT / "research" / "evonex" / "datasets" / "v2-curated-500" / "dataset_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}

    h5_files = [str(p.relative_to(ROOT)) for p in (ROOT / "research").rglob("*.h5")]
    weight_files = [str(p.relative_to(ROOT)) for p in (ROOT / "research" / "evonex" / "weights").glob("*") if p.is_file()]

    meta = {
        "n_catalog": len(rows),
        "category_counts": {},
        "data_mode_counts": {},
        "n_targets_json": len(targets),
        "target_status_counts": status_counts,
        "labels_rows": None if labels is None else int(len(labels)),
        "labels_ids": None if labels is None else labels["tic_id"].astype(str).tolist(),
        "labels_classes": None if labels is None else labels["label_class"].tolist(),
        "dataset_state": state,
        "h5_files": h5_files,
        "weight_files": weight_files,
        "trained_weights_present": (ROOT / "research" / "evonex" / "weights" / "evomoe_weights.pth").exists(),
    }
    for row in rows:
        meta["category_counts"][row["category"]] = meta["category_counts"].get(row["category"], 0) + 1
        meta["data_mode_counts"][row["data_mode"]] = meta["data_mode_counts"].get(row["data_mode"], 0) + 1
    return rows, curves, meta


def untrained_forward(curves: dict) -> list[dict]:
    torch.manual_seed(42)
    model = EvoMoE_Model()
    model.eval()
    out = []
    with torch.no_grad():
        for key, stats in curves.items():
            y = stats["y"].astype(np.float32)
            if len(y) >= 2000:
                flux = y[:2000]
            else:
                flux = np.pad(y, (0, 2000 - len(y)), constant_values=1.0)
            lc = torch.tensor(flux, dtype=torch.float32).unsqueeze(0)
            tic = torch.zeros(1, 12)
            logits, weights = model(lc, tic)
            prob = torch.softmax(logits, dim=1)[0, 1].item()
            w = weights[0].tolist()
            out.append(
                {
                    "id": key,
                    "prob_class1": float(prob),
                    "w_cnn": float(w[0]),
                    "w_transformer": float(w[1]),
                    "w_physics": float(w[2]),
                    "tic_input": "zeros",
                    "note": "Random initialization, seed 42. Not a trained detection.",
                }
            )
    return out


def plot_lightcurves(curves: dict) -> None:
    picks = [
        ("HD-209458b", "HD 209458 b"),
        ("Kepler-10b", "Kepler-10 b"),
        ("TOI-700d", "TOI-700 d"),
        ("FP-Grazing-Binary-01", "Grazing binary"),
        ("FP-Stellar-Flare-Noise", "Flare-like series"),
        ("Giant-Star-Deep-False-Alarm", "Deep false alarm"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), sharey=False)
    for ax, (key, title) in zip(axes.ravel(), picks):
        stats = curves[key]
        x = stats["x"]
        y = stats["y"]
        # Display at most ~800 points so the stroke stays readable.
        step = max(1, len(x) // 800)
        ax.plot(x[::step], y[::step], color=STEEL, lw=0.7)
        ax.axhline(stats["flux_median"], color=OCHRE, lw=0.6, ls="--")
        ax.ticklabel_format(useOffset=False, style="plain", axis="y")
        ax.set_title(title, loc="left", color=INK, fontsize=8)
        ax.set_xlabel(stats["xcol"])
        ax.tick_params(length=2)
        ax.grid(True, color=GRID, lw=0.4)
        for spine in ax.spines.values():
            spine.set_color("#85929e")
    axes[0, 0].set_ylabel("flux")
    axes[1, 0].set_ylabel("flux")
    fig.tight_layout()
    fig.savefig(FIG / "fig_lightcurves.pdf")
    fig.savefig(FIG / "fig_lightcurves.png")
    plt.close(fig)


def plot_params(breakdown: dict) -> None:
    names = list(breakdown.keys())
    vals = np.array([breakdown[n] for n in names], dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    colors = [STEEL, "#1a5276", RUST, GREEN]
    bars = ax.barh(names[::-1], (vals / 1e6)[::-1], color=colors[::-1], height=0.62)
    ax.set_xlabel("Trainable parameters (millions)")
    ax.grid(True, axis="x", color=GRID, lw=0.4)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for bar, value in zip(bars, vals[::-1]):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2, f"{int(value):,}", va="center", fontsize=8, color=INK)
    ax.set_xlim(0, max(vals) / 1e6 * 1.28)
    fig.tight_layout()
    fig.savefig(FIG / "fig_parameters.pdf")
    fig.savefig(FIG / "fig_parameters.png")
    plt.close(fig)


def plot_depth(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    color = {
        "CONFIRMED_PLANET": STEEL,
        "FALSE_POSITIVE": RUST,
        "BORDERLINE_CANDIDATE": OCHRE,
    }
    marker = {"REAL_REFERENCE": "o", "SYNTHETIC_MODEL": "s"}
    for row in rows:
        cat = row["catalog_depth_ppm"]
        meas = row["measured_depth_ppm"]
        if cat is None:
            continue
        ax.scatter(
            max(cat, 1e-3),
            max(meas, 1e-3),
            c=color.get(row["category"], INK),
            marker=marker.get(row["data_mode"], "o"),
            s=28,
            zorder=3,
        )
    limits = [1, 2e5]
    ax.plot(limits, limits, color="#7f8c8d", lw=0.7, ls="--", zorder=1, label="equal depths")
    ax.scatter([], [], c=STEEL, marker="o", s=28, label="Planet, archive-style")
    ax.scatter([], [], c=STEEL, marker="s", s=28, label="Planet, synthetic")
    ax.scatter([], [], c=RUST, marker="o", s=28, label="False positive, archive-style")
    ax.scatter([], [], c=RUST, marker="s", s=28, label="False positive, synthetic")
    ax.scatter([], [], c=OCHRE, marker="s", s=28, label="Threshold, synthetic")
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(20, 2e5)
    ax.set_ylim(20, 2e5)
    ax.set_xlabel("Design depth (ppm)")
    ax.set_ylabel("Measured peak-to-median depth (ppm)")
    ax.grid(True, which="both", color=GRID, lw=0.4)
    fig.tight_layout()
    fig.savefig(FIG / "fig_depth_comparison.pdf")
    fig.savefig(FIG / "fig_depth_comparison.png")
    plt.close(fig)


def main() -> None:
    model = measure_model()
    rows, curves, meta = measure_library()
    forwards = untrained_forward(curves)
    plot_lightcurves(curves)
    plot_params(model["breakdown"])
    plot_depth(rows)

    # Drop arrays before serialization.
    serial_rows = rows
    payload = {
        "platform": {
            "system": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "model": model,
        "library_meta": meta,
        "curves": serial_rows,
        "untrained_forward": forwards,
    }
    (OUT / "measurements.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"params": model["total_parameters"], "latency_median_ms": model["latency_ms"]["median"], "n_curves": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
