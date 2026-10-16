"""Printable architecture diagrams. ASCII labels so the PDF never depends on SVG parsing."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUT = Path(__file__).resolve().parent / "figures"
INK = "#1c2833"
MUTED = "#5d6d7e"
BLUE = "#1a5276"
RED = "#922b21"
GREEN = "#196f3d"


def box(ax, x, y, w, h, title, lines, edge):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.08",
        linewidth=0.9, edgecolor=edge, facecolor="white",
    )
    ax.add_patch(patch)
    ax.text(x + 0.12, y + h - 0.28, title, fontsize=8.5, color=INK, fontweight="medium", va="top")
    for i, line in enumerate(lines):
        ax.text(x + 0.12, y + h - 0.58 - i * 0.28, line, fontsize=7.4, color=MUTED, va="top")


def arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8),
    )


def system():
    fig, ax = plt.subplots(figsize=(7.4, 4.15))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    fig.patch.set_facecolor("#fbfaf6")
    ax.set_facecolor("#fbfaf6")
    ax.text(0.15, 5.9, "Sources", fontsize=10, color=INK, fontweight="bold")
    ax.text(3.35, 5.9, "Reduction", fontsize=10, color=INK, fontweight="bold")
    ax.text(6.7, 5.9, "Service", fontsize=10, color=INK, fontweight="bold")

    box(ax, 0.15, 4.35, 2.9, 1.25, "MAST photometry", ["lightkurve, SPOC search"], BLUE)
    box(ax, 0.15, 2.85, 2.9, 1.25, "TIC stellar columns", ["astroquery Catalogs"], BLUE)
    box(ax, 0.15, 1.35, 2.9, 1.25, "Disposition labels", ["Exoplanet Archive table"], BLUE)

    box(
        ax, 3.35, 2.55, 3.05, 2.55,
        "What the dataset class runs",
        [
            "1. Divide by the median",
            "2. Savitzky-Golay, window 151",
            "3. Sigma-clip at 5 sigma",
            "4. BLS from 0.5 to 20 days",
            "5. Fold and resample to 2000",
            "6. Write an HDF5 group",
        ],
        INK,
    )
    box(ax, 3.35, 0.85, 3.05, 1.35, "Caches on disk", ["v1 and v2 each hold 6 TIC ids"], INK)

    box(ax, 6.7, 4.35, 3.1, 1.25, "POST /predict", ["Runs EvoMoE", "Stellar input is zeros"], GREEN)
    box(ax, 6.7, 2.7, 3.1, 1.4, "POST /predict-demo", ["Working tree only", "Authored scores", "Network is not called"], RED)
    box(ax, 6.7, 1.15, 3.1, 1.25, "AstroLens", ["Next.js client, four tabs"], BLUE)

    arrow(ax, 3.05, 4.95, 3.35, 4.4)
    arrow(ax, 3.05, 3.45, 3.35, 3.7)
    arrow(ax, 3.05, 1.95, 3.35, 3.1)
    arrow(ax, 6.4, 4.2, 6.7, 4.9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_system.png", dpi=200)
    fig.savefig(OUT / "fig_system.pdf")
    plt.close(fig)


def evomoe():
    fig, ax = plt.subplots(figsize=(7.4, 4.35))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.4)
    ax.axis("off")
    fig.patch.set_facecolor("#fbfaf6")
    ax.set_facecolor("#fbfaf6")

    box(ax, 0.15, 4.7, 2.7, 1.15, "Flux x, length 2000", ["one vector per target"], BLUE)
    box(ax, 0.15, 1.55, 2.7, 1.15, "Stellar u, length 12", ["not z-scored in the loader"], RED)

    box(ax, 3.2, 4.85, 3.15, 1.2, "CNN expert", ["kernels 5, 11, and 21", "then Linear 8000 to 128"], BLUE)
    box(ax, 3.2, 3.4, 3.15, 1.25, "Transformer expert", ["100 patches of length 20", "2 layers, 4 heads, width 32"], BLUE)
    box(ax, 3.2, 1.5, 3.15, 1.25, "Physics MLP", ["12 to 64 to 128", "dropout 0.2 in training"], RED)

    box(ax, 6.7, 4.15, 3.1, 1.45, "Three scalar confidences", ["g = softmax(c)", "inactive experts get -inf"], INK)
    box(ax, 6.7, 2.55, 3.1, 1.3, "Weighted sum", ["h is 128-dimensional", "not a mix of logits"], INK)
    box(ax, 6.7, 0.95, 3.1, 1.3, "Shared classifier", ["128 to 64 to 2 logits", "gate weights returned too"], GREEN)

    arrow(ax, 2.85, 5.3, 3.2, 5.4)
    arrow(ax, 2.85, 5.2, 3.2, 4.0)
    arrow(ax, 2.85, 2.1, 3.2, 2.1)
    arrow(ax, 6.35, 5.4, 6.7, 4.9)
    arrow(ax, 6.35, 4.0, 6.7, 4.7)
    arrow(ax, 6.35, 2.1, 6.7, 4.5)
    arrow(ax, 8.25, 4.15, 8.25, 3.85)
    arrow(ax, 8.25, 2.55, 8.25, 2.25)

    ax.text(
        0.15, 0.45,
        "1,485,605 parameters. The CNN block holds 1,040,321. No evolutionary search: Adam and cross-entropy.",
        fontsize=7.6, color=INK,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_evomoe.png", dpi=200)
    fig.savefig(OUT / "fig_evomoe.pdf")
    plt.close(fig)


def pipeline():
    fig, ax = plt.subplots(figsize=(7.35, 2.15))
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 2.4)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    stages = [
        (0.1, "Photometry", ["PDCSAP flux", "TIC columns"]),
        (2.5, "Reduction", ["median, SG 151", "sigma-clip 5"]),
        (4.9, "BLS fold", ["0.5 to 20 d", "2,000 samples"]),
        (7.3, "EvoMoE", ["three experts", "softmax gate"]),
        (9.7, "Decision", ["two logits", "gate weights"]),
    ]
    for x, title, lines in stages:
        box(ax, x, 0.35, 2.15, 1.7, title, lines, BLUE)
    for x in (2.25, 4.65, 7.05, 9.45):
        arrow(ax, x, 1.2, x + 0.25, 1.2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_pipeline.png", dpi=220)
    fig.savefig(OUT / "fig_pipeline.pdf")
    plt.close(fig)


def evomoe_pub():
    fig, ax = plt.subplots(figsize=(7.4, 4.15))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.15)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    box(ax, 0.15, 4.55, 2.7, 1.15, "Flux, length 2000", ["one series per target"], BLUE)
    box(ax, 0.15, 1.45, 2.7, 1.15, "Stellar vector, length 12", ["TIC columns, native scale"], BLUE)

    box(ax, 3.2, 4.7, 3.15, 1.15, "CNN expert", ["kernels 5, 11, and 21", "Linear 8000 to 128"], BLUE)
    box(ax, 3.2, 3.25, 3.15, 1.2, "Transformer expert", ["100 patches of length 20", "2 layers, 4 heads, width 32"], BLUE)
    box(ax, 3.2, 1.4, 3.15, 1.2, "Physics MLP", ["12 to 64 to 128", "dropout 0.2"], BLUE)

    box(ax, 6.7, 4.05, 3.1, 1.4, "Scalar confidences", ["g = softmax(c)", "one weight per expert"], INK)
    box(ax, 6.7, 2.5, 3.1, 1.25, "Weighted sum", ["128-dimensional", "sum of embeddings"], INK)
    box(ax, 6.7, 0.9, 3.1, 1.25, "Shared classifier", ["128 to 64 to 2 logits", "gate returned with z"], GREEN)

    arrow(ax, 2.85, 5.15, 3.2, 5.25)
    arrow(ax, 2.85, 5.05, 3.2, 3.85)
    arrow(ax, 2.85, 2.0, 3.2, 2.0)
    arrow(ax, 6.35, 5.25, 6.7, 4.8)
    arrow(ax, 6.35, 3.85, 6.7, 4.6)
    arrow(ax, 6.35, 2.0, 6.7, 4.4)
    arrow(ax, 8.25, 4.05, 8.25, 3.75)
    arrow(ax, 8.25, 2.5, 8.25, 2.15)

    ax.text(
        0.15, 0.35,
        "1,485,605 parameters. Training uses Adam and cross-entropy.",
        fontsize=7.6, color=INK,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_evomoe_pub.png", dpi=220)
    fig.savefig(OUT / "fig_evomoe_pub.pdf")
    plt.close(fig)


if __name__ == "__main__":
    pipeline()
    evomoe_pub()
    print("publication diagrams written")
