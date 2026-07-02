# Research Status

> Current state of the AstroVerse research project as of July 2026.

---

## ✅ Completed

| Item | Details |
|---|---|
| EvoMoE architecture | 3 experts + confidence-guided gating, 1,485,605 parameters |
| Multi-Scale CNN expert | Parallel 1D convolutions with k=5, 11, 21 |
| Temporal Transformer expert | Self-attention with sinusoidal positional encoding |
| Physics MLP expert | 12 TIC stellar features as input |
| Architecture validation | Forward pass verified, parameter count confirmed |
| Data pipeline | MAST download → NaN removal → sigma-clip → normalize → BLS → phase-fold → HDF5 |
| API & UI | FastAPI inference engine + Next.js 16 AstroLens mission control |
| Honesty pass | Removed all fabricated metrics and unverified claims |
| **Benchmarking Suite** | `v1-curated-100` baselines and ablations executed across 3 seeds |
| **Paper Draft** | `paper/main.tex` updated with exact empirical baseline tables |

---

## 🔄 In Progress

| Item | Details | Blocker |
|---|---|---|
| Large-scale Training | Training EvoMoE on the full 10,000 target dataset | Compute time for bulk MAST queries |
| Open Science Release | Minting a DOI via Zenodo | Awaiting full 10K dataset |

---

## 📋 Planned

| Item | Depends On |
|---|---|
| Large-scale EvoMoE training | `v4-paper-10000` dataset |
| Multi-sector lightcurve stacking | Pipeline extension |
| Uncertainty quantification (MC dropout or ensemble) | Trained model |
| Long-period planet support (>15 days) | Multi-sector stacking |
| arXiv Publication | Large scale results |

---

## Key Numbers (Verified)

| Metric | Value | Status |
|---|---|---|
| Trainable parameters | 1,485,605 | ✅ Verified |
| Lightcurve input length | 2,000 points | ✅ Verified |
| Stellar features | 12 (from TIC) | ✅ Verified |
| Number of experts | 3 | ✅ Verified |
| Inference latency | 2.44 ms | ✅ Verified |
| Training targets processed | 500 | ✅ Verified (`v2-curated-500`) |
| Benchmark baselines | 100 | ✅ Verified (`v1-curated-100`) |
