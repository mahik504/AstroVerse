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
| FastAPI inference API | 5 endpoints: health, targets, predict, lightcurve, report |
| AstroLens dashboard | Next.js 16 with lightcurve visualization |
| Monorepo structure | `apps/`, `services/`, `research/`, `docs/` |
| Documentation | Architecture, API, reproducibility, limitations, decisions |
| Honesty pass | Removed all fabricated metrics and unverified claims |

---

## 🔄 In Progress

| Item | Details | Blocker |
|---|---|---|
| TOI catalog processing | Downloading and labeling full TESS Objects of Interest catalog | Compute time for bulk MAST queries |
| Experiment tracking | Setting up MLflow or W&B for training runs | None — implementation in progress |
| Training pipeline hardening | Random seeds, checkpointing, early stopping | Waiting on labeled dataset |
| CI/CD pipeline | Linting, testing, architecture validation on push | None — implementation in progress |

---

## 📋 Planned

| Item | Depends On |
|---|---|
| Large-scale EvoMoE training | Labeled TOI dataset |
| Hyperparameter search | Training pipeline |
| Baseline comparisons (CNN-only, Transformer-only, BLS-only) | Training pipeline |
| Ablation studies (remove each expert) | Trained model |
| Gating weight analysis | Trained model |
| Multi-sector lightcurve stacking | Pipeline extension |
| Uncertainty quantification (MC dropout or ensemble) | Trained model |
| Long-period planet support (>15 days) | Multi-sector stacking |
| Research paper | All of the above |
| Public model weights release | Trained + validated model |

---

## 🚫 Blocked

| Item | Blocking Issue |
|---|---|
| Model evaluation metrics | No trained weights — cannot compute precision, recall, AUC |
| End-to-end predictions | No trained weights — `/predict` returns 503 |
| Labeled training dataset | TOI catalog processing not yet complete |

---

## Key Numbers (Verified)

| Metric | Value | Status |
|---|---|---|
| Trainable parameters | 1,485,605 | ✅ Verified |
| Lightcurve input length | 2,000 points | ✅ Verified |
| Stellar features | 12 (from TIC) | ✅ Verified |
| Number of experts | 3 | ✅ Verified |
| Training targets processed | 4 | ✅ Verified |
| Precision / Recall / AUC | — | ⏳ Pending evaluation |
| Inference latency | — | ⏳ Pending evaluation |
