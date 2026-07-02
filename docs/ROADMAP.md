# Roadmap

> AstroVerse project phases and milestones.

---

## Phase 1 — Foundation ✅ Complete

Architecture design, data infrastructure, and visualization.

- [x] EvoMoE architecture design (3 experts + confidence-guided gating)
- [x] Multi-Scale CNN expert (k=5, 11, 21)
- [x] Temporal Transformer with sinusoidal positional encoding
- [x] Physics MLP with 12 TIC stellar features
- [x] Architecture validation (forward pass, parameter count)
- [x] Data pipeline: MAST download → preprocessing → BLS → phase-fold → HDF5
- [x] FastAPI inference API (`evonex-api`)
- [x] AstroLens Next.js 16 dashboard
- [x] Monorepo structure and project organization
- [x] Honest documentation (no fabricated metrics)

---

## Phase 2 — Training & Validation 🔄 In Progress

Large-scale training on real TESS data with proper experiment tracking.

- [ ] Process full TOI catalog from NASA Exoplanet Archive
- [ ] Build labeled training set (confirmed planets + false positives)
- [ ] Configure random seeds for reproducibility
- [ ] Implement experiment tracking (MLflow or Weights & Biases)
- [ ] Train EvoMoE on labeled dataset
- [ ] Hyperparameter search (learning rate, expert dimensions, gating temperature)
- [ ] Validation on held-out TOI targets
- [ ] Cross-validation across TESS sectors
- [ ] Baseline comparisons (standalone CNN, standalone Transformer, BLS-only)

---

## Phase 3 — Publication & Extension 📋 Planned

Research output, ablation studies, and multi-sector generalization.

- [ ] Ablation studies (remove each expert, measure impact)
- [ ] Gating analysis (routing weight distributions across target types)
- [ ] Multi-sector lightcurve analysis (stacking observations)
- [ ] Uncertainty quantification (MC dropout or ensemble)
- [ ] Long-period planet support (>15 day orbits)
- [ ] Write and submit research paper
- [ ] Public model weights release
- [ ] Community benchmark contribution

---

## Timeline

| Phase | Target |
|---|---|
| Phase 1 | June–July 2026 (complete) |
| Phase 2 | July–September 2026 |
| Phase 3 | Q4 2026 |

> [!NOTE]
> Phase 2 timelines depend on TOI catalog processing and compute availability. Phase 3 dates are aspirational.
