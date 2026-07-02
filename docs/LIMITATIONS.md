# Known Limitations

> Honest accounting of what this system cannot do yet.

---

## Dataset

**Current training set: 4 targets.**

The data pipeline has been validated on 4 TESS targets. This is sufficient for architecture validation but far too small for meaningful training or evaluation. Large-scale training on the TOI catalog is planned but not yet complete.

No trained model weights are available. All prediction outputs are untested.

---

## Known Failure Modes

### Grazing Eclipsing Binaries

Eclipsing binary stars with grazing geometry produce V-shaped transits that closely mimic planetary transits. The model has no dedicated mechanism to distinguish these. Centroid analysis or secondary eclipse detection would be needed — neither is currently implemented.

### Long-Period Planets (>15 days)

BLS phase-folding degrades for orbital periods exceeding ~15 days in single-sector TESS data (~27 days of coverage). With only 1–2 transit events visible, the period estimate is unreliable and the phase-folded lightcurve is noisy.

### Multi-Planet Transit Overlap

When a star hosts multiple transiting planets, their transit signals overlap in the lightcurve. The current pipeline runs BLS once and folds on a single period. Secondary planet transits become noise in the phase-folded data.

### Single-Sector Assumption

The pipeline currently processes one TESS sector at a time. Multi-sector stacking — which would improve SNR and period precision — is not yet implemented.

---

## Missing Capabilities

### No Uncertainty Quantification

The model outputs a single transit probability with no confidence interval. There is no MC dropout, ensemble, or calibration mechanism. A high output probability does not necessarily indicate calibrated confidence.

### No Trained Weights

Architecture validation is complete but the model has not been trained on a labeled dataset. The `/predict` endpoint will return `503` until weights are available.

### No Random Seed Configuration

Training reproducibility is not guaranteed — random seeds for PyTorch, NumPy, and CUDA are not yet set in the training pipeline.

### No Automated Evaluation Pipeline

There is no automated script to compute precision, recall, AUC, or other metrics on a test set. This will be built alongside the training pipeline.

---

## Scope Boundaries

This system is designed for **transit detection** (is there a periodic dip?), not for:

- Transit timing variations (TTVs)
- Atmospheric characterization
- Radial velocity confirmation
- False positive probability estimation (e.g., VESPA)

These are complementary analyses that would use EvoMoE predictions as one input among many.
