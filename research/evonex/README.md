# EvoNex — Exoplanet Transit Detection Engine

An Adaptive Confidence-Guided Mixture-of-Experts (MoE) framework for robust exoplanet transit detection from NASA TESS light curves.

## Architecture

EvoNex uses three specialized expert networks fused via confidence-guided gating:

| Expert | Architecture | Input | Purpose |
|---|---|---|---|
| Local Expert | Multi-Scale CNN (k=5, 11, 21) | Phase-folded flux (2000 pts) | Transit dip morphology |
| Global Expert | Transformer + Positional Encoding | Phase-folded flux (2000 pts) | Orbital periodicity |
| Stellar Expert | Physics MLP | 12 TIC features | Stellar parameter validation |

**Gating:** Each expert outputs an embedding and a confidence logit. Softmax-normalized weights dynamically fuse the embeddings based on data quality.

## Pipeline

1. **Ingest** — Query NASA MAST API via `lightkurve` for SPOC light curves
2. **Preprocess** — Sigma-clipping, Savitzky-Golay detrending, median normalization
3. **Detect** — Box Least Squares (BLS) period extraction
4. **Phase-fold** — Fold at detected period, interpolate to fixed 2000-point sequence
5. **Cache** — Store processed tensors in HDF5 for fast re-access
6. **Classify** — EvoMoE binary classification (exoplanet vs. false positive)

## Quick Start

```bash
cd research/evonex
pip install -r requirements.txt

# Architecture validation
python src/model_evonex.py

# Evaluation (smoke test, no trained weights)
python src/evaluate_evomoe.py

# Training on sample targets
python src/train_evomoe.py --config configs/default.yaml
```

## Model Parameters

- **Total:** 1,485,605
- **Input:** (batch, 2000) lightcurve + (batch, 12) stellar features
- **Output:** (batch, 2) logits + (batch, 3) expert routing weights
- **Status:** Architecture validated. Large-scale training in progress.

## Notebooks

| Notebook | Purpose |
|---|---|
| `01_EDA_and_Preprocessing.ipynb` | Visualizes detrending and sigma-clipping |
| `02_TIC_Exploration.ipynb` | Stellar parameter analysis via MAST API |
| `03_BLS_Detection.ipynb` | BLS period extraction and phase-folding |
| `04_EvoMoE_Explainability.ipynb` | Expert routing weight visualization |

## References

- Shallue & Vanderburg (2018). AJ 155(2):94.
- Ansdell et al. (2018). ApJL 869(1):L7.
- Ricker et al. (2015). JATIS 1(1):014003.

*Built for BAH 2026. Data from the Barbara A. Mikulski Archive for Space Telescopes (MAST).*
