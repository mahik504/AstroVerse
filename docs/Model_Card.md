# Model Card: EvoMoE

## Model Details
- **Name:** EvoMoE (Evolutionary Mixture of Experts)
- **Version:** 1.0.0-research
- **Type:** Neural Classification Model (Binary: False Positive / Confirmed Planet)
- **Parameters:** 1.48M (Trainable)
- **License:** MIT

## Architecture
EvoMoE utilizes a **Confidence-Guided Mixture-of-Experts** mechanism designed for astrophysical explainability.
1. **Multi-Scale 1D CNN:** Extracts local morphological features of the transit (U-shape vs V-shape).
2. **Time-Series Transformer:** Extracts global periodic features using sinusoidal positional encoding over the phase-folded light curve.
3. **Physics MLP:** Validates the physical constraints of the transit event using 12 TIC stellar parameters (Teff, Radius, Mass, etc.).
- **Gating Network:** A dynamic Softmax layer that calculates the optimal weighting of the three experts based on the input signal quality.

## Intended Use
- **Primary Use:** Automated vetting of TESS threshold-crossing events to prioritize follow-up observations.
- **Secondary Use:** Identifying instrumental false positives and eclipsing binaries.
- **Out of Scope:** This model is not intended to replace human-in-the-loop validation for defining official Kepler/TESS catalogs.

## Training Data
The model is trained on the AstroVerse Curated TOI dataset (`v2-curated-500` for initial benchmarking). The data is split temporally or by random seed into 80/10/10 Train/Val/Test partitions.

## Evaluation Results
The model is evaluated based on precision-critical metrics due to the high class imbalance of exoplanet surveys:
- **AUPRC:** Pending Large-Scale Run
- **F1 Score:** Pending Large-Scale Run
- **ECE (Expected Calibration Error):** Pending Large-Scale Run

## Computational Requirements
- **Inference Time:** ~2.4 ms per target (CPU)
- **FLOPs:** 11.2M
- **Hardware:** Can be trained natively on a standard GPU (NVIDIA T4 or higher) or CPU due to its lightweight parameter footprint.
