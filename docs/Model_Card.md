# Model Card: EvoMoE

## Model Details
- **Name:** EvoMoE (Evolutionary Mixture of Experts)
- **Version:** 1.0.0-research
- **Type:** Neural Classification Model (Binary: False Positive / Confirmed Planet)
- **Parameters:** 1,485,605 (trainable)
- **License:** MIT

## Architecture
EvoMoE utilizes a **Confidence-Guided Mixture-of-Experts** mechanism designed for astrophysical explainability.
1. **Multi-Scale 1D CNN:** Extracts local morphological features of the transit (U-shape vs V-shape).
2. **Time-Series Transformer:** Extracts global periodic features using sinusoidal positional encoding over the phase-folded light curve.
3. **Physics MLP:** Validates the physical constraints of the transit event using 12 TIC stellar parameters (Teff, Radius, Mass, etc.).
- **Gate:** Each expert emits one scalar confidence. A softmax over those three scalars weights the embeddings. A shared head maps the weighted sum to two logits. The gate is returned with the logits.

## Intended Use
- **Primary Use:** Automated vetting of TESS threshold-crossing events to prioritize follow-up observations.
- **Secondary Use:** Identifying instrumental false positives and eclipsing binaries.
- **Out of Scope:** This model is not intended to replace human-in-the-loop validation for defining official Kepler/TESS catalogs.

## Training Data
The catalog schedule is `research/evonex/configs/paper.yaml`: 100 epochs, batch 256, learning rate 10^-4, cosine decay, seed 42, periods from 0.5 to 20 days. A scored run on a held-out TESS Objects of Interest sample has not been published from this repository.

## Evaluation Results
The model is evaluated based on precision-critical metrics due to the high class imbalance of exoplanet surveys:
- **AUPRC:** Pending Large-Scale Run
- **F1 Score:** Pending Large-Scale Run
- **ECE (Expected Calibration Error):** Pending Large-Scale Run

## Computational Requirements
- **Inference Time:** Median 7.64 ms per target, batch size 1, CPU (PyTorch 2.12.1, 100 calls after 20 warm-up calls)
- **FLOPs:** Pending Evaluation
- **Hardware:** Can be trained natively on a standard GPU (NVIDIA T4 or higher) or CPU due to its lightweight parameter footprint.
