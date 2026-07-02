# AstroVerse Project Overview

Welcome to AstroVerse! This document serves as the primary onboarding guide for new contributors and researchers.

## The Idea
Exoplanet transit detection is inherently plagued by high false-positive rates (e.g., eclipsing binaries, stellar variability, instrumental noise). While deep learning has proven highly effective at identifying transits, monolithic architectures operate as "black boxes" that astrophysicists struggle to trust or debug. 

AstroVerse bridges this gap by introducing **EvoMoE**, a Confidence-Guided Mixture-of-Experts architecture that explicitly fuses morphological, temporal, and physical signals, explaining *why* it made a prediction based on the weighting of these experts.

## The Architecture
The AstroVerse platform consists of three pillars:
1. **The Research Pipeline (`research/evonex/`)**: A PyTorch-based training and evaluation framework for building and validating EvoMoE against classical and neural baselines.
2. **The Inference Engine (`services/evonex-api/`)**: A FastAPI backend that hosts the trained weights and provides low-latency REST endpoints for live astronomical data processing.
3. **The AstroLens Dashboard (`apps/astrolens-web/`)**: A Next.js 16 frontend that visualizes the light curves and expert routing contributions for human-in-the-loop validation.

## Key Design Decisions
- **HDF5 over SQLite/Parquet**: We store the preprocessed phase-folded light curves in HDF5 (`.h5`) format. It natively supports high-dimensional tensors and allows partial loading directly into PyTorch DataLoaders.
- **FastAPI over Flask**: Selected for its asynchronous capabilities (crucial for long MAST API queries) and native Pydantic validation.
- **Explainability over Raw Accuracy**: We explicitly engineered the model to trade minor performance gains for the interpretability provided by the Confidence-Gating mechanism.

## File-by-File Breakdown (Core Research)
- `src/build_dataset.py`: Fetches real TOI labels and MAST lightcurves, applying BLS phase-folding.
- `src/model_evonex.py`: Contains the raw PyTorch definition of EvoMoE and its experts.
- `src/run_baselines.py`: Evaluates classical, CNN, ResNet, and XGBoost baselines over multiple random seeds for statistical rigor.
- `src/run_ablations.py`: Systematically disables individual experts to quantify their contribution to the F1 and AUPRC scores.
- `analysis/reliability.py`: Computes Expected Calibration Error (ECE) and plots reliability diagrams.

## How to Extend
To add a new expert to EvoMoE:
1. Define the expert class in `model_evonex.py` (ensure it returns `embedding, confidence`).
2. Add the expert to `EvoMoE_Model.__init__`.
3. Update the `forward` method to include the new confidence in the `gating_weights` softmax calculation.
4. Update `run_ablations.py` to include the new expert in the ablation dictionary.
