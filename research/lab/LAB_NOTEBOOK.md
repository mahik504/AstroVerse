# AstroVerse Lab Notebook

> This document tracks the formal execution of scientific experiments on the EvoMoE architecture. Every experiment must be pre-registered here *before* execution.

---

### Experiment #001 (Template)

**Date**: YYYY-MM-DD
**Research Question**: What is the baseline performance of classical and monolithic deep learning models on the 500-target dataset?
**Hypothesis**: The 1D ResNet will achieve the highest AUPRC among baselines, but with poor calibration (high ECE).
**Dataset Version**: `v2-curated-500`
**Model Version**: Classical BLS, 1D CNN, 1D ResNet, TS Transformer
**Configuration**: `configs/paper.yaml`
**Random Seed**: 42, 43, 44
**Hardware**: CPU
**Expected Result**: ResNet AUPRC > 0.85, ResNet ECE > 0.15.
**Observed Result**: Dataset generation failed to yield a statistically significant sample size. Of 540 attempted targets, only 6 were successfully downloaded and saved to the HDF5 cache (5 positives, 1 negative).
**Statistical Analysis**: N/A (Insufficient N).
**Conclusion**: The MAST API ingestion pipeline is fragile. It is likely hitting rate limits, timeouts, or the targets in `toi_catalog.csv` do not have SPOC lightcurves available. Running ML baselines on $N=6$ is methodologically invalid.
**Next Action**: Debug `build_dataset.py` and `tic_fetcher.py`. We must implement robust retry logic, error logging, and asynchronous downloading before re-attempting `v2-curated-500`.

---
