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
**Observed Result**: *[To be filled after execution]*
**Statistical Analysis**: *[To be filled after execution]*
**Conclusion**: *[To be filled after execution]*
**Next Action**: Evaluate EvoMoE on the same dataset.

---
