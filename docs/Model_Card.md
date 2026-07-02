# Model Card: EvoMoE (EvoNex Mixture-of-Experts)

## Model Details
- **Name:** EvoMoE v0.3
- **Architecture:** Confidence-Guided Mixture-of-Experts
- **Framework:** PyTorch 2.0+
- **Parameters:** 1,485,605 trainable
- **Input Tensors:**
  1. `Lightcurve Tensor`: Shape `(Batch, 2000)` — Phase-folded, normalized flux time-series.
  2. `Stellar Features Tensor`: Shape `(Batch, 12)` — Normalized astrophysical parameters from TIC.
- **Output:** Binary classification (1 = Exoplanet Candidate, 0 = False Positive) and Expert Routing Weights.

## Intended Use
EvoMoE is designed as an automated triaging tool for NASA TESS photometry. It filters astrophysical false positives (eclipsing binaries, instrumental artifacts) and highlights exoplanet candidates for human follow-up.

## Architecture

Three expert subnetworks with confidence-guided fusion:

1. **Multi-Scale CNN Expert:** Parallel 1D convolutions with kernel sizes k={5, 11, 21} capture both sharp V-shaped eclipsing binary dips and shallow U-shaped planetary transits.
2. **Temporal Transformer Expert:** Patch-based transformer encoder with sinusoidal positional encoding captures long-range periodicities and out-of-transit stellar variability.
3. **Physics MLP Expert:** Processes 12 normalized TIC stellar parameters (Teff, logg, radius, mass, etc.) to validate transit depth against physical constraints.

**Gating:** Each expert outputs an embedding and a scalar confidence logit. Softmax-normalized confidence scores dynamically weight the expert embeddings before classification.

## Performance

> **Status: Architecture validated. Full-scale evaluation pending.**
>
> The model has been validated for correct forward-pass behavior, dimensional consistency, and gating distribution. Training on a large labeled dataset (NASA TOI catalog) is in progress. No benchmark metrics are available yet.

## Known Limitations & Failure Modes
1. **Grazing Eclipsing Binaries:** Eccentric grazing binaries can mimic planetary U-shaped transits. The Physics MLP mitigates this when TIC metadata is accurate, but may fail with high-error stellar parameters.
2. **Ultra-Long Period Planets:** Planets with orbital periods >15 days produce only one transit per TESS sector (27 days). The Transformer expert struggles with single-event periodicity.
3. **Multi-Planet Systems:** Overlapping transit signals (Transit Timing Variations) can confuse period extraction, reducing BLS accuracy.
4. **TIC Data Quality:** Missing or erroneous stellar parameters (common for faint stars) degrade Physics MLP performance.

## Ethical Considerations
Automated detection pipelines influence allocation of major observatory resources (e.g., JWST follow-up). Over-trusting AI predictions without explainability risks wasting telescope time. EvoMoE mitigates this by exposing internal routing weights, requiring the model to justify its decision process.
