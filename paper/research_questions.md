# Research Questions

The empirical execution of AstroVerse is strictly guided by the following core hypotheses. Every experiment, baseline, and ablation study must contribute evidence toward answering at least one of these questions.

### RQ1: Architecture Superiority
**Does confidence-guided multimodal fusion provide statistically significant improvements in exoplanet transit detection compared with monolithic architectures under identical datasets, preprocessing pipelines, and evaluation protocols?**
*Success Criteria:* EvoMoE demonstrates a statistically significant improvement in Area Under the Precision-Recall Curve (AUPRC) compared to monolithic baselines across multiple random seeds on the `v4-paper-10000` dataset.

### RQ2: Multimodal Fusion
**Does incorporating explicit stellar metadata (T_eff, Mass, Radius) improve classification performance under specific stellar populations?**
*Success Criteria:* The inclusion of the Physics MLP Expert results in a measurable decrease in false positive rates for specific edge cases (e.g., grazing eclipsing binaries or giant stars) compared to the CNN+Transformer ablation.

### RQ3: Expert Specialization
**Which expert contributes most under different signal conditions?**
*Success Criteria:* Analysis of the Confidence-Guided Routing weights demonstrates predictable specialization (e.g., the Transformer Expert dominates for long-period signals, while the CNN Expert dominates for high-SNR single transits).

### RQ4: Scientific Reliability
**Is EvoMoE better calibrated than conventional monolithic models?**
*Success Criteria:* EvoMoE yields a lower Expected Calibration Error (ECE), Negative Log Likelihood (NLL), and a better Brier Score than standard baselines, proving that its confidence scores are reliable enough for automated scientific vetting.

### RQ5: Computational Trade-off
**What is the computational trade-off between predictive performance and model complexity?**
*Success Criteria:* The performance gains achieved by EvoMoE justify the overhead in FLOPs, Parameter Count, GPU Memory, and Inference Time compared to the fastest monolithic baseline.
