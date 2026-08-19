# Experiment Protocol (The Constitution - Part II)

This document dictates *how* experiments are run in AstroVerse. It must be frozen before any scientific evaluation takes place.

## 1. Dataset Preprocessing (v1)
- **Quality Masking:** `quality_bitmask = 17087` (the standard TESS-SPOC mask). This rejects severe anomalies (e.g., momentum dumps, stray light) while preserving cadences with minor flags that do not definitively destroy transit signals. We do NOT use 'hard' (which over-rejects valid transits) or 'default' (which under-rejects momentum dumps).
- **Flattening:** Median filtering with window size 101 to remove long-term stellar variability.
- **Normalization:** Min-Max scaling to `[0, 1]` or Z-score normalization computed **strictly on the training split**. (Leakage check #1).
- **Phase Folding:** BLS period detection applied. Note: If the BLS period is incorrect, the fold is incorrect. This is an accepted failure mode and must be documented as "Preprocessing Failure" in the taxonomy.
- **Interpolation:** Fixed 2000-cadence linear interpolation across the phase fold.

## 2. Dataset Splits & Cross Validation
- **Method:** 5-fold Nested Cross-Validation.
- **Outer CV:** 5 folds for model evaluation (80/20 train/test split per fold).
- **Inner CV:** 5 folds *within the training data* for hyperparameter optimization (HPO).
- **Seed Aggregation:** The outer loop must be run across a minimum of **3 random seeds**.
- **Data Leakage Check #2:** No single TIC ID may appear in both the training and test set within any fold.

## 3. Evaluation Metrics
Every model evaluation must produce:
1.  ROC Curve & AUROC
2.  Precision-Recall Curve (PR Curve) & AUPRC
3.  Expected Calibration Error (ECE)
4.  Brier Score
5.  Confusion Matrix
6.  Precision, Recall, Specificity, Sensitivity
7.  Matthews Correlation Coefficient (MCC)

## 4. Statistical Tests
Model superiority claims must be backed by paired statistical tests across the CV folds/seeds:
- **Paired t-test** (if normality assumptions hold).
- **Wilcoxon signed-rank test** (for non-parametric comparison).
- **Bootstrap Confidence Intervals** (95% CI) for primary metrics (AUPRC, F1).

## 5. Failure Taxonomy
When a False Positive is detected by the model, it must be categorized into one of the following to aid failure analysis:
- `Instrument Noise`
- `Variable Star`
- `Eclipsing Binary`
- `Low SNR`
- `Missing Transit`
- `Preprocessing Failure`
- `Unknown`

## 6. Hardware Logging
The following must be recorded in the `dataset_manifest.json` or `model_card.json` for every run:
- CPU model
- GPU model & VRAM
- CUDA Version
- System RAM
- Operating System
