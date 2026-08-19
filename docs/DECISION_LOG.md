# AstroVerse Decision Log

> This document tracks major scientific and architectural decisions in the project. It explains the *why* behind the *what*.

---

## 2026-07-03: Data Storage Format (HDF5)

**Why HDF5 over Parquet or SQLite?**

* **Alternatives Considered:** 
  * Parquet: Excellent for tabular data, but clumsy for fixed-length tensor arrays.
  * SQLite: Good for relational metadata, but terrible for high-throughput raw float arrays (lightcurves).
  * TFRecord: Native to TensorFlow, but we are using PyTorch.

* **Decision:** HDF5 (`h5py`).
* **Reason:** HDF5 allows us to store the 2000-step phase-folded light curves as native binary tensors. Crucially, it supports chunking and partial reads, meaning the PyTorch `DataLoader` can read individual batches directly from disk without loading the entire 10,000-target dataset into RAM.

---

## 2026-07-03: Confidence-Guided Gating vs. Standard Attention

**Why build a custom Confidence-Guided Gating mechanism instead of using a standard Vision Transformer?**

* **Alternatives Considered:**
  * Standard ViT: Treats the time-series as patches.
  * Monolithic CNN (ResNet-1D): High accuracy, zero explainability.

* **Decision:** EvoMoE with Softmax Confidence-Gating.
## 2026-07-03: TESS Data Author Prioritization

**Why include TESS-SPOC and QLP instead of filtering purely for SPOC?**

* **Decision:** We prioritize data in the following order: `['SPOC', 'TESS-SPOC', 'QLP', 'TGLC', 'ELEANOR']`. 
* **Reason:** Initial diagnosis (EXP-DIAG-001) showed that strictly filtering for `SPOC` caused a 99% data loss due to coverage limitations on specific targets. By falling back to other community-provided pipelines (like QLP and TESS-SPOC), we maintain a statistically robust sample size. We acknowledge that this introduces minor distribution shifts (as the baseline flux detrending differs slightly between pipelines), but the subsequent Savitzky-Golay filtering and normalization steps mitigate this variance. The exact pipeline distribution for each dataset is logged in its respective `dataset_report.json`.

---

## 2026-07-03: Local Bulk Catalog Processing vs Live API

**Decision:**
Use bulk CTL / local metadata indices and cached dataset chunks instead of strictly live MAST API queries.

**Reasoning:**
Live querying the MAST API for tens of thousands of targets caused severe rate limiting, timeouts, and incomplete datasets. While suitable for exploration (N<500), it proved too fragile for reproducible, publication-scale dataset construction (N=10,000+).

**Alternatives considered:**
Live MAST queries exclusively with aggressive exponential backoff (rejected due to excessive overall duration and transient failure susceptibility).

**Expected impact:**
Significantly higher reproducibility. The dataset generation script can now cleanly track and drop duplicates locally, ensuring that an exact dataset snapshot can be recreated reliably without depending on external API uptime.
