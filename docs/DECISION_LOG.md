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
* **Reason:** In astrophysics, the cost of a false positive follow-up (e.g., using JWST time) is immense. A monolithic network cannot easily explain *which* physical characteristics led to the prediction. By routing the decision through three explicit experts (Morphology, Periodicity, Stellar Physics), researchers can visually inspect the gating weights to see if the model trusted the stellar parameters or the raw transit shape.
