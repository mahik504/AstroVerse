# Research Policy (The Constitution - Part I)

This document defines the behavioral and ethical policies governing the AstroVerse project. It acts alongside the `EXPERIMENT_PROTOCOL.md` to ensure maximum scientific integrity.

## 1. Publication Rules
- No empirical claims (accuracy, FLOPs, latency, etc.) may be published in the `README.md`, `Model_Card.md`, or anywhere else until they are backed by an executed, logged experiment.
- Results must be published honestly, regardless of whether EvoMoE outperforms classical or deep learning baselines. Negative results are considered valuable scientific contributions.

## 2. Experiment Registration
- Every experiment must be logged in `EXPERIMENT_REGISTRY.md` with an immutable, sequential ID (e.g., `EXP-001`, `EXP-002`).
- An experiment must define its Purpose, Hypothesis, Expected Failure Modes, and Reviewer Notes *before* execution.
- Overwriting or silently modifying an existing experiment ID is strictly prohibited.

## 3. Benchmark Gate
- The automated `Benchmark Gate` script must be passed before any baseline or EvoMoE model is evaluated.
- The gate enforces minimum dataset sizes, duplicate removal verification, and checksum validation.

## 4. Integrity Rules
- The experimental protocol (`EXPERIMENT_PROTOCOL.md`) must be **frozen** before the first benchmark is run.
- There will be no changing of preprocessing, metrics, or dataset splits after observing results.
- There will be no silent dropping of difficult samples to inflate metrics.
- If a protocol change is deemed scientifically necessary, it must be documented as a new protocol version (`v2`), and all affected experiments must be rerun from scratch.

## 5. Versioning
- Versioning follows scientific milestones, not software features:
  - `v1.0.0`: Engineering Freeze
  - `v1.1.0`: Dataset Generation
  - `v1.2.0`: Baseline Evaluation
  - `v1.3.0`: EvoMoE Evaluation
  - `v1.4.0`: Ablations & Calibration
  - `v1.5.0`: Paper Submission
  - `v2.0.0`: Zenodo DOI & Public Release

## 6. Authorship
- Contributions must strictly follow the CRediT (Contributor Roles Taxonomy).
