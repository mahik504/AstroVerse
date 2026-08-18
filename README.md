# AstroVerse

[![CI](https://github.com/mahik504/AstroVerse/actions/workflows/ci.yml/badge.svg)](https://github.com/mahik504/AstroVerse/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Open-source exoplanet transit detection pipeline: explainable Mixture-of-Experts (EvoMoE) model, FastAPI inference, AstroLens Next.js dashboard. Built for BAH 2026. Benchmark numbers ship only after the dataset gate passes — see `docs/DECISION_LOG.md`.

## Current Research Status: v1.0.0-beta

- **Engineering Phase:** Complete (v1.0.0-beta)
- **Pipeline Validation:** Complete
- **Dataset Construction:** In Progress
- **Baseline Benchmarking:** Pending
- **Large-scale Evaluation:** Pending

> **Note on Scientific Integrity:** This repository intentionally avoids publishing unsupported empirical metrics. Results will be released only after experiments have been completed, verified across multiple seeds, and fully reproduced. See `docs/DECISION_LOG.md` for our research methodology.

---

### Architecture
![AstroVerse Architecture](docs/architecture.svg)

---

### AstroLens UI Demo
*[Placeholder: Animated GIF of AstroLens UI processing a NASA TESS Target]*

---

## ⚡ Quick Start

```bash
git clone https://github.com/mahik504/AstroVerse.git
cd AstroVerse

# 1. Pipeline Validation
make validate-pipeline

# 2. Dataset Generation (Local Cache)
make build-dataset VERSION=v2-curated-500

# 3. Scientific Benchmark (Requires Benchmark Gate to pass)
make benchmark

# 4. Launch the API and Next.js Dashboard
make dashboard
```
Open [http://localhost:3000](http://localhost:3000) to view the detection mission control.

---

## 🔬 Current Research Status (v1.0.0-research)

AstroVerse is currently in the **Research Execution Phase**. The software engineering feature-freeze is active while we execute empirical benchmarking.

* **Implemented:** Full AstroLens Next.js UI, FastAPI inference engine, EvoMoE PyTorch model, MAST automated ingestion, Box Least Squares phase-folding.
* **Currently Executing:** Evaluating EvoMoE against classical baselines on the `v2-curated-500` dataset.
* **Future Work:** Scaling to `v4-paper-10000` (10,000 targets) and arXiv publication.

See our [ROADMAP.md](docs/ROADMAP.md) for detailed progress.

---

## 📊 Empirical Benchmarks
*(AstroVerse is currently in the empirical benchmarking phase. Baseline evaluations against Classical BLS, 1D CNN, ResNet, and Transformers on the `v2-curated-500` dataset are actively running. Real metrics will be published here upon completion. EvoMoE full results will follow the `v4-paper-10000` distributed training run.)*

---

## 📂 Repository Structure

```text
AstroVerse/
├── apps/astrolens-web/     # Next.js 16 UI (Mission Control Dashboard)
├── services/evonex-api/    # FastAPI Inference Engine
├── research/evonex/        # PyTorch Model, Training, and Evaluation Pipeline
├── paper/                  # LaTeX Academic Manuscript
└── docs/                   # Scientific Documentation & Reproducibility Guides
```

For onboarding, please read the [Project Overview](docs/PROJECT_OVERVIEW.md).

---

## 📚 Documentation
For a deep dive into the scientific and engineering principles of AstroVerse:
- **[Project Overview](docs/PROJECT_OVERVIEW.md)** — Onboarding guide and file-by-file breakdown.
- **[Architecture](docs/ARCHITECTURE.md)** — EvoMoE math, expert routing, and data flow.
- **[API Reference](docs/API.md)** — FastAPI endpoints and usage.
- **[Reproducibility](docs/REPRODUCIBILITY.md)** — How to recreate our experiments.
- **[Model Card](docs/MODEL_CARD.md)** / **[Dataset Card](docs/DATASET_CARD.md)** — Details on EvoMoE and the dataset bias.

## 🤝 Contributing
We welcome contributions! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for our code style (ruff, ESLint), PR templates, and testing requirements.

## 📝 Citation
If you use AstroVerse or EvoMoE in your research, please cite:
```bibtex
@software{AstroVerse2026,
  author = {Mahi K},
  title = {AstroVerse: Adaptive Mixture-of-Experts for Exoplanet Detection},
  year = {2026},
  url = {https://github.com/mahik504/AstroVerse}
}
```
