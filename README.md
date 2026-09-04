# AstroVerse

[![CI](https://github.com/mahik504/AstroVerse/actions/workflows/ci.yml/badge.svg)](https://github.com/mahik504/AstroVerse/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Open-source exoplanet transit detection pipeline: explainable Mixture-of-Experts (EvoMoE), FastAPI inference, AstroLens Next.js dashboard. Built for BAH 2026.

Benchmark numbers ship only after the dataset gate passes — see `docs/DECISION_LOG.md`. There is no live hosted URL and no published accuracy claim.

## Status (v1.0.0-beta)

| Track | State |
| --- | --- |
| Engineering | Complete (v1.0.0-beta) |
| Pipeline validation | Complete |
| Dataset `v2-curated-500` | In progress (6 of 500 targets on the last honest run) |
| Baseline benchmarking | Pending |
| Large-scale evaluation | Pending (`v4-paper-10000`) |

Implemented today: AstroLens UI, FastAPI inference, EvoMoE PyTorch model, MAST ingestion, Box Least Squares phase-folding. See [ROADMAP.md](docs/ROADMAP.md).

## Architecture

![AstroVerse Architecture](docs/architecture.svg)

## Quick start

```bash
git clone https://github.com/mahik504/AstroVerse.git
cd AstroVerse
make test        # research + API pytest
make dashboard   # FastAPI :8000 and Next.js :3000
```

Demo light curves are in `demo/`. There is no `make benchmark` target yet.

## Repository

```text
AstroVerse/
├── apps/astrolens-web/     # Next.js UI (Mission Control)
├── services/evonex-api/    # FastAPI inference
├── research/evonex/        # PyTorch model, training, evaluation
├── paper/                  # LaTeX manuscript
└── docs/                   # methodology and reproducibility
```

Onboarding: [Project Overview](docs/PROJECT_OVERVIEW.md).

## Documentation

- [Project Overview](docs/PROJECT_OVERVIEW.md)
- [Architecture](docs/ARCHITECTURE.md)
- [API Reference](docs/API.md)
- [Reproducibility](docs/REPRODUCIBILITY.md)
- [Model Card](docs/MODEL_CARD.md) / [Dataset Card](docs/DATASET_CARD.md)

## Contributing

[CONTRIBUTING.md](docs/CONTRIBUTING.md) — ruff, ESLint, PR templates, tests.

## Citation

```bibtex
@software{AstroVerse2026,
  author = {Mahi K},
  title = {AstroVerse: Adaptive Mixture-of-Experts for Exoplanet Detection},
  year = {2026},
  url = {https://github.com/mahik504/AstroVerse}
}
```
