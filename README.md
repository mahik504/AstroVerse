# AstroVerse 🔭

**AI-Powered Exoplanet Transit Detection using Mixture-of-Experts Architecture on NASA TESS Data**

AstroVerse is a modular research platform for automated exoplanet detection. It combines a novel neural network architecture (**EvoMoE**) with a full-stack scientific dashboard (**AstroLens**).

## Architecture Overview

```
AstroVerse/
├── apps/astrolens-web/     # Next.js 16 scientific dashboard
├── research/evonex/        # PyTorch EvoMoE model + training pipeline
├── services/evonex-api/    # FastAPI inference server
└── docs/                   # Model Card, Dataset Card
```

### EvoMoE (Evolutionary Mixture of Experts)

EvoMoE routes exoplanet classification decisions across three specialized expert networks:

| Expert | Architecture | Purpose |
|---|---|---|
| **Local Expert** | Multi-Scale CNN (k=5, 11, 21) | Transit dip morphology |
| **Global Expert** | Transformer + Positional Encoding | Orbital periodicity |
| **Stellar Expert** | Physics-Aware MLP | Stellar parameter validation |

A **Confidence-Guided Gating Network** dynamically weights each expert's contribution using softmax-normalized confidence scores, providing inherent explainability.

### AstroLens (Frontend)

A Next.js dashboard that accepts light curve CSV uploads, sends them to the EvoMoE API, and visualizes:
- Time-series flux analysis with transit zone highlighting
- Expert routing matrix (CNN/Transformer/Physics contribution %)
- Downloadable transit analysis reports

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+

### Backend
```bash
cd services/evonex-api
python -m venv venv
.\venv\Scripts\Activate.ps1      # Windows
pip install fastapi uvicorn torch pandas python-multipart
python -m uvicorn main:app --reload
```

### Frontend
```bash
cd apps/astrolens-web
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

### Research Pipeline
```bash
cd research/evonex
pip install -r requirements.txt
python src/model_evonex.py        # Architecture validation
python src/evaluate_evomoe.py     # Run evaluation
```

## Data Sources
- **TESS Light Curves:** NASA MAST Archive via `lightkurve`
- **Stellar Parameters:** TESS Input Catalog (TIC v8.1)
- **Target List:** STScI Exoplanet Candidate Target List (xCTL v08.01)

## Project Status
- ✅ EvoMoE architecture implemented and validated
- ✅ HDF5-cached data pipeline with BLS phase-folding
- ✅ Full-stack AstroLens dashboard
- ✅ FastAPI inference server
- 🔄 Large-scale training on NASA TOI catalog (in progress)

## Documentation
- [Model Card](docs/Model_Card.md) — Architecture details and limitations
- [Dataset Card](docs/Dataset_Card.md) — Data sources and processing
- [Research Paper](research/evonex/docs/Research_Paper.md) — Technical paper

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE).

## Acknowledgments
- NASA TESS mission and the MAST archive
- STScI for the TIC and CTL catalogs
- ISRO BAH 2026 for the problem statement
