# Changelog

All notable changes to the AstroVerse project are documented here.

---

## V4 — July 2026

**Research-grade transformation.**

- Added YAML-based experiment configs
- Added unit and integration test suites
- Added CI/CD pipeline (linting, tests, architecture validation)
- Added experiment tracking infrastructure
- Structured monorepo with clear separation: `research/`, `services/`, `apps/`
- Added comprehensive documentation suite (this file and siblings)

---

## V3 — July 2026

**Honesty and correctness pass.**

- Removed all fabricated metrics and benchmarks from codebase and docs
- Fixed CSS import crashes in AstroLens dashboard
- Fixed broken module imports across the monorepo
- Added sinusoidal positional encoding to the Temporal Transformer expert
- Replaced all unverified claims with "pending evaluation" markers
- Rewrote documentation to reflect actual project state

---

## V2 — June 2026

**Repository cleanup.**

- Reorganized into monorepo structure (`apps/`, `services/`, `research/`, `docs/`)
- Separated frontend (AstroLens) from API (EvoNex) from research code
- Added FastAPI inference API with Swagger docs
- Standardized Python packaging and dependency management

---

## V1 — June 2026

**Initial prototype.**

- EvoMoE architecture: Multi-Scale CNN + Temporal Transformer + Physics MLP
- Confidence-guided gating network
- Data pipeline: lightkurve → BLS → phase-fold → HDF5
- Basic Next.js dashboard with lightcurve visualization
- Mock data for development testing
