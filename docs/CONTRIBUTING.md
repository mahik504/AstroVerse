# Contributing to AstroVerse

Thank you for your interest in contributing. This document covers the standards and process for all contributions.

---

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Create a feature branch from `main`
4. Make your changes
5. Submit a pull request

---

## Code Style

### Python (`research/`, `services/`)

- **Linter/formatter:** [ruff](https://docs.astral.sh/ruff/)
- Line length: 88 characters
- Type hints required for all public functions
- Docstrings: Google style

```bash
# Check
ruff check .

# Format
ruff format .
```

### TypeScript (`apps/`)

- **Linter:** ESLint with the project config
- **Formatter:** Prettier (via ESLint integration)
- Strict TypeScript — no `any` unless justified with a comment

```bash
cd apps/astrolens-web
npm run lint
```

### General

- No commented-out code in PRs
- No `print()` statements — use `logging` (Python) or structured logs (TypeScript)
- Keep functions under 50 lines where practical

---

## Branch Naming

Use the following prefixes:

| Prefix | Purpose | Example |
|---|---|---|
| `feat/` | New feature | `feat/multi-sector-support` |
| `fix/` | Bug fix | `fix/bls-nan-handling` |
| `docs/` | Documentation only | `docs/api-examples` |
| `refactor/` | Code restructuring | `refactor/gating-network` |
| `test/` | Adding or fixing tests | `test/cnn-expert-unit` |
| `experiment/` | Research experiments | `experiment/ablation-no-physics` |

---

## Pull Request Process

1. **One concern per PR.** Don't mix a bug fix with a new feature.
2. **Write a clear description.** Explain *what* changed and *why*.
3. **Link related issues** using `Closes #N` or `Relates to #N`.
4. **All CI checks must pass** before merge.
5. **Request review** from at least one maintainer.
6. **Squash merge** into `main` — keep the commit history clean.

### PR Title Format

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(model): add dropout to physics expert
fix(api): handle missing TIC stellar params
docs: update reproducibility guide
test(pipeline): add BLS edge case coverage
```

---

## Testing Requirements

### Before Submitting a PR

- [ ] `ruff check .` passes with no errors
- [ ] `ruff format --check .` shows no formatting changes
- [ ] Existing tests pass: `pytest research/evonex/tests/`
- [ ] New code has tests (unit tests at minimum)
- [ ] Architecture validation still passes if model code was changed

### What to Test

| Area | Test type | Tool |
|---|---|---|
| Model components | Unit tests | pytest |
| Data pipeline | Integration tests | pytest |
| API endpoints | Endpoint tests | pytest + httpx |
| Frontend | Component tests | Jest / Vitest |

### Running Tests

```bash
# Python tests
pytest research/evonex/tests/ -v

# API tests
pytest services/evonex-api/tests/ -v

# Frontend tests
cd apps/astrolens-web
npm test
```

---

## Research Contributions

If your contribution involves model changes or new experiments:

1. **Document your hypothesis** in the PR description
2. **Record experiment config** — commit the YAML config used
3. **Do not fabricate metrics** — if the experiment hasn't been run, mark results as "pending evaluation"
4. **Track experiments** using the project's experiment tracking setup

---

## Questions?

Open a [Discussion](https://github.com/<org>/AstroVerse/discussions) or file an issue.
