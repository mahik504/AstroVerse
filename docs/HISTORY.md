# The Journey of AstroVerse

This document archives the evolution of AstroVerse. It is not a technical changelog, but a narrative of how a project shifted its core philosophy from "impressive software" to "reproducible science."

## Phase 1: The Hackathon Origin
AstroVerse began as an ambitious hackathon concept: *"Can we build a state-of-the-art exoplanet detector using a Mixture-of-Experts (EvoMoE) architecture?"* The early repository was flashy. It featured a Next.js dashboard, FastAPI endpoints, and documentation making claims about model accuracy, latency, and FLOPs. 

However, there was a fundamental flaw: those metrics were placeholders. The model had not yet been rigorously trained on a full dataset. The project looked like a finished product, but scientifically, it was a hollow shell.

## Phase 2: The Engineering Overhaul
To move beyond a prototype, the repository underwent a massive engineering overhaul. 
- The codebase was restructured into a professional monorepo.
- Dependencies, CI/CD pipelines, and environment variables were standardized.
- The documentation was expanded. 
The software was now robust, but it was still trying to solve an engineering problem rather than a scientific one.

## Phase 3: The Methodological Pivot
The critical turning point occurred during an external review. It became clear that claiming state-of-the-art performance without reproducible evidence was an anti-pattern in research. The philosophy of the repository changed overnight.

Instead of asking, *"How do we get better accuracy?"* the project started asking, *"How do we ensure another researcher can reproduce our results?"*

This led to the most difficult but important decision in the project's history: **The Great Purge**.
Every unsupported claim, fake metric, placeholder ROC curve, and hallucinated FLOP count was deleted. The `README.md` and `Model_Card.md` were rewritten to explicitly state: *"Results: Pending Evaluation."* It was painful to make the repository look "less impressive," but it was the only way to regain scientific credibility.

## Phase 4: Rigorous Scientific Software
With the slate wiped clean, AstroVerse was rebuilt around the scientific method.
- **The Constitution (`EXPERIMENT_PROTOCOL.md`)** was drafted to freeze the rules of engagement (Nested 5-fold CV, metrics, statistical tests) *before* any experiments were run.
- **The Tri-Source Pipeline** was constructed to decouple ground truth labels (NASA Exoplanet Archive), metadata (TIC), and observations (MAST), generating self-describing HDF5 tensors.
- **The Benchmark Gate** was introduced to physically prevent models from running unless the dataset passed a strict audit for label leakage and class balance.

## The Engineering Freeze
AstroVerse is now under a strict Engineering Freeze. The UI, the folder structure, and the architecture will not change unless reproducibility demands it. 

## The Unanswered Question
The remaining challenge is no longer about writing code. It is about producing evidence.

The guiding research question of AstroVerse is now:
> *"Under what conditions does a confidence-guided Mixture-of-Experts architecture help, and under what conditions does it not?"*

If EvoMoE outperforms classical baselines (BLS) and deep learning baselines (ResNet, Transformer), we will publish the weights. If it fails to outperform them, we will publish the failure analysis and explain *why*. 

The success of AstroVerse is no longer defined by winning a benchmark. It is defined by conducting the benchmark fairly, reproducibly, and transparently.

> *AstroVerse does not aim to prove that EvoMoE is the best architecture. It aims to provide a reproducible framework for rigorously evaluating when confidence-guided multimodal expert systems help, when they do not, and why.*
