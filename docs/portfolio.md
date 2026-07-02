# AstroVerse Portfolio Assets

This document contains pre-written materials for quickly referencing AstroVerse in resumes, LinkedIn posts, or hackathon pitches.

## 1. One-Sentence Summary
AstroVerse is an open-source, reproducible exoplanet detection pipeline powered by an explainable Mixture-of-Experts (EvoMoE) neural architecture.

## 2. Resume Bullet Points
- **Machine Learning & Astrophysics**: Engineered "AstroVerse," an end-to-end exoplanet detection platform utilizing a Confidence-Guided Mixture-of-Experts architecture (EvoMoE) to classify NASA TESS light curves.
- **Explainable AI Architecture**: Designed a multi-modal fusion mechanism combining a 1D CNN, Time-Series Transformer, and Physics MLP, reducing "black-box" uncertainty for astronomical validation.
- **Reproducible Pipeline Engineering**: Developed a robust data pipeline integrating MAST API querying, Box Least Squares (BLS) phase-folding, and HDF5 tensor caching for large-scale PyTorch training.
- **Full-Stack Deployment**: Deployed the model weights via a FastAPI backend connected to a Next.js 16 UI (AstroLens), enabling interactive human-in-the-loop expert routing visualization.

## 3. LinkedIn Launch Post Draft
I'm excited to share my latest research project: **AstroVerse**! 🚀

Exoplanet transit detection with deep learning often suffers from a "black-box" problem—models make predictions, but astrophysicists can't trust *why*.

Over the past few months, I built an end-to-end pipeline that introduces **EvoMoE**, a Confidence-Guided Mixture-of-Experts architecture. Instead of a monolithic network, EvoMoE dynamically routes NASA TESS light curve data between:
🔹 A Multi-Scale CNN (for local transit morphology)
🔹 A Time-Series Transformer (for global periodicity)
🔹 A Physics MLP (validating stellar parameters)

Not only does it detect transits, but the dashboard explicitly visualizes *which expert* made the decision, bringing interpretability back to deep learning.

The entire project is open-source, reproducible with a few make commands, and includes a full Next.js mission control dashboard. 

Check out the code, architecture diagrams, and the preprint draft here: [GitHub Link]

## 4. 30-Second Elevator Pitch
"Hi, I'm the creator of AstroVerse. It's an AI-powered scientific pipeline for discovering exoplanets in NASA TESS data. The core problem it solves is that deep learning models are usually black boxes, which scientists don't trust. So, I built a Confidence-Guided Mixture-of-Experts architecture that splits the analysis into morphological, temporal, and physical 'experts.' It predicts if a signal is a planet and explicitly tells the researcher *why*. I built the entire stack—from the PyTorch model and the automated dataset generation script to the FastAPI backend and Next.js mission control dashboard."

## 5. The OIST Professor Pitch
"I designed and implemented an end-to-end research platform for reproducible exoplanet transit detection using a confidence-guided Mixture-of-Experts architecture. The project emphasizes reproducibility, open science, explainability, and systematic evaluation against established baselines."

## 6. The Recruiter Pitch
"This repository demonstrates my experience across machine learning, scientific software engineering, backend APIs, frontend visualization, reproducible experimentation, and technical documentation. It includes a complete research workflow rather than only a trained model."
