# Data Governance

This document outlines the sourcing, licensing, and attribution rules for the AstroVerse dataset construction pipeline.

## 1. Sourcing
- **Ground Truth Labels**: Sourced exclusively from the NASA Exoplanet Archive (Caltech/IPAC).
- **Stellar Metadata**: Sourced from the TESS Input Catalog (TIC) and Candidate Target List (CTL).
- **Observations**: Sourced from the Mikulski Archive for Space Telescopes (MAST).

## 2. Licensing and Redistribution
- TESS data is public domain. There are no restrictions on the redistribution of the downloaded `.fits` files or the processed `.h5` files.
- If this dataset is republished (e.g., on Kaggle, HuggingFace, Zenodo), it must be licensed under CC BY 4.0 or equivalent, acknowledging the original NASA sources.

## 3. Storage Policy
- Raw `.fits` files downloaded via `lightkurve` are cached temporarily but are **not** committed to version control due to file size limits.
- The processed phase-folded tensors are stored in `tess_cache.h5` files locally. These are ignored by git (`.gitignore`).
- Only the `dataset_manifest.json`, `duplicate_report.csv`, and configurations are tracked in git to ensure reproducibility without bloating the repository.

## 4. Attribution Requirements
Any publications resulting from this project must include standard acknowledgments for:
- NASA Exoplanet Archive
- MAST (Mikulski Archive for Space Telescopes)
- The TESS Science Team
- The `lightkurve` developer community

## 5. Privacy
No personally identifiable information (PII) is involved in this astrophysical dataset.
