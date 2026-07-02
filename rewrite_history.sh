#!/bin/bash
set -e

echo "Starting Git history rewrite..."

# Create a new orphaned branch
git checkout --orphan new_main

# Unstage all files
git rm -rf --cached .

# 1. Initial project structure
git add .gitignore README.md Makefile CITATION.cff Dockerfile
git commit -m "Initial project structure"

# 2. Implement EvoNex architecture
git add research/evonex/src/model_evonex.py research/evonex/tests/test_model.py research/evonex/configs/
git commit -m "Implement EvoNex architecture"

# 3. Implement AstroLens frontend
git add apps/astrolens-web/
git commit -m "Implement AstroLens frontend"

# 4. Implement FastAPI inference service
git add services/evonex-api/
git commit -m "Implement FastAPI inference service"

# 5. Implement TESS preprocessing pipeline
git add research/evonex/src/build_dataset.py research/evonex/src/fetch_toi_catalog.py research/evonex/tests/test_preprocessing.py research/evonex/datasets/
git commit -m "Implement TESS preprocessing pipeline"

# 6. Add experiment tracking
git add research/evonex/src/run_baselines.py research/evonex/src/run_ablations.py research/evonex/experiments/ research/evonex/analysis/
git commit -m "Add experiment tracking"

# 7. Improve documentation
git add docs/ paper/ reproducibility_manifest.json Zenodo_archive.sh
git commit -m "Improve documentation"

# 8. Prepare v1.0.0-beta
# Add everything else that might have been missed
git add .
git commit -m "Prepare v1.0.0-beta"

# Force replace the main branch
git branch -D main || true
git branch -m main

# Create the tag
git tag -f v1.0.0-beta

echo "Git history successfully rewritten!"
git log --oneline
