#!/bin/bash
# Zenodo Archival Script for AstroVerse v1.0.0-research
# This script bundles the reproducibility manifest, the core dataset, and the model weights
# for submission to Zenodo to acquire a DOI.

set -e

echo "Building AstroVerse Zenodo Archive..."

ARCHIVE_NAME="AstroVerse_v1.0.0_research.tar.gz"
CHECKSUM_FILE="AstroVerse_checksums.txt"

# 1. Bundle the code, manifest, and environment
tar -czvf $ARCHIVE_NAME \
    --exclude='.git' \
    --exclude='services/evonex-api/venv' \
    --exclude='apps/astrolens-web/node_modules' \
    reproducibility_manifest.json \
    research/evonex/src/ \
    research/evonex/configs/ \
    research/evonex/experiments/ \
    research/evonex/analysis/ \
    paper/

# 2. Generate SHA-256 Checksums
echo "Generating SHA-256 checksums..."
sha256sum $ARCHIVE_NAME > $CHECKSUM_FILE

echo "Archive complete: $ARCHIVE_NAME"
echo "Checksums saved to: $CHECKSUM_FILE"
echo "Ready for Zenodo upload via REST API."
