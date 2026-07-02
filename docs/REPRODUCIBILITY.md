# Reproducibility Guide

> How to reproduce every result in the AstroVerse project from scratch.

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| CPU | Any modern x86-64 | 8+ cores |
| RAM | 8 GB | 16 GB |
| GPU | Not required | CUDA-capable (training speedup) |
| Disk | 10 GB free | 50 GB (for full TOI catalog) |

CPU is sufficient for architecture validation and small-scale experiments. GPU accelerates training but is not required.

---

## Software Versions

| Tool | Version |
|---|---|
| Python | 3.11+ |
| PyTorch | 2.0+ |
| Node.js | 18+ |
| npm/pnpm | Latest stable |
| lightkurve | 2.x |
| FastAPI | 0.100+ |
| ruff | Latest |

> [!TIP]
> Use a virtual environment (`venv` or `conda`) to isolate Python dependencies.

---

## Random Seeds

> [!WARNING]
> Random seeds are **not yet configured** in the training pipeline. Results will vary between runs until deterministic seeding is added.
>
> **Recommendation:** Add `torch.manual_seed()`, `numpy.random.seed()`, and `torch.backends.cudnn.deterministic = True` to the training entry point. Track the seed value in experiment configs.

---

## Step-by-Step Reproduction

### Step 1: Clone and Install

```bash
git clone https://github.com/<org>/AstroVerse.git
cd AstroVerse

# Python environment (research + API)
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

pip install -r research/evonex/requirements.txt
pip install -r services/evonex-api/requirements.txt

# Frontend
cd apps/astrolens-web
npm install
cd ../..
```

**Expected output:** All dependencies install without errors.

### Step 2: Download Data

Data comes from the NASA MAST archive via the `lightkurve` Python library.

```bash
# Download TESS lightcurves for specific targets
python research/evonex/scripts/download_data.py --targets "TIC 261136679,TIC 307210830"
```

This fetches PDCSAP flux lightcurves and TIC stellar parameters.

**Expected output:** Raw FITS files downloaded to `data/raw/`.

> [!NOTE]
> For the full TOI catalog (Threshold Crossing Events), download from the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/). This is required for large-scale training but is not yet integrated into the automated pipeline.

### Step 3: Preprocess

```bash
python research/evonex/scripts/preprocess.py
```

Pipeline:
1. Remove NaN flux values
2. Sigma-clip outliers
3. Normalize flux to zero median
4. Run BLS period detection
5. Phase-fold on best period
6. Resample to 2000 points
7. Cache as HDF5

**Expected output:** `data/processed/*.h5` files — one per target containing lightcurve tensors, stellar features, and metadata.

### Step 4: Validate Architecture

```bash
python research/evonex/scripts/validate_architecture.py
```

Runs a forward pass with random data to verify tensor shapes and parameter counts.

**Expected output:**
- Confirmation that the model accepts `(batch, 2000)` lightcurves and `(batch, 12)` stellar features
- Total trainable parameters: 1,485,605
- Output shape: `(batch, 1)` classification + `(batch, 3)` routing weights

### Step 5: Train

> [!IMPORTANT]
> Large-scale training is **not yet complete**. The commands below describe the intended workflow once a sufficient labeled dataset is prepared.

```bash
python research/evonex/train.py \
  --config research/evonex/configs/default.yaml \
  --epochs 100 \
  --batch-size 32 \
  --lr 1e-4
```

**Expected output:** Model checkpoints saved to `research/evonex/checkpoints/`. Training metrics logged to experiment tracker.

### Step 6: Evaluate

```bash
python research/evonex/evaluate.py \
  --checkpoint research/evonex/checkpoints/best.pt \
  --data data/processed/test.h5
```

**Expected output:** Pending evaluation — no trained weights or test set are available yet.

### Step 7: Serve API

```bash
cd services/evonex-api
uvicorn main:app --reload --port 8000
```

**Expected output:** API running at `http://localhost:8000`. The `/predict` endpoint returns `503` until trained weights are loaded.

### Step 8: Launch Dashboard

```bash
cd apps/astrolens-web
npm run dev
```

**Expected output:** Next.js dev server at `http://localhost:3000`. Dashboard displays lightcurve visualizations and connects to the API.

---

## What Is Verified vs. Pending

| Step | Status |
|---|---|
| Installation | ✅ Verified |
| Data download (small set) | ✅ Verified |
| Preprocessing pipeline | ✅ Verified |
| Architecture validation | ✅ Verified |
| Large-scale training | ⏳ Pending |
| Evaluation metrics | ⏳ Pending |
| End-to-end prediction | ⏳ Pending (requires trained weights) |
