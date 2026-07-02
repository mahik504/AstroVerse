# Architecture Decision Records

> Key design decisions and their rationale.

---

## ADR-1: FastAPI over Flask / Django

**Decision:** Use FastAPI for the inference API.

**Context:** The API serves model predictions and lightcurve data. It needs to handle async I/O (loading HDF5 files, potential MAST queries), validate request/response schemas, and provide auto-generated documentation.

**Rationale:**
- **Async-native** — built on Starlette/ASGI; handles concurrent requests without thread pool hacks
- **Pydantic validation** — request and response models with type checking at runtime
- **Auto-generated OpenAPI docs** — Swagger UI and ReDoc out of the box, critical for a research API where endpoints are evolving
- **Performance** — faster than Flask for I/O-bound workloads

**Rejected alternatives:**
- Flask — synchronous by default, requires extensions for validation and docs
- Django / DRF — too heavyweight for a single-purpose inference API; ORM is unnecessary

---

## ADR-2: Mixture-of-Experts over Monolithic CNN

**Decision:** Decompose the model into three specialized experts with learned gating.

**Context:** Transit detection involves multiple signal types — local morphology, global periodicity, and astrophysical constraints. A single CNN conflates these concerns.

**Rationale:**
- **Explainability** — routing weights show which expert drove each prediction (was it shape? periodicity? stellar physics?)
- **Physics integration** — the Physics MLP expert directly ingests stellar parameters, which cannot be naturally encoded as a 1D signal for a CNN
- **Modularity** — each expert can be developed, tested, and ablated independently
- **Research value** — gating analysis reveals which signal types matter for different target classes

**Trade-off:** Higher architectural complexity and more hyperparameters than a single network.

---

## ADR-3: HDF5 over SQLite / Parquet

**Decision:** Cache preprocessed data as HDF5 files.

**Context:** Preprocessed lightcurves are fixed-length float32 tensors (2000 points). Stellar features are 12-element float vectors. The training loop needs fast, random-access reads.

**Rationale:**
- **Tensor-native** — HDF5 stores N-dimensional arrays natively; no serialization/deserialization overhead
- **Partial reads** — can load a single dataset or slice without reading the entire file
- **Compression** — built-in gzip/lzf compression reduces disk usage
- **PyTorch compatibility** — `h5py` integrates cleanly with `torch.utils.data.Dataset`

**Rejected alternatives:**
- SQLite — excellent for relational data, but storing float arrays as BLOBs adds complexity and loses partial-read capability
- Parquet — columnar format optimized for tabular analytics, not tensor workloads; poor support for multi-dimensional arrays

---

## ADR-4: Transformer over LSTM

**Decision:** Use a Transformer with sinusoidal positional encoding for the temporal expert.

**Context:** The temporal expert must capture long-range periodic patterns across 2000-point lightcurves.

**Rationale:**
- **Parallel training** — self-attention processes all positions simultaneously; LSTMs are inherently sequential
- **Positional encoding** — sinusoidal encoding provides explicit temporal structure without learned parameters
- **Attention maps** — can visualize which time regions the model attends to, aiding interpretability
- **Long-range dependencies** — attention has O(1) path length between any two positions; LSTMs degrade over long sequences

**Trade-off:** Higher memory cost (O(n²) attention) than LSTMs for very long sequences, but manageable at n=2000.

---

## ADR-5: BLS over Wavelet / CNN Period Detection

**Decision:** Use Box Least Squares (BLS) for period detection in the preprocessing pipeline.

**Context:** Before the model sees data, the lightcurve must be phase-folded on the orbital period. This requires a period detection step.

**Rationale:**
- **Astrophysics standard** — BLS (Kovács et al. 2002) is the established method for transit period detection; results are directly comparable to published literature
- **Interpretable** — outputs a period, epoch, duration, and depth — all physically meaningful quantities
- **Well-tested** — mature implementations in `astropy` and `lightkurve`
- **No training required** — a signal processing algorithm, not a learned model

**Rejected alternatives:**
- Wavelet transforms — powerful for multi-scale analysis but period extraction requires additional post-processing
- CNN-based period detection — would require a separate trained model, adding a chicken-and-egg problem for an untrained system

---

## ADR-6: Next.js over Plain React

**Decision:** Use Next.js 16 for the AstroLens dashboard.

**Context:** The dashboard displays lightcurve plots, prediction results, and target metadata. It connects to the FastAPI backend.

**Rationale:**
- **Server-side rendering** — pre-renders pages for faster initial load; useful when displaying large lightcurve datasets
- **File-based routing** — maps directory structure to URL paths; reduces boilerplate
- **Tailwind CSS integration** — first-class support for utility-first styling
- **API routes** — can proxy requests to the FastAPI backend, simplifying CORS configuration

**Rejected alternatives:**
- Plain React (Vite) — would work, but loses SSR, file routing, and integrated API proxying; more configuration needed for the same result
