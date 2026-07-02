# EvoNex API Reference

> FastAPI inference API for the EvoMoE exoplanet transit detector.

Base URL: `http://localhost:8000` (default)

Interactive docs: `http://localhost:8000/docs` (Swagger UI)

---

## Endpoints

### `GET /` — Health Check

Returns API status and version.

```bash
curl http://localhost:8000/
```

**Response** `200 OK`:

```json
{
  "service": "evonex-api",
  "status": "running",
  "version": "0.4.0"
}
```

---

### `GET /targets` — List Available Targets

Returns the list of TIC targets that have been preprocessed and cached.

```bash
curl http://localhost:8000/targets
```

**Response** `200 OK`:

```json
{
  "targets": [
    {
      "tic_id": "TIC 261136679",
      "name": "WASP-18 b",
      "sector": 2,
      "cached": true
    }
  ],
  "count": 4
}
```

---

### `POST /predict` — Run Transit Prediction

Submit a TIC ID for transit classification. The model returns a transit probability and expert routing weights.

> [!IMPORTANT]
> This endpoint requires trained model weights to be loaded. If no weights are available, the API will return a `503 Service Unavailable` error. As of this writing, trained weights are **not yet available** — only architecture validation has been completed.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tic_id": "TIC 261136679"}'
```

**Request body:**

```json
{
  "tic_id": "TIC 261136679"
}
```

**Response** `200 OK` (when model weights are loaded):

```json
{
  "tic_id": "TIC 261136679",
  "transit_probability": 0.0,
  "routing_weights": {
    "cnn_expert": 0.0,
    "transformer_expert": 0.0,
    "physics_expert": 0.0
  },
  "model_version": "0.4.0"
}
```

> [!NOTE]
> The values above are placeholders. Actual predictions require trained weights. The `routing_weights` field shows the softmax gating output — which expert contributed most to the prediction.

**Response** `503 Service Unavailable` (no weights loaded):

```json
{
  "detail": "Model weights not loaded. Train the model first."
}
```

**Response** `404 Not Found` (unknown TIC ID):

```json
{
  "detail": "Target TIC 999999999 not found in cache."
}
```

---

### `GET /lightcurve/{tic_id}` — Fetch Lightcurve Data

Returns the raw and phase-folded lightcurve data for a given TIC target.

```bash
curl http://localhost:8000/lightcurve/261136679
```

**Response** `200 OK`:

```json
{
  "tic_id": "TIC 261136679",
  "sector": 2,
  "n_points": 2000,
  "bls_period_days": null,
  "phase": [],
  "flux": []
}
```

> [!NOTE]
> The `phase` and `flux` arrays contain the phase-folded lightcurve resampled to 2000 points. The `bls_period_days` field contains the BLS-detected orbital period, if available.

**Response** `404 Not Found`:

```json
{
  "detail": "Lightcurve for TIC 999999999 not found."
}
```

---

### `POST /report` — Generate Target Report

Generates a summary report for a given target, combining lightcurve metadata, stellar parameters, and prediction results (if available).

```bash
curl -X POST http://localhost:8000/report \
  -H "Content-Type: application/json" \
  -d '{"tic_id": "TIC 261136679"}'
```

**Request body:**

```json
{
  "tic_id": "TIC 261136679"
}
```

**Response** `200 OK`:

```json
{
  "tic_id": "TIC 261136679",
  "stellar_params": {
    "Teff": null,
    "logg": null,
    "radius": null
  },
  "lightcurve_summary": {
    "n_points": 2000,
    "sector": 2,
    "bls_period_days": null
  },
  "prediction": null,
  "generated_at": "2026-07-02T00:00:00Z"
}
```

> [!NOTE]
> The `prediction` field is `null` when no trained model weights are loaded.

---

## Error Codes

| Code | Meaning |
|---|---|
| `200` | Success |
| `404` | Target not found in cache |
| `422` | Validation error (malformed request body) |
| `500` | Internal server error |
| `503` | Model weights not loaded |

---

## Running the API

```bash
cd services/evonex-api
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

See Swagger UI at `http://localhost:8000/docs` for interactive testing.
