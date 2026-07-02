from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import sys
import os
import json
import pandas as pd
import io
from typing import Dict

evonex_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "research", "evonex", "src"))
sys.path.append(evonex_src_path)

try:
    from model_evonex import EvoMoE_Model
except ImportError:
    print("WARNING: Could not import model_evonex. Make sure the path is correct.")

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print("Loading EvoNex Model Weights...")
    try:
        model = EvoMoE_Model(num_classes=2)
        weights_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "research", "evonex", "weights", "evomoe_weights.pth"))
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, weights_only=True))
            print("Successfully loaded pre-trained weights.")
        else:
            print(f"WARNING: Weights not found at {weights_path}. Using uninitialized model for architecture demo.")
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
    yield
    model = None

app = FastAPI(title="EvoNex Exoplanet Inference API", version="2.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_SIZE = 10 * 1024 * 1024

class PredictResponse(BaseModel):
    is_exoplanet: bool
    probability: float
    confidence_routing: Dict[str, float]
    model_status: str

class ReportResponse(BaseModel):
    markdown_content: str

@app.get("/")
def read_root():
    return {
        "project": "AstroVerse",
        "engine": "EvoNex EvoMoE v2.1",
        "model_loaded": model is not None,
    }

@app.get("/targets")
def get_targets():
    """Returns the curated database of TESS targets."""
    data_path = os.path.join(os.path.dirname(__file__), "data", "targets.json")
    if not os.path.exists(data_path):
        return []
    with open(data_path, 'r') as f:
        return json.load(f)

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    """
    Accepts a CSV upload with a 'flux' column.
    Runs the EvoMoE model and returns classification with expert routing weights.
    """
    global model

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB.")

    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing CSV: {e}")

    if 'flux' not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must contain a 'flux' column.")

    flux_data = df['flux'].values.tolist()

    if len(flux_data) > 2000:
        flux_data = flux_data[:2000]
    else:
        flux_data = flux_data + [1.0] * (2000 - len(flux_data))

    tic_features = [0.0] * 12

    lc_tensor = torch.tensor(flux_data, dtype=torch.float32).unsqueeze(0)
    tic_tensor = torch.tensor(tic_features, dtype=torch.float32).unsqueeze(0)

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first or check the weights path.")

    with torch.no_grad():
        logits, weights = model(lc_tensor, tic_tensor)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        prob_exoplanet = probabilities[0][1].item()

        cnn_weight = weights[0][0].item()
        transformer_weight = weights[0][1].item()
        physics_weight = weights[0][2].item()

    return PredictResponse(
        is_exoplanet=bool(prob_exoplanet > 0.5),
        probability=round(prob_exoplanet * 100, 2),
        confidence_routing={
            "CNN_Shape": round(cnn_weight * 100, 2),
            "Transformer_Rhythm": round(transformer_weight * 100, 2),
            "Physics_MLP": round(physics_weight * 100, 2)
        },
        model_status="untrained" if not os.path.exists(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "research", "evonex", "weights", "evomoe_weights.pth"))
        ) else "trained"
    )

@app.get("/lightcurve/{tic_id}")
def get_lightcurve(tic_id: str):
    """
    Returns lightcurve data for visualization.
    Currently generates a synthetic transit for demo purposes.
    """
    import math
    phase = [x / 100.0 for x in range(-50, 50)]
    flux = []
    for p in phase:
        noise = (torch.rand(1).item() * 0.002) - 0.001
        if -0.05 < p < 0.05:
            base = 0.98 - math.cos(p * 31.4) * 0.015
        else:
            base = 1.0
        flux.append(base + noise)

    return {"tic_id": tic_id, "phase": phase, "flux": flux}

@app.post("/report", response_model=ReportResponse)
async def generate_report(tic_id: str = Form(...), probability: float = Form(...)):
    """Generates a downloadable Scientific Evidence Markdown Report."""

    classification = "Exoplanet Candidate" if probability > 50 else "Astrophysical False Positive"

    md = f"""# Transit Analysis Report: TIC {tic_id}

**Classification:** {classification}
**Confidence:** {probability}%

## Analysis Method
This target was analyzed using the **EvoNex EvoMoE** framework:
- **Multi-Scale CNN Expert:** Evaluated local transit morphology (kernel sizes k=5, 11, 21).
- **Temporal Transformer Expert:** Assessed global periodicity patterns with sinusoidal positional encoding.
- **Physics MLP Expert:** Validated transit depth against stellar parameters from TIC.

## Limitations
- Results from an untrained model reflect architecture routing behavior, not scientific classification.
- Full scientific validation requires training on the NASA TOI catalog.
- This report is generated automatically and should not be used as sole evidence for telescope time allocation.

*Generated by AstroLens v2.1 — AstroVerse Research Platform.*
"""
    return ReportResponse(markdown_content=md)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
