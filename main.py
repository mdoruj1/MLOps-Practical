"""
Phase 3: FastAPI Model Serving
-------------------------------
Run locally : uvicorn main:app --host 0.0.0.0 --port 80 --reload
Swagger UI  : http://localhost:80/docs

Send a prediction:
  curl -X POST http://localhost:80/predict \\
       -H "Content-Type: application/json" \\
       -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
"""

import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── Pydantic schema (input validation) ────────────────────────────────────────
class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, description="Sepal length in cm", example=5.1)
    sepal_width:  float = Field(..., gt=0, description="Sepal width in cm",  example=3.5)
    petal_length: float = Field(..., gt=0, description="Petal length in cm", example=1.4)
    petal_width:  float = Field(..., gt=0, description="Petal width in cm",  example=0.2)


class PredictionResponse(BaseModel):
    species:      str
    class_id:     int
    confidence:   float
    probabilities: dict


# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Iris Classifier API",
    description="MLOps demo: RandomForest model served via FastAPI + Docker",
    version="1.0.0",
)

SPECIES_MAP = {0: "setosa", 1: "versicolor", 2: "virginica"}

# Load model at startup (not per-request)
try:
    with open("models/rf_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Iris Classifier API is running 🌸"}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(features: IrisFeatures):
    """
    Accept iris measurements and return predicted species + confidence.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ]])

    class_id   = int(model.predict(X)[0])
    proba      = model.predict_proba(X)[0]
    confidence = float(proba[class_id])

    return PredictionResponse(
        species=SPECIES_MAP[class_id],
        class_id=class_id,
        confidence=round(confidence, 4),
        probabilities={SPECIES_MAP[i]: round(float(p), 4) for i, p in enumerate(proba)},
    )
