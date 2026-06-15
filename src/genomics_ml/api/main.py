"""
FastAPI prediction service for the genomics ML pipeline.

Endpoints needed:
  GET  /health      — return {"status": "ok"}
  GET  /model-info  — return model metadata (type, n_features, status)
  POST /predict     — accept feature values, return prediction + probability

Steps:
  1. Create FastAPI app with title and version.
  2. Load the trained model using load_model() from genomics_ml.models.predict.
  3. Extract MODEL_TYPE and N_FEATURES from the loaded pipeline.
  4. Define Pydantic models for request/response.
  5. Implement the three endpoints using existing functions.

Reuse existing code:
  - genomics_ml.models.predict.load_model  — instead of raw joblib.load
  - genomics_ml.models.predict.predict     — instead of model.predict()
  - genomics_ml.models.predict.predict_proba — instead of model.predict_proba()

See teaching_examples/example_fastapi_service.py for reference.
"""

from importlib.metadata import version

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
# Reuse central model utilities instead of calling joblib directly
from genomics_ml.models.model_utils import (
    load_model,
    predict,
    predict_proba,
    get_model_type,
    get_n_features,
)
from genomics_ml.utils.config import get_config_path, load_config

# ── FastAPI app ───────────────────────────────────────────────────────
# Read version from installed package metadata (single source of truth = pyproject.toml)
APP_VERSION = version("genomics-ml")
app = FastAPI(title="Genomics ML API", version=APP_VERSION)

# ── Model loading (runs once at startup) ──────────────────────────────
# Load model path from config (configs/default.yaml → model.path)
config = load_config(get_config_path())
MODEL_PATH = config["model"]["path"]
try:
    model = load_model(MODEL_PATH)
    # Extract model type from the pipeline's classifier step
    MODEL_TYPE = get_model_type(model)
    # Number of features expected (from the scaler)
    N_FEATURES = get_n_features(model)
except Exception as e:
    model = None
    MODEL_TYPE = "unknown"
    N_FEATURES = 0
    print(f"WARNING: Could not load model from {MODEL_PATH}: {e}")

# ── Pydantic models ───────────────────────────────────────────────────
class PredictRequest(BaseModel):
    """Input: a list of gene expression feature values."""
    features: list[float] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Gene expression feature values",
    )


class PredictResponse(BaseModel):
    """Output: predicted class label and confidence."""
    prediction: int = Field(..., description="Predicted class (0 or 1)")
    probability: float = Field(..., description="Confidence score for class 1")


class ModelInfoResponse(BaseModel):
    """Metadata about the loaded model."""
    model_type: str
    n_features: int
    status: str


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness check — returns OK if the service is running."""
    return {"status": "ok"}


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    """Return metadata about the currently loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    return ModelInfoResponse(
        model_type=MODEL_TYPE,
        n_features=N_FEATURES,
        status="loaded",
    )

@app.post("/predict", response_model=PredictResponse)
def predict_(req: PredictRequest):
    """Accept feature values, return a prediction and probability."""
    if model is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    # Validate feature count matches model expectations
    if len(req.features) != N_FEATURES:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {N_FEATURES} features, got {len(req.features)}",
        )

    try:
        # Convert input to numpy array (shape: 1 x n_features)
        X = np.array([req.features])
        pred = predict(model, X)          # returns array like [0] or [1]
        proba = predict_proba(model, X)   # returns array like [[p0, p1]]
        return PredictResponse(
            prediction=int(pred[0]),
            probability=round(float(proba[0][1]), 4),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
