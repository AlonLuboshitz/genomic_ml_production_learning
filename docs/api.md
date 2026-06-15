# API Documentation

## Overview

The Genomics ML API serves predictions from a trained scikit-learn model via REST endpoints. It loads the model once at startup and accepts POST requests with gene expression feature values.

**Base URL:** `http://localhost:8000`

**Interactive Docs:** `http://localhost:8000/docs` (Swagger UI)

---

## Run the server

```bash
make serve
# or directly:
uvicorn genomics_ml.api.main:app --reload --port 8000
```

---

## Endpoints

### GET /health

Liveness check — proves the service is running.

**Request:**
```bash
curl http://localhost:8000/health
```

**Response:** `200 OK`
```json
{
  "status": "ok"
}
```

---

### GET /model-info

Returns metadata about the currently loaded model.

**Request:**
```bash
curl http://localhost:8000/model-info
```

**Response:** `200 OK`
```json
{
  "model_type": "RandomForestClassifier",
  "n_features": 30,
  "status": "loaded"
}
```

**Errors:**
- `503` — No model loaded (model file missing or failed to load)

---

### POST /predict

Run inference on a set of gene expression features.

**Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}'
```

Or use the example file:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @examples/predict_example.json
```

**Request body:**
```json
{
  "features": [0.1, 0.2, ..., 0.3]
}
```

| Field | Type | Description |
|---|---|---|
| `features` | `list[float]` | Exactly 30 gene expression values (must match model's `n_features`) |

**Response:** `200 OK`
```json
{
  "prediction": 0,
  "probability": 0.87
}
```

| Field | Type | Description |
|---|---|---|
| `prediction` | `int` | Predicted class (0 or 1) |
| `probability` | `float` | Confidence score for class 1 |

**Errors:**
- `422` — Wrong number of features, missing body, or invalid JSON
- `503` — No model loaded

---

## Error codes summary

| Status | Meaning |
|---|---|
| `200` | Success |
| `422` | Validation error (wrong input format) |
| `500` | Internal server error |
| `503` | Service unavailable (model not loaded) |
