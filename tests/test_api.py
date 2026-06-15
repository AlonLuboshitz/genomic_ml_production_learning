"""Tests for the FastAPI prediction service."""

from fastapi.testclient import TestClient

from genomics_ml.api.main import app

client = TestClient(app)


class TestHealth:
    """Tests for GET /health."""

    def test_health_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestModelInfo:
    """Tests for GET /model-info."""

    def test_model_info_returns_metadata(self):
        response = client.get("/model-info")
        assert response.status_code == 200
        data = response.json()
        assert "model_type" in data
        assert "n_features" in data
        assert data["status"] == "loaded"


class TestPredict:
    """Tests for POST /predict."""

    def test_predict_valid_input(self):
        features = [0.1] * 30
        response = client.post("/predict", json={"features": features})
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert data["prediction"] in (0, 1)

    def test_predict_wrong_feature_count(self):
        response = client.post("/predict", json={"features": [0.1, 0.2]})
        assert response.status_code == 422

    def test_predict_empty_body(self):
        response = client.post("/predict", json={})
        assert response.status_code == 422
