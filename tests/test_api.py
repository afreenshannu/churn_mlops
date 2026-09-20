"""
Tests for the churn prediction API.
Run locally with: python -m pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.predictor import predictor

# Load model once before all tests run
@pytest.fixture(scope="session", autouse=True)
def load_model():
    predictor.load()

client = TestClient(app)

SAMPLE_CUSTOMER = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 844.20,
}

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True

def test_predict_returns_200():
    response = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert response.status_code == 200

def test_predict_response_schema():
    response = client.post("/predict", json=SAMPLE_CUSTOMER)
    data = response.json()
    assert "churn_prediction" in data
    assert "churn_label" in data
    assert "churn_probability" in data
    assert "latency_ms" in data

def test_predict_values_are_valid():
    response = client.post("/predict", json=SAMPLE_CUSTOMER)
    data = response.json()
    assert data["churn_prediction"] in [0, 1]
    assert data["churn_label"] in ["Yes", "No"]
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["latency_ms"] >= 0

def test_predict_label_matches_prediction():
    response = client.post("/predict", json=SAMPLE_CUSTOMER)
    data = response.json()
    expected_label = "Yes" if data["churn_prediction"] == 1 else "No"
    assert data["churn_label"] == expected_label

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert "f1" in data["metrics"]

def test_missing_field_returns_422():
    incomplete = {k: v for k, v in SAMPLE_CUSTOMER.items() if k != "tenure"}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422

def test_latency_under_threshold():
    response = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert response.json()["latency_ms"] < 500
