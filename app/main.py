"""
FastAPI serving app for the Churn Prediction model.

Endpoints:
  GET  /health   - liveness check (used by Docker, Render, load balancers)
  GET  /metrics  - returns latest model metrics from training
  POST /predict  - accepts customer data, returns churn prediction
"""

import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.predictor import predictor


# ── Startup / shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor.load()   # load model once at startup
    yield              # app runs here
    print("Shutting down.")


app = FastAPI(
    title="Churn Prediction API",
    description="Predicts customer churn probability using a trained RandomForest model.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response schemas ─────────────────────────────────────────────────

class CustomerFeatures(BaseModel):
    """Raw customer features — same column names as the original dataset."""
    gender: str = Field(example="Male")
    SeniorCitizen: int = Field(example=0)
    Partner: str = Field(example="Yes")
    Dependents: str = Field(example="No")
    tenure: int = Field(example=12)
    PhoneService: str = Field(example="Yes")
    MultipleLines: str = Field(example="No")
    InternetService: str = Field(example="Fiber optic")
    OnlineSecurity: str = Field(example="No")
    OnlineBackup: str = Field(example="Yes")
    DeviceProtection: str = Field(example="No")
    TechSupport: str = Field(example="No")
    StreamingTV: str = Field(example="No")
    StreamingMovies: str = Field(example="No")
    Contract: str = Field(example="Month-to-month")
    PaperlessBilling: str = Field(example="Yes")
    PaymentMethod: str = Field(example="Electronic check")
    MonthlyCharges: float = Field(example=70.35)
    TotalCharges: float = Field(example=844.2)


class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_label: str
    churn_probability: float
    latency_ms: float


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness check — returns 200 if the app is running and model is loaded."""
    if predictor.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return {"status": "ok", "model_loaded": True}


@app.get("/metrics")
def metrics():
    """Returns the training metrics of the currently loaded model."""
    try:
        with open("models/metrics.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="metrics.json not found.")


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    """
    Accepts customer features and returns:
    - churn_prediction: 0 (no churn) or 1 (churn)
    - churn_label: 'Yes' or 'No'
    - churn_probability: confidence score (0.0 - 1.0)
    - latency_ms: inference time in milliseconds
    """
    start = time.time()
    try:
        result = predictor.predict(customer.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = round((time.time() - start) * 1000, 2)
    return PredictionResponse(**result, latency_ms=latency_ms)
