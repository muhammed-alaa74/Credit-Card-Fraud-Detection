from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import joblib
import numpy as np
from pathlib import Path
import time
 
# =========================
# App Setup
# =========================
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud detection using Random Forest & XGBoost ensemble.",
    version="1.0.0",
)
 
# =========================
# Load Models
# =========================
BASE_DIR = Path(__file__).resolve().parent
 
try:
    rf_model  = joblib.load(BASE_DIR / "models" / "random_forest.pkl")
    xgb_model = joblib.load(BASE_DIR / "models" / "xgboost.pkl")
except FileNotFoundError as e:
    raise RuntimeError(f"Model file not found: {e}")
 
# =========================
# Expected feature count
# (V1-V28 + scaled_amount + scaled_time +
#  log_amount + Hour + is_night + high_amount + amount_per_hour)
# =========================
EXPECTED_FEATURES = 35
FRAUD_THRESHOLD   = 0.20
 
# =========================
# Schemas
# =========================
class FeaturesInput(BaseModel):
    features: list[float]
 
    @field_validator("features")
    @classmethod
    def check_length(cls, v):
        if len(v) != EXPECTED_FEATURES:
            raise ValueError(
                f"Expected {EXPECTED_FEATURES} features, got {len(v)}."
            )
        return v
 
class SinglePrediction(BaseModel):
    model:             str
    fraud_probability: float
    fraud_percentage:  str
    prediction:        str
    confidence:        str
 
class EnsembleResponse(BaseModel):
    random_forest:    SinglePrediction
    xgboost:          SinglePrediction
    ensemble_verdict: str
    most_confident:   str
    latency_ms:       float
 
# =========================
# Helper
# =========================
def _predict_single(model, data: np.ndarray, model_name: str) -> SinglePrediction:
    prob = float(model.predict_proba(data)[0][1])
    pred = prob > FRAUD_THRESHOLD
    conf = abs(prob - 0.5)
 
    return SinglePrediction(
        model             = model_name,
        fraud_probability = round(prob, 6),
        fraud_percentage  = f"{prob * 100:.2f}%",
        prediction        = "Fraud" if pred else "Legit",
        confidence        = f"{conf:.4f}",
    )
 
# =========================
# Routes
# =========================
@app.get("/", tags=["General"])
def root():
    return {
        "api":     "Fraud Detection API",
        "version": "1.0.0",
        "status":  "running",
        "docs":    "/docs",
    }
 
 
@app.get("/health", tags=["General"])
def health():
    return {
        "status":          "healthy",
        "models_loaded":   ["random_forest", "xgboost"],
        "fraud_threshold": FRAUD_THRESHOLD,
        "expected_features": EXPECTED_FEATURES,
    }
 
 
@app.post("/predict", response_model=EnsembleResponse, tags=["Prediction"])
def predict(payload: FeaturesInput):
    """
    Run both Random Forest and XGBoost on a single transaction.
 
    - **features**: list of exactly 35 floats (engineered features).
    - Returns fraud probability, prediction, and confidence for each model,
      plus an ensemble verdict.
    """
    try:
        start = time.perf_counter()
        data  = np.array(payload.features).reshape(1, -1)
 
        rf_result  = _predict_single(rf_model,  data, "Random Forest")
        xgb_result = _predict_single(xgb_model, data, "XGBoost")
 
        # Ensemble verdict: fraud if either model flags it
        both_fraud   = rf_result.prediction == "Fraud" and xgb_result.prediction == "Fraud"
        either_fraud = rf_result.prediction == "Fraud" or  xgb_result.prediction == "Fraud"
        verdict = "Fraud" if either_fraud else "Legit"
 
        # Most confident model
        rf_conf  = float(rf_result.confidence)
        xgb_conf = float(xgb_result.confidence)
        most_confident = "Random Forest" if rf_conf > xgb_conf else "XGBoost"
 
        latency = round((time.perf_counter() - start) * 1000, 3)
 
        return EnsembleResponse(
            random_forest    = rf_result,
            xgboost          = xgb_result,
            ensemble_verdict = verdict,
            most_confident   = most_confident,
            latency_ms       = latency,
        )
 
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
 
 
@app.post("/predict/random-forest", response_model=SinglePrediction, tags=["Prediction"])
def predict_rf(payload: FeaturesInput):
    """Run Random Forest only."""
    try:
        data = np.array(payload.features).reshape(1, -1)
        return _predict_single(rf_model, data, "Random Forest")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 
 
@app.post("/predict/xgboost", response_model=SinglePrediction, tags=["Prediction"])
def predict_xgb(payload: FeaturesInput):
    """Run XGBoost only."""
    try:
        data = np.array(payload.features).reshape(1, -1)
        return _predict_single(xgb_model, data, "XGBoost")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 