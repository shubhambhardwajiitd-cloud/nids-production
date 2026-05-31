import time
import uuid
import logging
import numpy as np
import pandas as pd
import joblib
import shap
from datetime import datetime
from collections import defaultdict
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ── Rate limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="NIDS API",
    description="Network Intrusion Detection System with SHAP explainability",
    version="1.0.0"
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API Keys ──────────────────────────────────────────────────────────────────
API_KEYS = {
    "dev-key-free-001":     {"tier": "free",         "limit": 1000},
    "starter-key-001":      {"tier": "starter",      "limit": 50000},
    "professional-key-001": {"tier": "professional", "limit": 100000},
}

# ── Severity thresholds ───────────────────────────────────────────────────────
SEVERITY_THRESHOLDS = [
    ("CRITICAL", 0.95),
    ("HIGH",     0.80),
    ("MEDIUM",   0.60),
    ("LOW",      0.0),
]

# ── In-memory stores ──────────────────────────────────────────────────────────
alert_store: List[Dict[str, Any]] = []
stats_store: Dict[str, int] = defaultdict(int)

# ── Load models ───────────────────────────────────────────────────────────────
import os
MODEL_DIR = os.environ.get("MODEL_DIR", "./model")

try:
    models = {
        "xgboost":       joblib.load(f"{MODEL_DIR}/model_xgb.pkl"),
        "random_forest": joblib.load(f"{MODEL_DIR}/model_sklearn.pkl"),
    }
    label_encoder = joblib.load(f"{MODEL_DIR}/label_encoder.pkl")
    feature_names = joblib.load(f"{MODEL_DIR}/feature_names.pkl")

    # SHAP explainers — created once at startup
    explainers = {
        name: shap.TreeExplainer(model)
        for name, model in models.items()
    }
    logger.info(f"✅ Models loaded successfully")
    logger.info(f"   Classes : {list(label_encoder.classes_)}")
    logger.info(f"   Features: {len(feature_names)}")
except Exception as e:
    logger.error(f"❌ Failed to load models: {e}")
    raise

# ── Helpers ───────────────────────────────────────────────────────────────────

def verify_api_key(x_api_key: Optional[str]) -> dict:
    if not x_api_key or x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key. Pass X-Api-Key header.")
    return API_KEYS[x_api_key]


def get_severity(confidence: float) -> str:
    for severity, threshold in SEVERITY_THRESHOLDS:
        if confidence >= threshold:
            return severity
    return "LOW"


def get_shap_explanation(model_name: str, X_row: np.ndarray, pred_class: int) -> List[Dict]:
    try:
        explainer   = explainers[model_name]
        shap_values = explainer.shap_values(X_row)

        # Handle both old (list) and new (3D array) SHAP formats
        if isinstance(shap_values, list):
            vals = shap_values[pred_class][0]
        else:
            vals = shap_values[0, :, pred_class]

        top3_idx = np.argsort(np.abs(vals))[::-1][:3]

        return [
            {
                "feature":    str(feature_names[i]),
                "shap_value": round(float(vals[i]), 4),
                "direction":  "increases_risk" if vals[i] > 0 else "decreases_risk",
            }
            for i in top3_idx
        ]
    except Exception as e:
        logger.warning(f"SHAP explanation failed: {e}")
        return []


def run_rule_engine(flow: dict, ml_prediction: str, ml_confidence: float):
    """Port-based rule overrides for deterministic attack types."""
    if ml_prediction == "BENIGN":
        return ml_prediction, ml_confidence

    dst_port     = int(flow.get("Dst Port", 0))
    tot_fwd_pkts = float(flow.get("Tot Fwd Pkts", 0))
    pkt_len_mean = float(flow.get("Fwd Pkt Len Mean", 0))

    if dst_port == 21 and tot_fwd_pkts > 10 and pkt_len_mean < 100:
        return "FTP-BruteForce", max(ml_confidence, 0.88)

    if dst_port == 22 and tot_fwd_pkts > 10 and pkt_len_mean < 100:
        return "SSH-BruteForce", max(ml_confidence, 0.88)

    return ml_prediction, ml_confidence


def predict_flow(flow: dict, model_name: str) -> dict:
    """Core prediction logic — reused by /predict and /predict/batch."""
    model = models[model_name]

    # Build feature vector in exact training order
    row = [float(flow.get(f, 0)) for f in feature_names]
    X   = np.array(row, dtype=np.float32).reshape(1, -1)
    X   = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    pred_idx   = int(model.predict(X)[0])
    pred_label = label_encoder.classes_[pred_idx]
    confidence = float(model.predict_proba(X).max())

    # Rule engine override
    pred_label, confidence = run_rule_engine(flow, pred_label, confidence)

    # SHAP only on attacks — saves CPU
    shap_features = []
    if pred_label != "BENIGN":
        shap_features = get_shap_explanation(model_name, X, pred_idx)

    severity = get_severity(confidence) if pred_label != "BENIGN" else None

    return {
        "pred_label":   pred_label,
        "pred_idx":     pred_idx,
        "confidence":   round(confidence, 4),
        "severity":     severity,
        "shap_features": shap_features,
        "is_attack":    pred_label != "BENIGN",
    }

# ── Schemas ───────────────────────────────────────────────────────────────────

class FlowRecord(BaseModel):
    features: Dict[str, Any]

class BatchFlowRecord(BaseModel):
    flows: List[Dict[str, Any]]

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service":  "NIDS API",
        "version":  "1.0.0",
        "status":   "running",
        "docs":     "/docs",
        "endpoints": ["/predict", "/predict/batch", "/alerts", "/stats", "/health"],
    }


@app.get("/health")
def health():
    return {
        "status":        "healthy",
        "models_loaded": list(models.keys()),
        "total_alerts":  len(alert_store),
        "timestamp":     datetime.utcnow().isoformat(),
    }


@app.post("/predict")
@limiter.limit("100/minute")
def predict(
    request:    Request,
    body:       FlowRecord,
    model_name: str = "xgboost",
    x_api_key:  Optional[str] = Header(None),
):
    verify_api_key(x_api_key)

    if model_name not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model_name}'. Available: {list(models.keys())}"
        )

    try:
        result    = predict_flow(body.features, model_name)
        alert_id  = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        if result["is_attack"]:
            alert = {
                "alert_id":         alert_id,
                "timestamp":        timestamp,
                "model_used":       model_name,
                "attack_type":      result["pred_label"],
                "severity":         result["severity"],
                "confidence":       result["confidence"],
                "shap_explanation": result["shap_features"],
                "raw_flow":         body.features,
            }
            alert_store.append(alert)
            stats_store[result["pred_label"]] += 1
            logger.info(f"ALERT | {result['severity']} | {result['pred_label']} | conf={result['confidence']}")

        return {
            "alert_id":         alert_id,
            "prediction":       result["pred_label"],
            "is_attack":        result["is_attack"],
            "confidence":       result["confidence"],
            "severity":         result["severity"],
            "shap_explanation": result["shap_features"],
            "model_used":       model_name,
            "timestamp":        timestamp,
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
@limiter.limit("10/minute")
def predict_batch(
    request:    Request,
    body:       BatchFlowRecord,
    model_name: str = "xgboost",
    x_api_key:  Optional[str] = Header(None),
):
    verify_api_key(x_api_key)

    if model_name not in models:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model_name}'")

    if len(body.flows) > 1000:
        raise HTTPException(status_code=400, detail="Max 1000 flows per batch request")

    results   = []
    timestamp = datetime.utcnow().isoformat()

    for flow in body.flows:
        try:
            result = predict_flow(flow, model_name)

            if result["is_attack"]:
                alert = {
                    "alert_id":         str(uuid.uuid4()),
                    "timestamp":        timestamp,
                    "model_used":       model_name,
                    "attack_type":      result["pred_label"],
                    "severity":         result["severity"],
                    "confidence":       result["confidence"],
                    "shap_explanation": result["shap_features"],
                    "raw_flow":         flow,
                }
                alert_store.append(alert)
                stats_store[result["pred_label"]] += 1

            results.append({
                "prediction":       result["pred_label"],
                "is_attack":        result["is_attack"],
                "confidence":       result["confidence"],
                "severity":         result["severity"],
                "shap_explanation": result["shap_features"],
            })

        except Exception as e:
            results.append({"error": str(e)})

    return {
        "total":     len(results),
        "results":   results,
        "model":     model_name,
        "timestamp": timestamp,
    }


@app.get("/alerts")
@limiter.limit("60/minute")
def get_alerts(
    request:   Request,
    severity:  Optional[str] = None,
    limit:     int = 50,
    x_api_key: Optional[str] = Header(None),
):
    verify_api_key(x_api_key)

    alerts = alert_store.copy()

    if severity:
        severity = severity.upper()
        valid    = [s for s, _ in SEVERITY_THRESHOLDS]
        if severity not in valid:
            raise HTTPException(status_code=400, detail=f"Invalid severity. Choose: {valid}")
        alerts = [a for a in alerts if a["severity"] == severity]

    alerts = sorted(alerts, key=lambda x: x["timestamp"], reverse=True)
    alerts = alerts[:limit]

    return {
        "total":    len(alert_store),
        "returned": len(alerts),
        "alerts":   alerts,
    }


@app.get("/stats")
@limiter.limit("60/minute")
def get_stats(
    request:   Request,
    x_api_key: Optional[str] = Header(None),
):
    verify_api_key(x_api_key)

    severity_dist: Dict[str, int] = defaultdict(int)
    for alert in alert_store:
        severity_dist[alert["severity"]] += 1

    return {
        "total_alerts":        len(alert_store),
        "attack_distribution": dict(stats_store),
        "severity_breakdown":  dict(severity_dist),
        "models_available":    list(models.keys()),
        "classes":             list(label_encoder.classes_),
        "timestamp":           datetime.utcnow().isoformat(),
    }


@app.delete("/alerts/clear")
def clear_alerts(x_api_key: Optional[str] = Header(None)):
    """Dev only — clears in-memory alert store."""
    verify_api_key(x_api_key)
    alert_store.clear()
    stats_store.clear()
    logger.info("Alert store cleared")
    return {"message": "Alert store cleared", "timestamp": datetime.utcnow().isoformat()}
