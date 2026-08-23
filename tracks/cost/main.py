"""
Cargo Payload Track – FastAPI Application (Docker entry point)

Exposes:
  GET  /health          – liveness probe
  POST /api/v1/payload  – predict engine wear from payload metrics
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import os, sys, logging, pathlib
import pandas as pd

# Ensure predict.py is importable regardless of working directory
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

# Resolve live log path relative to repo root regardless of cwd
_REPO_ROOT  = pathlib.Path(__file__).resolve().parents[2]
_LIVE_PATH  = _REPO_ROOT / "data" / "processed" / "payload_live.parquet"


def _log_request(row: dict) -> None:
    """Append one prediction row to the live parquet file."""
    df_new = pd.DataFrame([row])
    try:
        if _LIVE_PATH.exists():
            df_new = pd.concat([pd.read_parquet(_LIVE_PATH), df_new], ignore_index=True)
        df_new.to_parquet(_LIVE_PATH, index=False)
    except Exception as exc:
        logging.warning(f"Live logging failed: {exc}")


# Lazy import so the container starts even if the model isn't staged yet
try:
    from predict import predict as _predict
    _model_ready = True
except Exception as exc:
    logging.warning(f"Model not loaded at startup: {exc}")
    _model_ready = False

app = FastAPI(
    title="Cargo Payload API",
    description=(
        "Predicts engine_wear_score (0–100) based on cargo payload metrics. "
        "Scores above 75 indicate critical wear risk."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class PayloadRequest(BaseModel):
    cargo_weight_kg: float = Field(
        ..., gt=0, description="Cargo weight in kilograms (order_weight_kg)"
    )
    vehicle_capacity_pct: float = Field(
        ..., ge=0, le=150,
        description="Payload as % of vehicle capacity (vehicle_utilization_ratio × 100)"
    )
    delivery_deadline_hrs: float = Field(
        ..., ge=0, description="Hours available for delivery (delivery_time_window_hrs)"
    )
    trip_distance_km: float = Field(
        ..., gt=0, description="Trip distance in kilometres (distance_km)"
    )
    traffic_density_index: float = Field(
        ..., ge=0, le=1, description="Traffic density index 0–1"
    )
    weather_impact_index: float = Field(
        ..., ge=0, le=1, description="Weather impact index 0–1"
    )


class PayloadResponse(BaseModel):
    engine_wear_score: float
    overloaded: bool
    risk_level: str          # LOW / MEDIUM / HIGH / CRITICAL
    warning: Optional[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _risk_level(score: float) -> str:
    if score < 30:
        return "LOW"
    elif score < 55:
        return "MEDIUM"
    elif score < 75:
        return "HIGH"
    return "CRITICAL"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", tags=["ops"])
def health():
    return {"status": "ok", "track": "cargo_payload", "model_ready": _model_ready}


@app.post("/api/v1/payload", response_model=PayloadResponse, tags=["payload"])
def predict_engine_wear(req: PayloadRequest):
    if not _model_ready:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Run train.py and promote to 'Production'.",
        )

    overloaded_flag = 1 if req.vehicle_capacity_pct > 100 else 0

    score = _predict(
        cargo_weight_kg=req.cargo_weight_kg,
        vehicle_capacity_pct=req.vehicle_capacity_pct,
        delivery_deadline_hrs=req.delivery_deadline_hrs,
        trip_distance_km=req.trip_distance_km,
        traffic_density_index=req.traffic_density_index,
        weather_impact_index=req.weather_impact_index,
        overloaded_flag=overloaded_flag,
    )

    warning = None
    if overloaded_flag:
        warning = (
            f"OVERLOADED: {req.vehicle_capacity_pct:.1f}% of capacity exceeds the "
            "100% rated limit. Reduce cargo to prevent structural damage."
        )
    elif score > 75:
        warning = (
            f"CRITICAL WEAR RISK: Predicted score {score:.1f}/100. "
            "Schedule maintenance before this trip."
        )

    _log_request({
        "cargo_weight_kg":       req.cargo_weight_kg,
        "vehicle_capacity_pct":  req.vehicle_capacity_pct,
        "delivery_deadline_hrs": req.delivery_deadline_hrs,
        "trip_distance_km":      req.trip_distance_km,
        "traffic_density_index": req.traffic_density_index,
        "weather_impact_index":  req.weather_impact_index,
        "overloaded_flag":       overloaded_flag,
        "engine_wear_score":     round(score, 2),
    })

    return PayloadResponse(
        engine_wear_score=round(score, 2),
        overloaded=bool(overloaded_flag),
        risk_level=_risk_level(score),
        warning=warning,
    )
