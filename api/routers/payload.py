"""
Cargo Payload Track – Shared FastAPI Router
Included by the project's unified API server (api/main.py).
"""

from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel, Field
import logging

app = FastAPI(
    title="Payload Engine Wear Prediction API",
    description="Serves Payload/Vehicle engine wear predictions from MLflow",
    version="1.0",
)

try:
    from tracks.vehicle.predict import predict as _predict
    _model_ready = True
except Exception as exc:
    logging.warning(f"Payload model not loaded: {exc}")
    _model_ready = False

router = APIRouter(prefix="/api/v1", tags=["payload"])


class PayloadRequest(BaseModel):
    cargo_weight_kg: float = Field(..., gt=0)
    vehicle_capacity_pct: float = Field(..., ge=0, le=150)
    delivery_deadline_hrs: float = Field(..., ge=0)
    trip_distance_km: float = Field(..., gt=0)
    traffic_density_index: float = Field(..., ge=0, le=1)
    weather_impact_index: float = Field(..., ge=0, le=1)


class PayloadResponse(BaseModel):
    engine_wear_score: float
    overloaded: bool
    risk_level: str
    warning: str | None


def _risk_level(score: float) -> str:
    if score < 30:   return "LOW"
    if score < 55:   return "MEDIUM"
    if score < 75:   return "HIGH"
    return "CRITICAL"


@router.post("/payload", response_model=PayloadResponse)
def predict_engine_wear(req: PayloadRequest):
    if not _model_ready:
        raise HTTPException(status_code=503, detail="Payload model not available.")

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
        warning = f"OVERLOADED: {req.vehicle_capacity_pct:.1f}% exceeds 100% rated capacity."
    elif score > 75:
        warning = f"CRITICAL WEAR RISK: Score {score:.1f}/100. Schedule maintenance."

    return PayloadResponse(
        engine_wear_score=round(score, 2),
        overloaded=bool(overloaded_flag),
        risk_level=_risk_level(score),
        warning=warning,
    )


app.include_router(router)