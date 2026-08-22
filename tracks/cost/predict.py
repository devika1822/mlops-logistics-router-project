"""
Cargo Payload Track – Inference Utilities

Loads the registered 'payload_engine_wear_model' from MLflow
and exposes a predict() function used by the FastAPI app.
"""

import os
import mlflow.pyfunc
import pandas as pd

MLFLOW_URI    = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME    = "payload_engine_wear_model"
MODEL_ALIAS   = os.getenv("MODEL_ALIAS", "champion")

# Module-level cache so the model is loaded only once per process
_model = None


def _load_model() -> mlflow.pyfunc.PyFuncModel:
    global _model
    if _model is None:
        mlflow.set_tracking_uri(MLFLOW_URI)
        _model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    return _model


def predict(
    cargo_weight_kg: float,
    vehicle_capacity_pct: float,
    delivery_deadline_hrs: float,
    trip_distance_km: float,
    traffic_density_index: float,
    weather_impact_index: float,
    overloaded_flag: int,
) -> float:
    """Return predicted engine_wear_score (0–100)."""
    model = _load_model()
    row = pd.DataFrame([{
        "cargo_weight_kg":        cargo_weight_kg,
        "vehicle_capacity_pct":   vehicle_capacity_pct,
        "delivery_deadline_hrs":  delivery_deadline_hrs,
        "trip_distance_km":       trip_distance_km,
        "traffic_density_index":  traffic_density_index,
        "weather_impact_index":   weather_impact_index,
        "overloaded_flag":        overloaded_flag,
    }])
    return float(model.predict(row)[0])
