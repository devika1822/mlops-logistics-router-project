import os
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI(
    title="Truck Telemetry Breakdown Risk Service",
    description="Production MLOps API with automated fallback handling for team integration.",
    version="1.1"
)

MODEL_PATH = os.path.join("models", "truck_telemetry_model_all_features.pkl")
if not os.path.exists(MODEL_PATH) and os.path.exists("truck_telemetry_model_all_features.pkl"):
    MODEL_PATH = "truck_telemetry_model_all_features.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

MODEL_FEATURES = list(model.feature_names_in_)

class TelemetryPayload(BaseModel):
    order_latitude: Optional[float] = 0.0
    order_longitude: Optional[float] = 0.0
    distance_km: Optional[float] = 0.0
    delivery_time_window_hrs: Optional[float] = 0.0
    order_priority: Optional[float] = 0.0
    vehicle_capacity_kg: Optional[float] = 0.0
    order_weight_kg: Optional[float] = 0.0
    vehicle_utilization_ratio: Optional[float] = 0.0
    traffic_density_index: Optional[float] = 0.0
    average_speed_kmph: Optional[float] = 40.0
    weather_impact_index: Optional[float] = 0.0
    time_of_day: Optional[float] = 12.0
    fuel_cost_per_km: Optional[float] = 0.0
    driver_cost_per_hour: Optional[float] = 0.0
    optimized_route_time_min: Optional[float] = 0.0
    optimized_route_cost: Optional[float] = 0.0
    delivery_efficiency_score: Optional[float] = 0.0
    route_reliability_index: Optional[float] = 1.0
    speed_strain_factor: Optional[float] = 0.0

@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": os.path.basename(MODEL_PATH),
        "expected_features": MODEL_FEATURES
    }

@app.post("/predict")
def predict_risk(payload: TelemetryPayload):
    try:
        input_dict = payload.model_dump()
        input_df = pd.DataFrame([input_dict])
        input_df = input_df[MODEL_FEATURES]
        
        prediction = model.predict(input_df)
        
        return {
            "prediction": str(prediction),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
