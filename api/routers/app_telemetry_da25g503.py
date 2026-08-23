import os
import time
import pickle
import logging
import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] Latency: %(process)dms | %(message)s",
    handlers=[logging.FileHandler("api_production_monitoring.log"), logging.StreamHandler()]
)
logger = logging.getLogger("TelemetryAPI")

app = FastAPI(title="Truck Telemetry Efficiency Regression Service", version="1.2")

MODEL_PATH = "models/truck_telemetry_model_all_features.pkl"
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    MODEL_FEATURES = list(model.feature_names_in_)
    logger.info("Successfully loaded champion regression model artifact.")
else:
    model = None
    MODEL_FEATURES = []
    logger.warning("Warning: Model binary missing. Core endpoints are offline.")

class TelemetryPayload(BaseModel):
    order_latitude: float = 12.9498
    order_longitude: float = 77.4740
    distance_km: float = 13.82
    delivery_time_window_hrs: float = 16.47
    order_priority: int = 2
    vehicle_capacity_kg: float = 500.0
    order_weight_kg: float = 42.94
    vehicle_utilization_ratio: float = 0.538
    traffic_density_index: float = 0.91
    average_speed_kmph: float = 27.91
    weather_impact_index: float = 0.35
    time_of_day: int = 12
    fuel_cost_per_km: float = 6.16
    driver_cost_per_hour: float = 279.70
    route_reliability_index: float = 0.529

@app.middleware("http")
async def monitor_api_latency(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    latency_ms = int((time.time() - start_time) * 1000)
    response.headers["X-Response-Time-Ms"] = str(latency_ms)
    return response

@app.get("/")
def health_check():
    return {"status": "healthy" if model else "unhealthy", "expected_features": MODEL_FEATURES}

@app.post("/predict")
def predict_efficiency(payload: TelemetryPayload):
    if not model:
        raise HTTPException(status_code=503, detail="Inference cluster offline. Model file not generated.")
    try:
        input_df = pd.DataFrame([payload.model_dump()])
        input_df = input_df[MODEL_FEATURES]  # Enforce strict feature array alignment
        
        prediction = model.predict(input_df)
        
        return {
            "predicted_delivery_efficiency_score": float(prediction[0]),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Inference Failure: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
