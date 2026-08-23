import os
import time
import pickle
import logging
from typing import Dict, Any
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field

# 1. Setup Structured Production Logging (Mandatory Monitoring Requirement)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] Latency: %(process)dms | %(message)s",
    handlers=[
        logging.FileHandler("api_production_monitoring.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TelemetryAPI")

app = FastAPI(
    title="E-Commerce Logistics Telemetry API",
    description="MLOps Track 2 Production Inference Server predicting vehicle breakdown risk levels.",
    version="1.0.0"
)

# 2. Dynamic Model Loading Mechanism
MODEL_PATH = os.path.join("models", "truck_telemetry_model_all_features.pkl")

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Successfully loaded champion model artifact from {MODEL_PATH}")
else:
    model = None
    logger.warning(f"Warning: Model artifact not found at {MODEL_PATH}. Standby mode activated.")

# 3. Pydantic Input Data Contract (Mapped directly to your route dataset dimensions)
class VehicleTelemetryInput(BaseModel):
    order_latitude: float = Field(..., example=12.9498)
    order_longitude: float = Field(..., example=77.4740)
    distance_km: float = Field(..., example=13.82)
    delivery_time_window_hrs: float = Field(..., example=16.47)
    order_priority: int = Field(..., example=2)
    vehicle_capacity_kg: float = Field(..., example=500.0)
    order_weight_kg: float = Field(..., example=42.94)
    vehicle_utilization_ratio: float = Field(..., example=0.538)
    traffic_density_index: float = Field(..., example=0.91)
    average_speed_kmph: float = Field(..., example=27.91)
    weather_impact_index: float = Field(..., example=0.35)
    time_of_day: int = Field(..., example=12)
    fuel_cost_per_km: float = Field(..., example=6.16)
    driver_cost_per_hour: float = Field(..., example=279.70)
    optimized_route_time_min: float = Field(..., example=43.23)
    optimized_route_cost: float = Field(..., example=286.70)
    delivery_efficiency_score: float = Field(..., example=0.045)
    route_reliability_index: float = Field(..., example=0.529)
    speed_strain_factor: float = Field(..., example=0.30)

# 4. Latency Monitoring Middleware
@app.middleware("http")
async def monitor_api_latency(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time_ms = int((time.time() - start_time) * 1000)
    
    # Injects operational latency telemetry directly into header arrays
    response.headers["X-Response-Time-Ms"] = str(process_time_ms)
    
    # Log details if hitting prediction endpoints
    if request.url.path == "/predict":
        logger.info(f"Path: {request.url.path} | Status: {response.status_code}")
        
    return response

# 5. Core Health Endpoint
@app.get("/health")
def health_check():
    if model is None:
        return {"status": "unhealthy", "message": "Model binary missing out of workspace folder nodes."}
    return {"status": "healthy", "model_loaded": True}

# 6. Core Prediction Endpoint
@app.post("/predict")
def predict_breakdown_risk(payload: VehicleTelemetryInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Prediction engine is offline. Model not available.")
    
    try:
        # Convert incoming contract layers directly into Pandas payload arrays
        input_data = pd.DataFrame([payload.model_dump()])
        
        # Enforce column sorting match rules from model training criteria
        model_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else input_data.columns
        input_data = input_data[model_features]
        
        # Execute validation prediction
        prediction = model.predict(input_data)[0]
        
        # Structure Day 1 contract-compliant response mapping
        response_data = {
            "track": "vehicle_and_telemetry",
            "prediction": str(prediction),
            "status": "success"
        }
        
        return response_data

    except Exception as e:
        logger.error(f"Inference failure encountered: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal system scoring fault: {str(e)}")
