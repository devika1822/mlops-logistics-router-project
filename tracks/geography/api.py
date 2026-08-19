import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)

MODEL_NAME = "geography_route_time_model"
MODEL_URI = f"models:/{MODEL_NAME}/latest"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = FastAPI(
    title="Geography Route Time API",
    version="1.0",
)


class PredictionInput(BaseModel):
    order_latitude: float
    order_longitude: float
    distance_km: float
    delivery_time_window_hrs: float
    order_priority: int
    traffic_density_index: float


model = mlflow.sklearn.load_model(MODEL_URI)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "geography"
)

training_df = pd.read_parquet(PROCESSED_DATA_PATH)

q1 = training_df["distance_km"].quantile(0.25)
q3 = training_df["distance_km"].quantile(0.75)

iqr = q3 - q1

DISTANCE_LOWER = q1 - (1.5 * iqr)
DISTANCE_UPPER = q3 + (1.5 * iqr)


@app.get("/")
def root():
    return {
        "message": "Geography Route Time API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": MODEL_NAME,
    }


@app.post("/predict")
def predict(data: PredictionInput):

    distance_traffic_interaction = (
        data.distance_km
        * data.traffic_density_index
    )

    distance_per_delivery_window = (
        data.distance_km
        / data.delivery_time_window_hrs
    )

    distance_outlier_flag = int(
        data.distance_km < DISTANCE_LOWER
        or data.distance_km > DISTANCE_UPPER
    )

    input_df = pd.DataFrame(
        [
            {
                "order_latitude": data.order_latitude,
                "order_longitude": data.order_longitude,
                "distance_km": data.distance_km,
                "delivery_time_window_hrs":
                    data.delivery_time_window_hrs,
                "order_priority": data.order_priority,
                "traffic_density_index":
                    data.traffic_density_index,
                "distance_traffic_interaction":
                    distance_traffic_interaction,
                "distance_per_delivery_window":
                    distance_per_delivery_window,
                "distance_outlier_flag":
                    distance_outlier_flag,
            }
        ]
    )

    prediction = model.predict(input_df)[0]

    return {
        "predicted_route_time_min":
            round(float(prediction), 2)
    }