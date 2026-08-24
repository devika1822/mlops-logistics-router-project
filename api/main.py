import os

import mlflow
import mlflow.pyfunc
from fastapi import FastAPI

from api.routers import conditions


MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)

MODEL_URI = "models:/conditions-route-reliability@champion"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = FastAPI(
    title="Conditions Route Reliability API",
    version="1.0",
)


@app.on_event("startup")
def load_conditions_model():
    model = mlflow.pyfunc.load_model(MODEL_URI)

    client = mlflow.MlflowClient(
        tracking_uri=MLFLOW_TRACKING_URI
    )

    version = client.get_model_version_by_alias(
        "conditions-route-reliability",
        "champion",
    )

    conditions.set_model(
        loaded_model=model,
        version=str(version.version),
    )


@app.get("/")
def health():
    return {
        "status": "healthy",
        "conditions_model_loaded":
            conditions.is_model_loaded(),
        "model_uri": MODEL_URI,
    }


app.include_router(conditions.router)