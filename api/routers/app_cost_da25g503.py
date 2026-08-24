import os
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Body
import pandas as pd
import mlflow
import mlflow.sklearn


app = FastAPI(
    title="Truck Route Cost Prediction API",
    description="Serves predictions from MLflow Model Registry",
    version="1.0",
)


MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_URI = "models:/cost_route_cost_model/latest"


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTION_LOG_PATH = (
    PROJECT_ROOT
    / "monitoring"
    / "cost"
    / "prediction_logs.csv"
)


def load_model():
    try:
        return mlflow.sklearn.load_model(MODEL_URI)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


model = load_model()


@app.get("/")
def read_root():
    return {
        "status": "Active",
        "model_loaded": model is not None,
        "model_uri": MODEL_URI,
    }


@app.post("/predict")
def predict_cost(payload: dict = Body(...)):
    global model

    if model is None:
        model = load_model()

        if model is None:
            raise HTTPException(
                status_code=500,
                detail="Model not found in MLflow Registry.",
            )

    try:
        input_df = pd.DataFrame([payload])

        if hasattr(model, "feature_names_in_"):
            for col in model.feature_names_in_:
                if col not in input_df.columns:
                    input_df[col] = 0.0

            input_df = input_df[model.feature_names_in_]

        prediction = model.predict(input_df)

        predicted_cost = float(prediction[0])

        # --------------------------------------------------------------
        # Prediction logging for Evidently monitoring
        # --------------------------------------------------------------
        log_record = input_df.copy()

        log_record["predicted_optimized_route_cost"] = predicted_cost
        log_record["prediction_timestamp"] = (
            datetime.now().isoformat()
        )

        PREDICTION_LOG_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        log_record.to_csv(
            PREDICTION_LOG_PATH,
            mode="a",
            header=not PREDICTION_LOG_PATH.exists(),
            index=False,
        )

        return {
            "status": "success",
            "predicted_optimized_route_cost": predicted_cost,
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction error: {str(e)}",
        )