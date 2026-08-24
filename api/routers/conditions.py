from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import pandas as pd


router = APIRouter(
    prefix="/api/v1/conditions",
    tags=["Conditions"],
)


FEATURES = [
    "weather_impact_index",
    "average_speed_kmph",
    "time_of_day",
    "traffic_density_index",
]

MODEL_NAME = "conditions-route-reliability"

LABEL_MAPPING = {
    0: "Low",
    1: "Medium",
    2: "High",
}


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTION_LOG_PATH = (
    PROJECT_ROOT
    / "monitoring"
    / "conditions"
    / "prediction_logs.csv"
)


# Loaded by api/main.py during FastAPI startup.
model = None
model_version = "unknown"


def set_model(loaded_model, version):
    """Set the model loaded by the FastAPI application."""

    global model
    global model_version

    model = loaded_model
    model_version = version


def is_model_loaded():
    """Return whether the Conditions model is currently loaded."""

    return model is not None


class ConditionsInput(BaseModel):
    weather_impact_index: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Weather impact index",
    )

    average_speed_kmph: float = Field(
        ...,
        ge=0.0,
        le=150.0,
        description="Average vehicle speed in km/h",
    )

    time_of_day: int = Field(
        ...,
        ge=0,
        le=23,
        description="Hour of day, from 0 to 23",
    )

    traffic_density_index: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Traffic density index",
    )


class ConditionsOutput(BaseModel):
    predicted_reliability: str
    model_name: str
    model_version: str


@router.post(
    "/predict",
    response_model=ConditionsOutput,
)
def predict_conditions(data: ConditionsInput):
    """Predict route reliability from Conditions features."""

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Conditions model is not available.",
        )

    try:
        # Convert request to DataFrame and enforce
        # exactly the same column order as training.
        input_df = pd.DataFrame(
            [data.model_dump()]
        )[FEATURES]

        input_df["weather_impact_index"] = (
            input_df["weather_impact_index"].astype("float64")
        )

        input_df["average_speed_kmph"] = (
            input_df["average_speed_kmph"].astype("float64")
        )

        input_df["time_of_day"] = (
            input_df["time_of_day"].astype("int32")
        )

        input_df["traffic_density_index"] = (
            input_df["traffic_density_index"].astype("float64")
        )

        prediction = model.predict(
            input_df
        )

        predicted_class = int(
            prediction[0]
        )

        label = LABEL_MAPPING.get(
            predicted_class
        )

        if label is None:
            raise ValueError(
                f"Unexpected prediction class: "
                f"{predicted_class}"
            )

        # ------------------------------------------------------------------
        # Prediction logging for Evidently monitoring
        # ------------------------------------------------------------------
        log_record = input_df.copy()

        log_record["predicted_reliability"] = label
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

        return ConditionsOutput(
            predicted_reliability=label,
            model_name=MODEL_NAME,
            model_version=model_version,
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc