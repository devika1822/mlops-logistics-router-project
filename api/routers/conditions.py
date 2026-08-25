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



model = None
model_version = "unknown"


def set_model(loaded_model, version):

    global model
    global model_version

    model = loaded_model
    model_version = str(version)


def is_model_loaded():
    return model is not None


def get_model_version():
    return model_version



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

    predicted_route_reliability_index: float = Field(
        ...,
        description="Predicted route reliability index",
    )

    model_name: str

    model_version: str



@router.post(
    "/predict",
    response_model=ConditionsOutput,
)
def predict_conditions(
    data: ConditionsInput,
):




    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Conditions model is not available.",
        )

    try:



        input_df = pd.DataFrame(
            [data.model_dump()]
        )[FEATURES]



        input_df = input_df.astype(
            {
                "weather_impact_index": "float64",
                "average_speed_kmph": "float64",
                "time_of_day": "float64",
                "traffic_density_index": "float64",
            }
        )



        prediction = model.predict(
            input_df
        )

        if prediction is None or len(prediction) == 0:
            raise ValueError(
                "Model returned an empty prediction."
            )

        # XGBoost regression returns a numeric value.
        predicted_index = float(
            prediction[0]
        )



        if not pd.notna(
            predicted_index
        ):
            raise ValueError(
                "Model returned an invalid "
                "route reliability index."
            )


        return ConditionsOutput(
            predicted_route_reliability_index=(
                predicted_index
            ),
            model_name=MODEL_NAME,
            model_version=str(
                model_version
            ),
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Prediction failed: {exc}"
            ),
        ) from exc