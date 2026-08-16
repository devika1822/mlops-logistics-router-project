from pathlib import Path
import os

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


#paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "geography"
)


# Final model configuration

TARGET = "optimized_route_time_min"

FEATURES = [
    "order_latitude",
    "order_longitude",
    "distance_km",
    "delivery_time_window_hrs",
    "order_priority",
    "traffic_density_index",
    "distance_traffic_interaction",
    "distance_per_delivery_window",
    "distance_outlier_flag",
]

EXPERIMENT_NAME = "geography_track"
RUN_NAME = "geography_final_linear_regression"

# Cross-validation results from model_tuning.py
CV_MAE = 33.233676
CV_RMSE = 48.320444
CV_R2 = 0.456551


# MLflow configuration

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


# Load processed data

df = pd.read_parquet(PROCESSED_DATA_PATH)

print("Processed Geography dataset loaded successfully.")
print(f"Dataset shape: {df.shape}")


# Validate columns

required_columns = FEATURES + [TARGET]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# Prepare full training data

X = df[FEATURES].copy()
y = df[TARGET].copy()

print(f"Final training rows: {len(X)}")
print(f"Feature count: {len(FEATURES)}")



# Final selected model

final_model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ]
)



# Train and log final model

with mlflow.start_run(run_name=RUN_NAME):

    final_model.fit(X, y)

    mlflow.log_param(
        "model_name",
        "linear_regression",
    )

    mlflow.log_param(
        "feature_set",
        "engineered",
    )

    mlflow.log_param(
        "feature_count",
        len(FEATURES),
    )

    mlflow.log_param(
        "features",
        ",".join(FEATURES),
    )

    mlflow.log_param(
        "average_speed_excluded",
        True,
    )

    mlflow.log_param(
        "training_rows",
        len(X),
    )

    # These are cross-validation metrics from model selection,
    # not metrics calculated on the full training dataset.
    mlflow.log_metric(
        "cv_mae",
        CV_MAE,
    )

    mlflow.log_metric(
        "cv_rmse",
        CV_RMSE,
    )

    mlflow.log_metric(
        "cv_r2",
        CV_R2,
    )

    input_example = X.head(5)

    mlflow.sklearn.log_model(
        sk_model=final_model,
        artifact_path="model",
        input_example=input_example,
    )


print("\nFinal Geography model trained successfully.")

print("\nSelected model:")
print("Linear Regression with StandardScaler")

print("\nCross-validation performance:")
print(f"Mean MAE  : {CV_MAE:.4f}")
print(f"Mean RMSE : {CV_RMSE:.4f}")
print(f"Mean R2   : {CV_R2:.4f}")

print(
    "\nFinal model fitted on all "
    f"{len(X)} processed Geography records."
)