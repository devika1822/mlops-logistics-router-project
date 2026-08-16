from pathlib import Path
import os

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "geography"
)


# Configuration

TARGET = "optimized_route_time_min"

FEATURES = [
    "order_latitude",
    "order_longitude",
    "distance_km",
    "delivery_time_window_hrs",
    "order_priority",
    "traffic_density_index",

    # Engineered by Spark
    "distance_traffic_interaction",
    "distance_per_delivery_window",

    # Input-based outlier flag
    "distance_outlier_flag",
]

TEST_SIZE = 0.20
RANDOM_STATE = 42

EXPERIMENT_NAME = "geography_track"


# MLflow configuration

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


# Load Spark-processed data


df = pd.read_parquet(PROCESSED_DATA_PATH)

print("Processed Geography dataset loaded successfully.")
print(f"Dataset shape: {df.shape}")



# Validate model columns

required_columns = FEATURES + [TARGET]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required training columns: {missing_columns}"
    )


# Prepare train/test data

X = df[FEATURES].copy()
y = df[TARGET].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)


print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")


# Models

linear_regression = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ]
)

random_forest = RandomForestRegressor(
    n_estimators=200,
    random_state=RANDOM_STATE,
)


models = {
    "linear_regression": linear_regression,
    "random_forest": random_forest,
}


# Train, evaluate and log to MLflow

results = []

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print(f"Training: {model_name}")
    print("=" * 60)

    with mlflow.start_run(run_name=model_name):

        #train
        model.fit(X_train, y_train)

        #predict
        predictions = model.predict(X_test)

        
        mae = mean_absolute_error(
            y_test,
            predictions,
        )

        rmse = mean_squared_error(
            y_test,
            predictions,
        ) ** 0.5

        r2 = r2_score(
            y_test,
            predictions,
        )

        print(f"MAE  : {mae:.4f}")
        print(f"RMSE : {rmse:.4f}")
        print(f"R2   : {r2:.4f}")

        # Log common parameters
        mlflow.log_param(
            "model_name",
            model_name,
        )

        mlflow.log_param(
            "test_size",
            TEST_SIZE,
        )

        mlflow.log_param(
            "random_state",
            RANDOM_STATE,
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

        
        # Log model-specific parameters
        if model_name == "random_forest":

            mlflow.log_param(
                "n_estimators",
                200,
            )

       # Log metrics
        mlflow.log_metric(
            "mae",
            mae,
        )

        mlflow.log_metric(
            "rmse",
            rmse,
        )

        mlflow.log_metric(
            "r2",
            r2,
        )

        #log model
        input_example = X_train.head(5)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example,
        )

        results.append(
            {
                "Model": model_name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
            }
        )


#compare results
results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

print(
    results_df
    .sort_values(
        by="RMSE"
    )
    .to_string(index=False)
)