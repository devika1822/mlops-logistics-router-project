from pathlib import Path
import os

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# File paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "geography"
)


# Model configuration
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

RANDOM_STATE = 42
CV_FOLDS = 5


# Configure MLflow
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


# Load Spark processed dataset
df = pd.read_parquet(PROCESSED_DATA_PATH)

print("Processed Geography dataset loaded successfully.")
print(f"Dataset shape: {df.shape}")


# Check that all required columns are available
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


# Prepare features and target
X = df[FEATURES].copy()
y = df[TARGET].copy()

print(f"Training rows available: {len(X)}")
print(f"Feature count: {len(FEATURES)}")


# Selected final model
final_model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ]
)


# Evaluate the selected model using 5-fold cross-validation
cv = KFold(
    n_splits=CV_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

cv_scores = cross_validate(
    final_model,
    X,
    y,
    cv=cv,
    scoring={
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    },
)

cv_mae = -cv_scores["test_mae"].mean()
cv_rmse = -cv_scores["test_rmse"].mean()
cv_r2 = cv_scores["test_r2"].mean()

print("\n5-fold cross-validation completed.")
print(f"Mean MAE  : {cv_mae:.4f}")
print(f"Mean RMSE : {cv_rmse:.4f}")
print(f"Mean R2   : {cv_r2:.4f}")


# Train the final model using the complete processed dataset
final_model.fit(X, y)

print(
    f"\nFinal model fitted on all {len(X)} "
    "processed Geography records."
)


# Log final model and evaluation results to MLflow
with mlflow.start_run(run_name=RUN_NAME):

    mlflow.log_param("model_name", "linear_regression")
    mlflow.log_param("feature_set", "engineered")
    mlflow.log_param("feature_count", len(FEATURES))
    mlflow.log_param("features", ",".join(FEATURES))
    mlflow.log_param("average_speed_excluded", True)
    mlflow.log_param("training_rows", len(X))
    mlflow.log_param("cv_folds", CV_FOLDS)
    mlflow.log_param("random_state", RANDOM_STATE)

    mlflow.log_metric("cv_mae", cv_mae)
    mlflow.log_metric("cv_rmse", cv_rmse)
    mlflow.log_metric("cv_r2", cv_r2)

    input_example = X.head(5)

    mlflow.sklearn.log_model(
        sk_model=final_model,
        artifact_path="model",
        input_example=input_example,
        registered_model_name="geography_route_time_model",
    )


print("\nFinal Geography model trained successfully.")

print("\nSelected model:")
print("Linear Regression with StandardScaler")

print("\nCross-validation performance:")
print(f"Mean MAE  : {cv_mae:.4f}")
print(f"Mean RMSE : {cv_rmse:.4f}")
print(f"Mean R2   : {cv_r2:.4f}")