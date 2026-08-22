"""
Cargo Payload Track – Model Training Script

Algorithm comparison:
  • RandomForestRegressor  (sklearn)
  • XGBRegressor           (xgboost)

Both are trained on cleaned payload data to predict engine_wear_score.
All metrics are logged to MLflow. The model with the lower MAE is
registered as 'payload_engine_wear_model' in the MLflow Model Registry.

Usage:
    python train.py
    MLFLOW_TRACKING_URI=http://localhost:5000 python train.py
"""

import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from xgboost import XGBRegressor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROCESSED_PATH  = os.getenv("PAYLOAD_DATA_PATH", "data/processed/payload_clean.parquet")
MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "cargo_payload_track"
REGISTERED_NAME = "payload_engine_wear_model"

FEATURES = [
    "cargo_weight_kg",        # standardised from order_weight_kg
    "vehicle_capacity_pct",   # derived from vehicle_utilization_ratio * 100
    "delivery_deadline_hrs",  # from delivery_time_window_hrs
    "trip_distance_km",       # from distance_km
    "traffic_density_index",  # kept as additional stress feature
    "weather_impact_index",   # kept as additional stress feature
    "overloaded_flag",        # derived: 1 if capacity_pct > 100
]
TARGET = "engine_wear_score"  # derived in Spark from stress indicators


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data():
    df = pd.read_parquet(PROCESSED_PATH)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    X = df[FEATURES].fillna(0).astype(float)
    y = df[TARGET].astype(float)
    return train_test_split(X, y, test_size=0.2, random_state=42)


# ---------------------------------------------------------------------------
# Training + MLflow logging
# ---------------------------------------------------------------------------
def train_and_log(model, model_name: str, X_train, X_test, y_train, y_test) -> tuple:
    with mlflow.start_run(run_name=model_name) as run:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae  = mean_absolute_error(y_test, preds)
        rmse = root_mean_squared_error(y_test, preds)
        r2   = r2_score(y_test, preds)

        mlflow.log_params(model.get_params())
        mlflow.log_metric("mae",  mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2",   r2)
        mlflow.log_param("features", FEATURES)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=X_test.iloc[:1],
        )

        print(f"[{model_name}]  MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")
        return mae, run.info.run_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("[TRAIN] Loading data …")
    X_train, X_test, y_train, y_test = load_data()
    print(f"[TRAIN] Train={len(X_train)}  Test={len(X_test)}")

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )
    xgb = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )

    rf_mae,  rf_run_id  = train_and_log(rf,  "RandomForest_Payload", X_train, X_test, y_train, y_test)
    xgb_mae, xgb_run_id = train_and_log(xgb, "XGBoost_Payload",     X_train, X_test, y_train, y_test)

    # Register the model with lower MAE
    if rf_mae <= xgb_mae:
        best_run_id, best_name = rf_run_id,  "RandomForest_Payload"
    else:
        best_run_id, best_name = xgb_run_id, "XGBoost_Payload"

    model_uri = f"runs:/{best_run_id}/model"
    mlflow.register_model(model_uri, REGISTERED_NAME)

    print(f"\n[TRAIN] Best model: {best_name}  (MAE={min(rf_mae, xgb_mae):.3f})")
    print(f"[TRAIN] Registered as '{REGISTERED_NAME}' in MLflow Model Registry.")


if __name__ == "__main__":
    main()
