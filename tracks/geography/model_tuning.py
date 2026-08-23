from pathlib import Path
import os

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "geography"
)


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

RANDOM_STATE = 42
CV_FOLDS = 5

EXPERIMENT_NAME = "geography_track"

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


cv = KFold(
    n_splits=CV_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE,
)


df = pd.read_parquet(PROCESSED_DATA_PATH)

X = df[FEATURES].copy()
y = df[TARGET].copy()

print("Processed Geography dataset loaded successfully.")
print(f"Dataset shape: {df.shape}")


# Linear Regression candidate
linear_model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ]
)

linear_scores = cross_validate(
    linear_model,
    X,
    y,
    cv=cv,
    scoring={
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    },
)

linear_mae = -linear_scores["test_mae"].mean()
linear_rmse = -linear_scores["test_rmse"].mean()
linear_r2 = linear_scores["test_r2"].mean()

print("\nLINEAR REGRESSION - 5 FOLD CROSS VALIDATION")
print(f"Mean MAE  : {linear_mae:.4f}")
print(f"Mean RMSE : {linear_rmse:.4f}")
print(f"Mean R2   : {linear_r2:.4f}")


# Log Linear Regression candidate to MLflow
linear_model.fit(X, y)

with mlflow.start_run(
    run_name="geography_candidate_linear_regression"
):

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
        "cv_folds",
        CV_FOLDS,
    )

    mlflow.log_param(
        "random_state",
        RANDOM_STATE,
    )

    mlflow.log_metric(
        "cv_mae",
        linear_mae,
    )

    mlflow.log_metric(
        "cv_rmse",
        linear_rmse,
    )

    mlflow.log_metric(
        "cv_r2",
        linear_r2,
    )

    mlflow.sklearn.log_model(
        sk_model=linear_model,
        artifact_path="model",
        input_example=X.head(5),
    )


# Random Forest tuning
random_forest = RandomForestRegressor(
    random_state=RANDOM_STATE,
)

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2, 4],
}

grid_search = GridSearchCV(
    estimator=random_forest,
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",
    cv=cv,
    n_jobs=-1,
    verbose=1,
)

print("\nRANDOM FOREST GRID SEARCH")

grid_search.fit(X, y)

best_rf = grid_search.best_estimator_

print("\nBest Random Forest parameters:")
print(grid_search.best_params_)

print(
    f"\nBest CV RMSE: "
    f"{-grid_search.best_score_:.4f}"
)


# Evaluate tuned Random Forest
rf_scores = cross_validate(
    best_rf,
    X,
    y,
    cv=cv,
    scoring={
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    },
)

rf_mae = -rf_scores["test_mae"].mean()
rf_rmse = -rf_scores["test_rmse"].mean()
rf_r2 = rf_scores["test_r2"].mean()

print("\nTUNED RANDOM FOREST - 5 FOLD CROSS VALIDATION")
print(f"Mean MAE  : {rf_mae:.4f}")
print(f"Mean RMSE : {rf_rmse:.4f}")
print(f"Mean R2   : {rf_r2:.4f}")


# Log tuned Random Forest candidate to MLflow
best_rf.fit(X, y)

with mlflow.start_run(
    run_name="geography_candidate_random_forest"
):

    mlflow.log_param(
        "model_name",
        "random_forest",
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
        "cv_folds",
        CV_FOLDS,
    )

    mlflow.log_param(
        "random_state",
        RANDOM_STATE,
    )

    mlflow.log_param(
        "n_estimators",
        grid_search.best_params_["n_estimators"],
    )

    mlflow.log_param(
        "max_depth",
        grid_search.best_params_["max_depth"],
    )

    mlflow.log_param(
        "min_samples_split",
        grid_search.best_params_["min_samples_split"],
    )

    mlflow.log_param(
        "min_samples_leaf",
        grid_search.best_params_["min_samples_leaf"],
    )

    mlflow.log_metric(
        "cv_mae",
        rf_mae,
    )

    mlflow.log_metric(
        "cv_rmse",
        rf_rmse,
    )

    mlflow.log_metric(
        "cv_r2",
        rf_r2,
    )

    mlflow.sklearn.log_model(
        sk_model=best_rf,
        artifact_path="model",
        input_example=X.head(5),
    )


# Final comparison
results = pd.DataFrame(
    [
        {
            "Model": "Linear Regression",
            "Mean MAE": linear_mae,
            "Mean RMSE": linear_rmse,
            "Mean R2": linear_r2,
        },
        {
            "Model": "Tuned Random Forest",
            "Mean MAE": rf_mae,
            "Mean RMSE": rf_rmse,
            "Mean R2": rf_r2,
        },
    ]
)

results = results.sort_values(
    by="Mean RMSE"
).reset_index(drop=True)

print("\nFINAL CROSS-VALIDATION COMPARISON")

print(
    results.to_string(index=False)
)

print(
    f"\nSelected candidate: "
    f"{results.iloc[0]['Model']}"
)