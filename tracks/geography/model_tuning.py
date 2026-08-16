from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "geography"
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

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

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)


# ---------------------------------------------------------
# Load processed data
# ---------------------------------------------------------

df = pd.read_parquet(PROCESSED_DATA_PATH)

X = df[FEATURES].copy()
y = df[TARGET].copy()

print("Processed Geography dataset loaded successfully.")
print(f"Dataset shape: {df.shape}")


# ---------------------------------------------------------
# Linear Regression cross-validation
# ---------------------------------------------------------

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

print("\n" + "=" * 65)
print("LINEAR REGRESSION - 5 FOLD CROSS VALIDATION")
print("=" * 65)

print(f"Mean MAE  : {linear_mae:.4f}")
print(f"Mean RMSE : {linear_rmse:.4f}")
print(f"Mean R2   : {linear_r2:.4f}")


# ---------------------------------------------------------
# Random Forest tuning
# ---------------------------------------------------------

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

print("\n" + "=" * 65)
print("RANDOM FOREST GRID SEARCH")
print("=" * 65)

grid_search.fit(X, y)

best_rf = grid_search.best_estimator_

print("\nBest Random Forest parameters:")
print(grid_search.best_params_)

print(
    f"\nBest CV RMSE: "
    f"{-grid_search.best_score_:.4f}"
)


# ---------------------------------------------------------
# Evaluate best RF with multiple CV metrics
# ---------------------------------------------------------

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

print("\n" + "=" * 65)
print("TUNED RANDOM FOREST - 5 FOLD CROSS VALIDATION")
print("=" * 65)

print(f"Mean MAE  : {rf_mae:.4f}")
print(f"Mean RMSE : {rf_rmse:.4f}")
print(f"Mean R2   : {rf_r2:.4f}")


# ---------------------------------------------------------
# Final comparison
# ---------------------------------------------------------

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

print("\n" + "=" * 65)
print("FINAL CROSS-VALIDATION COMPARISON")
print("=" * 65)

print(
    results
    .sort_values(by="Mean RMSE")
    .to_string(index=False)
)