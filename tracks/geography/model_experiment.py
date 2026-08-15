from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



# Paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ecommerce_logistics_route_planning_dataset.csv"
)



# Configuration

TARGET = "optimized_route_time_min"

FEATURES_WITH_SPEED = [
    "order_latitude",
    "order_longitude",
    "distance_km",
    "delivery_time_window_hrs",
    "order_priority",
    "traffic_density_index",
    "average_speed_kmph",
]

FEATURES_WITHOUT_SPEED = [
    "order_latitude",
    "order_longitude",
    "distance_km",
    "delivery_time_window_hrs",
    "order_priority",
    "traffic_density_index",
]


df = pd.read_csv(DATA_PATH)


# Evaluate a feature set

def run_experiment(feature_names, experiment_name):

    X = df[feature_names]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    # Linear Regression:
    # scale input features before fitting the model.
    linear_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )

    # Random Forest:
    # scaling is not required for tree-based models.
    random_forest_model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
    )

    models = {
        "Linear Regression": linear_model,
        "Random Forest": random_forest_model,
    }

    print("\n" + "=" * 65)
    print(experiment_name)
    print("=" * 65)

    results = []

    for model_name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(
            y_test,
            predictions,
        ) ** 0.5
        r2 = r2_score(y_test, predictions)

        results.append(
            {
                "Feature Set": experiment_name,
                "Model": model_name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
            }
        )

        print(f"\n{model_name}")
        print(f"MAE  : {mae:.4f}")
        print(f"RMSE : {rmse:.4f}")
        print(f"R2   : {r2:.4f}")

    return results


# Run both experiments

all_results = []

all_results.extend(
    run_experiment(
        FEATURES_WITH_SPEED,
        "WITH average_speed_kmph",
    )
)

all_results.extend(
    run_experiment(
        FEATURES_WITHOUT_SPEED,
        "WITHOUT average_speed_kmph",
    )
)



# Final comparison

results_df = pd.DataFrame(all_results)

print("\n" + "=" * 65)
print("FINAL COMPARISON")
print("=" * 65)

print(results_df.to_string(index=False))