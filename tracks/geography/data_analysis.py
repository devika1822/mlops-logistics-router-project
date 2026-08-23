from pathlib import Path

import pandas as pd



PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ecommerce_logistics_route_planning_dataset.csv"
)


# Geography track configuration

TARGET = "optimized_route_time_min"

ORIGINAL_FEATURES = [
    "order_latitude",
    "order_longitude",
    "distance_km",
    "delivery_time_window_hrs",
    "order_priority",
    "traffic_density_index",
    "average_speed_kmph",
]

FINAL_FEATURES = [
    "order_latitude",
    "order_longitude",
    "distance_km",
    "delivery_time_window_hrs",
    "order_priority",
    "traffic_density_index",
]


df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print(f"Dataset shape: {df.shape}")

print("\nColumns:")
print(df.columns.tolist())


# Basic data-quality checks

geography_columns = ORIGINAL_FEATURES + [TARGET]
geography_df = df[geography_columns].copy()

print("\nMissing values:")
print(geography_df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nGeography data summary:")
print(geography_df.describe())



#Pairwise correlation analysis

print("\nPairwise correlations with target:")

correlations = (
    geography_df
    .corr(numeric_only=True)[TARGET]
    .sort_values(ascending=False)
)

print(correlations)



#Domain-informed leakage check

derived_route_time = (
    (df["distance_km"] / df["average_speed_kmph"]) * 60
) * (
    1 + 0.5 * df["traffic_density_index"]
)

derived_correlation = derived_route_time.corr(df[TARGET])

print("\nDerived route-time correlation:")
print(derived_correlation)



#Final feature selection

print("\nOriginal features:")
print(ORIGINAL_FEATURES)

print("\nFinal leakage-reduced features:")
print(FINAL_FEATURES)

print(
    "\naverage_speed_kmph excluded because, together with "
    "distance_km and traffic_density_index, it almost exactly "
    "reconstructs the target."
)