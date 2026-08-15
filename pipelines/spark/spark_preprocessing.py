from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = str(
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ecommerce_logistics_route_planning_dataset.csv"
)

PROCESSED_DATA_PATH = str(
    PROJECT_ROOT
    / "data"
    / "processed"
    / "geography"
)


# Geography configuration

TARGET = "optimized_route_time_min"

FINAL_FEATURES = [
    "order_latitude",
    "order_longitude",
    "distance_km",
    "delivery_time_window_hrs",
    "order_priority",
    "traffic_density_index",
]

REQUIRED_COLUMNS = FINAL_FEATURES + [TARGET]


# Create Spark session

spark = (
    SparkSession.builder
    .appName("FleetIQ-Geography-Preprocessing")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# Load raw data

df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(RAW_DATA_PATH)
)

print("\nRaw dataset loaded successfully.")
print(f"Raw row count: {df.count()}")
print(f"Raw column count: {len(df.columns)}")


# Validate required columns

missing_columns = [
    column
    for column in REQUIRED_COLUMNS
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required Geography columns: {missing_columns}"
    )

print("\nAll required Geography columns are present.")


# Select Geography columns

geography_df = df.select(*REQUIRED_COLUMNS)


# Missing-value check

print("\nMissing-value counts:")

missing_counts = geography_df.select(
    [
        F.sum(F.col(column).isNull().cast("int")).alias(column)
        for column in geography_df.columns
    ]
)

missing_counts.show(truncate=False)


# Remove rows containing null values in required fields.
# Current dataset has none, but this makes the pipeline robust
# to future incoming datasets.
geography_df = geography_df.dropna(
    subset=REQUIRED_COLUMNS
)


# Duplicate handling

rows_before_duplicates = geography_df.count()

geography_df = geography_df.dropDuplicates()

rows_after_duplicates = geography_df.count()

print(
    "\nDuplicate rows removed:",
    rows_before_duplicates - rows_after_duplicates,
)


# Data validity checks

rows_before_validation = geography_df.count()

geography_df = geography_df.filter(

    # Valid global latitude range
    F.col("order_latitude").between(-90, 90)

    # Valid global longitude range
    & F.col("order_longitude").between(-180, 180)

    # Delivery distance must be positive
    & (F.col("distance_km") > 0)

    # Delivery window must be positive
    & (F.col("delivery_time_window_hrs") > 0)

    # Traffic density is represented as a normalized 0–1 index
    & F.col("traffic_density_index").between(0, 1)

    # Dataset uses priority values 1–4
    & F.col("order_priority").between(1, 4)

    # Route time must be positive
    & (F.col(TARGET) > 0)
)

rows_after_validation = geography_df.count()

print(
    "Invalid rows removed:",
    rows_before_validation - rows_after_validation,
)


# Outlier detection

# We FLAG statistical outliers rather than automatically
# deleting them because long-distance/long-duration deliveries
# may still be legitimate logistics observations.


def get_iqr_bounds(dataframe, column_name):
    q1, q3 = dataframe.approxQuantile(
        column_name,
        [0.25, 0.75],
        0.0,
    )

    iqr = q3 - q1

    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    return lower_bound, upper_bound


distance_lower, distance_upper = get_iqr_bounds(
    geography_df,
    "distance_km",
)

time_lower, time_upper = get_iqr_bounds(
    geography_df,
    TARGET,
)


geography_df = geography_df.withColumn(
    "distance_outlier_flag",
    (
        (F.col("distance_km") < distance_lower)
        | (F.col("distance_km") > distance_upper)
    ).cast("int"),
)


geography_df = geography_df.withColumn(
    "route_time_outlier_flag",
    (
        (F.col(TARGET) < time_lower)
        | (F.col(TARGET) > time_upper)
    ).cast("int"),
)


print("\nDistance outliers detected:")

geography_df.groupBy(
    "distance_outlier_flag"
).count().show()


print("Route-time outliers detected:")

geography_df.groupBy(
    "route_time_outlier_flag"
).count().show()


# Feature engineering

# Traffic impact may depend on how long the route is.
geography_df = geography_df.withColumn(
    "distance_traffic_interaction",
    F.col("distance_km")
    * F.col("traffic_density_index"),
)


# Represents delivery-distance pressure relative to
# the promised delivery window.
geography_df = geography_df.withColumn(
    "distance_per_delivery_window",
    F.col("distance_km")
    / F.col("delivery_time_window_hrs"),
)

# Final processed dataset

OUTPUT_COLUMNS = [
    "order_latitude",
    "order_longitude",
    "distance_km",
    "delivery_time_window_hrs",
    "order_priority",
    "traffic_density_index",

    # Engineered features
    "distance_traffic_interaction",
    "distance_per_delivery_window",

    # Outlier indicators
    "distance_outlier_flag",
    "route_time_outlier_flag",

    # Target
    TARGET,
]

processed_df = geography_df.select(
    *OUTPUT_COLUMNS
)


print("\nFinal processed schema:")
processed_df.printSchema()

print("\nSample processed records:")
processed_df.show(5, truncate=False)

print(
    f"\nFinal processed row count: {processed_df.count()}"
)

# Write processed data

(
    processed_df.write
    .mode("overwrite")
    .parquet(PROCESSED_DATA_PATH)
)

print(
    f"\nProcessed Geography data written to:\n"
    f"{PROCESSED_DATA_PATH}"
)


spark.stop()