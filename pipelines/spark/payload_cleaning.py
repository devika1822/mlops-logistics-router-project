"""
Cargo Payload Track – Apache Spark Cleaning Job

Transformations applied:
  1. Drop rows with nulls in any key column
  2. Convert cargo_weight_lbs  →  cargo_weight_kg  (divide by 2.20462)
  3. Derive overloaded_flag    =  1 if vehicle_capacity_pct > 100
  4. Clip delivery_deadline_hrs to ≥ 0 (remove negative deadlines)
  5. Clip engine_wear_score to [0, 100]
  6. Cast all numeric columns to DoubleType for downstream ML

Input  : data/raw/payload_raw.parquet
Output : data/processed/payload_clean.parquet
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

INPUT_PATH  = "data/raw/payload_raw.parquet"
OUTPUT_PATH = "data/processed/payload_clean.parquet"

# Raw column names as they arrive from Kafka / the CSV
KEY_COLS = [
    "order_weight_kg",
    "vehicle_capacity_kg",
    "vehicle_utilization_ratio",
    "delivery_time_window_hrs",
    "distance_km",
    "traffic_density_index",
    "weather_impact_index",
]


def run() -> None:
    spark = (
        SparkSession.builder
        .appName("PayloadCleaning")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.parquet(INPUT_PATH)
    raw_count = df.count()
    print(f"[SPARK] Raw rows: {raw_count}")

    # 1. Cast all key columns to numeric (Kafka streams deliver strings)
    for col in KEY_COLS:
        df = df.withColumn(col, F.col(col).cast(DoubleType()))

    # 2. Drop rows with nulls in any key column
    df = df.dropna(subset=KEY_COLS)

    # 3. Standardise column names for downstream ML
    df = (
        df.withColumnRenamed("order_weight_kg",         "cargo_weight_kg")
          .withColumnRenamed("delivery_time_window_hrs", "delivery_deadline_hrs")
          .withColumnRenamed("distance_km",              "trip_distance_km")
    )

    # 4. Convert utilization ratio (0-1) to capacity percentage (0-100)
    df = df.withColumn(
        "vehicle_capacity_pct",
        F.round(F.col("vehicle_utilization_ratio") * 100.0, 2),
    ).drop("vehicle_utilization_ratio")

    # 5. Flag overloaded trucks (utilization > 100 % of rated capacity)
    df = df.withColumn(
        "overloaded_flag",
        F.when(F.col("vehicle_capacity_pct") > 100.0, 1).otherwise(0),
    )

    # 6. Clip negative deadlines to 0
    df = df.withColumn(
        "delivery_deadline_hrs",
        F.greatest(F.col("delivery_deadline_hrs"), F.lit(0.0)),
    )

    # 7. Derive engine_wear_score (0-100) from available stress indicators
    #    load stress 0-40, distance stress 0-30, traffic 0-20, weather 0-10
    df = df.withColumn(
        "engine_wear_score",
        F.least(
            F.round(
                (F.col("vehicle_capacity_pct") / 100.0) * 40.0
                + F.least(F.col("trip_distance_km") / 500.0, F.lit(1.0)) * 30.0
                + F.col("traffic_density_index") * 20.0
                + F.col("weather_impact_index") * 10.0,
                2,
            ),
            F.lit(100.0),
        ),
    )

    clean_count = df.count()
    print(f"[SPARK] Clean rows: {clean_count}  (dropped {raw_count - clean_count} rows)")
    df.printSchema()

    df.write.mode("overwrite").parquet(OUTPUT_PATH)
    print(f"[SPARK] Saved → {OUTPUT_PATH}")

    spark.stop()


if __name__ == "__main__":
    run()
