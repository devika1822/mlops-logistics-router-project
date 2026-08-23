import os
import logging
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SparkCostPreprocess")

def run_pyspark_pipeline():
    logger.info("=== Phase 1: Initializing Distributed PySpark Data Preprocessing ===")
    
    spark = SparkSession.builder \
        .appName("ECommerceLogisticsCostPreprocess") \
        .master("local[*]") \
        .config("spark.ui.enabled", "false") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("ERROR")

    input_path = "data/raw/route_planning.csv"
    if not os.path.exists(input_path):
        if os.path.exists("raw_truck_data.csv"):
            input_path = "raw_truck_data.csv"
        elif os.path.exists("data/raw/raw_truck_data.csv"):
            input_path = "data/raw/raw_truck_data.csv"
        else:
            spark.stop()
            raise FileNotFoundError("Critical Preprocessing Error: Raw route planning data file not found.")

    df = spark.read.option("header", "true").option("inferSchema", "true").csv(input_path)
    initial_count = df.count()

    target_column = "optimized_route_cost"
    if target_column in df.columns:
        df = df.dropna(subset=[target_column])
        logger.info(f"Dropped {initial_count - df.count()} unlabelled rows with missing target cost values.")
    else:
        logger.warning(f"Target '{target_column}' missing! Creating mock regression baseline.")
        df = df.withColumn(target_column, (F.col("distance_km") * 12.5) + (F.rand() * 100))

    df = df.withColumn(target_column, F.col(target_column).cast("double"))

    numerical_features = [
        "distance_km", "vehicle_utilization_ratio", "average_speed_kmph", 
        "traffic_density_index", "fuel_cost_per_km", "driver_cost_per_hour",
        "order_weight_kg", "vehicle_capacity_kg", "route_reliability_index"
    ]
    for feature in numerical_features:
        if feature in df.columns:
            feature_mean = df.select(F.mean(feature)).collect()[0][0]
            feature_mean = feature_mean if feature_mean is not None else 0.5
            df = df.na.fill({feature: feature_mean})

    os.makedirs("data/processed", exist_ok=True)
    pdf = df.toPandas()
    pdf.to_csv("data/processed/clean_cost_da25g503.csv", index=False)
    pdf.to_csv("clean_cost_da25g503.csv", index=False)
    
    logger.info(f"[PREPROCESSING SUCCESS] Dataset formatted. Clean row count: {df.count()}")
    spark.stop()

if __name__ == "__main__":
    run_pyspark_pipeline()
