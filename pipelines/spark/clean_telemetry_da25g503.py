import os
import logging
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SparkDataEngine")

def run_pyspark_pipeline():
    logger.info("=== Initializing Container-Native PySpark Regression Pipeline ===")
    
    spark = SparkSession.builder \
        .appName("ECommerceLogisticsTelemetryCleaning") \
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
            raise FileNotFoundError("Critical Ingestion Fault: Raw input telemetry data source file not found.")

    logger.info(f"Ingesting source data from: {input_path}")
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(input_path)
    
    initial_count = df.count()
    logger.info(f"Initial row count: {initial_count}")

    target_column = "delivery_efficiency_score"
    if target_column in df.columns:
        df = df.dropna(subset=[target_column])
        dropped_count = initial_count - df.count()
        logger.info(f"Dropped {dropped_count} rows due to missing '{target_column}' target values.")
    else:
        logger.warning(f"Target column '{target_column}' missing from raw file! Generating mockup data points.")
        df = df.withColumn(target_column, F.rand() * 0.95)

    df = df.withColumn(target_column, F.col(target_column).cast("double"))
    numerical_features = ["distance_km", "vehicle_utilization_ratio", "average_speed_kmph", "traffic_density_index"]
    
    for feature in numerical_features:
        if feature in df.columns:
            feature_mean = df.select(F.mean(feature)).collect()[0][0]
            feature_mean = feature_mean if feature_mean is not None else 0.5
            
            df = df.na.fill({feature: feature_mean})

    os.makedirs("data/processed", exist_ok=True)
    pdf = df.toPandas()
    pdf.to_csv("data/processed/clean_telemetery_da25g503.csv", index=False)
    pdf.to_csv("clean_telemetery_da25g503.csv", index=False)
    
    logger.info(f"[PIPELINE SUCCESS] Processed data saved. Final clean row count: {df.count()}")
    spark.stop()

if __name__ == "__main__":
    run_pyspark_pipeline()
