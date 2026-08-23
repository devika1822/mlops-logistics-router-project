import os
import logging
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SparkDataEngine")

def run_pyspark_pipeline():
    logger.info("=== Initializing Container-Native PySpark Processing Pipeline ===")
    
    spark = SparkSession.builder \
        .appName("ECommerceLogisticsTelemetryCleaning") \
        .master("local[*]") \
        .config("spark.driver.memory", "2g") \
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

    logger.info(f"Ingesting source data from target path: {input_path}")
    
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(input_path)
    logger.info(f"[SPARK DATA INGESTION COMPLETE] Initial record matrix row count: {df.count()}")

    if "vehicle_utilization_ratio" not in df.columns:
        logger.info("Applying target feature transformation: vehicle_utilization_ratio (Default fallback: 0.5)")
        df = df.withColumn("vehicle_utilization_ratio", F.lit(0.5))
        
    if "average_speed_kmph" not in df.columns:
        logger.info("Applying target feature transformation: average_speed_kmph (Default fallback: 40.0)")
        df = df.withColumn("average_speed_kmph", F.lit(40.0))
        
    if "speed_strain_factor" not in df.columns:
        logger.info("Applying target feature transformation: speed_strain_factor (Default fallback: 0.3)")
        df = df.withColumn("speed_strain_factor", F.lit(0.3))

    logger.info("Executing mathematical conditional profiling map to construct target: breakdown_risk_level")
    df = df.withColumn(
        "breakdown_risk_level",
        F.when((F.col("vehicle_utilization_ratio") > 0.85) | (F.col("speed_strain_factor") > 0.65), "HIGH")
         .when((F.col("vehicle_utilization_ratio") > 0.60) | (F.col("speed_strain_factor") > 0.35), "MEDIUM")
         .otherwise("LOW")
    )

    os.makedirs("data/processed", exist_ok=True)
    target_file = "data/processed/clean_telemtery_da25g503.csv"
    
    logger.info("Consolidating Spark frames. Exporting matrix to shared storage layout layers...")
    pdf = df.toPandas()
    
    pdf.to_csv(target_file, index=False)
    pdf.to_csv("clean_telemtery_da25g503.csv", index=False)
    
    logger.info(f"[PIPELINE SUCCESS] PySpark data transformations completed. Output saved to -> {target_file}")
    
    spark.stop()

if __name__ == "__main__":
    run_pyspark_pipeline()
