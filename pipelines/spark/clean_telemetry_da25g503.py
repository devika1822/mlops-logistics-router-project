import os
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

import os
import glob

# Automatically find and set JAVA_HOME to our local project folder if not already set
if not os.environ.get("JAVA_HOME"):
    jdk_dirs = glob.glob("jdk11_extracted/jdk-11*")
    if jdk_dirs:
        os.environ["JAVA_HOME"] = os.path.abspath(jdk_dirs[0])
        
def run():
    print("--- Running Local PySpark Telemetry Cleaning ---")
    
    # Initialize a lightweight local Spark session
    spark = SparkSession.builder \
        .appName("LocalTelemetryCleaning") \
        .master("local[*]") \
        .config("spark.ui.enabled", "false") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("ERROR")

    input_path = "data/raw/raw_truck_data.csv"
    if not os.path.exists(input_path):
        input_path = "raw_truck_data.csv"
        if not os.path.exists(input_path):
            raise FileNotFoundError("Raw telemetry data file not found.")

    # Read data using Spark
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(input_path)
    print(f"[SPARK] Initial row count: {df.count()}")

    # Feature engineering & transformations
    if "vehicle_utilization_ratio" not in df.columns:
        df = df.withColumn("vehicle_utilization_ratio", F.lit(0.5))
    if "average_speed_kmph" not in df.columns:
        df = df.withColumn("average_speed_kmph", F.lit(40.0))
    if "speed_strain_factor" not in df.columns:
        df = df.withColumn("speed_strain_factor", F.lit(0.3))

    if "breakdown_risk_level" not in df.columns:
        df = df.withColumn(
            "breakdown_risk_level",
            F.when((F.col("vehicle_utilization_ratio") > 0.85) | (F.col("speed_strain_factor") > 0.65), "HIGH")
             .when((F.col("vehicle_utilization_ratio") > 0.60) | (F.col("speed_strain_factor") > 0.35), "MEDIUM")
             .otherwise("LOW")
        )

    # Save output processed data
    os.makedirs("data/processed", exist_ok=True)
    target_file = "data/processed/clean_telemtery_da25g503.csv"
    
    # Convert to pandas locally to write out a clean CSV file for your training script
    pdf = df.toPandas()
    pdf.to_csv(target_file, index=False)
    pdf.to_csv("clean_telemtery_da25g503.csv", index=False)
    
    print(f"[SPARK] Cleaning complete! Saved to {target_file}")
    spark.stop()

if __name__ == "__main__":
    run()