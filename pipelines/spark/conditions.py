import argparse
import json
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when


FEATURES = [
    "weather_impact_index",
    "time_of_day",
    "average_speed_kmph",
    "traffic_density_index",
]

TARGET = "route_reliability_index"


def create_reliability_class(df, low_medium_threshold, medium_high_threshold):


    return df.withColumn(
        "reliability_class",
        when(
            col(TARGET) <= low_medium_threshold,
            "Low"
        )
        .when(
            col(TARGET) <= medium_high_threshold,
            "Medium"
        )
        .otherwise("High")
    )


def preprocess(input_path, output_path, test_size=0.2, seed=42):

    spark = (
        SparkSession.builder
        .appName("ConditionsPreprocessing")
        .master("local[*]")
        .getOrCreate()
    )

    try:

        print(f"Reading dataset: {input_path}")

        df = (
            spark.read
            .option("header", True)
            .option("inferSchema", True)
            .csv(input_path)
        )

        print(f"Input rows: {df.count()}")


        selected_df = df.select(
            *(FEATURES + [TARGET])
        )

        cleaned_df = selected_df.dropna(
            subset=FEATURES + [TARGET]
        )

        print(
            f"Rows after missing-value handling: "
            f"{cleaned_df.count()}"
        )

        train_df, test_df = cleaned_df.randomSplit(
            [1.0 - test_size, test_size],
            seed=seed
        )

        train_count = train_df.count()
        test_count = test_df.count()

        print(f"Training rows: {train_count}")
        print(f"Test rows: {test_count}")

        quantiles = train_df.approxQuantile(
            TARGET,
            [1 / 3, 2 / 3],
            0.0
        )

        low_medium_threshold = quantiles[0]
        medium_high_threshold = quantiles[1]

        print(
            "Reliability thresholds learned from training data:"
        )
        print(
            f"  Low / Medium  = {low_medium_threshold}"
        )
        print(
            f"  Medium / High = {medium_high_threshold}"
        )

        train_processed = create_reliability_class(
            train_df,
            low_medium_threshold,
            medium_high_threshold
        )

        test_processed = create_reliability_class(
            test_df,
            low_medium_threshold,
            medium_high_threshold
        )


        print("\nTraining class distribution:")

        (
            train_processed
            .groupBy("reliability_class")
            .count()
            .orderBy("reliability_class")
            .show()
        )

        print("Test class distribution:")

        (
            test_processed
            .groupBy("reliability_class")
            .count()
            .orderBy("reliability_class")
            .show()
        )


        train_output = os.path.join(
            output_path,
            "train"
        )

        test_output = os.path.join(
            output_path,
            "test"
        )

        os.makedirs(output_path, exist_ok=True)


        (
            train_processed
            .write
            .mode("overwrite")
            .parquet(train_output)
        )

        (
            test_processed
            .write
            .mode("overwrite")
            .parquet(test_output)
        )

        metadata = {
            "features": FEATURES,
            "target": TARGET,
            "target_type": "classification",
            "classes": [
                "Low",
                "Medium",
                "High"
            ],
            "low_medium_threshold": low_medium_threshold,
            "medium_high_threshold": medium_high_threshold,
            "test_size": test_size,
            "random_seed": seed,
            "input_rows": df.count(),
            "cleaned_rows": cleaned_df.count(),
            "train_rows": train_count,
            "test_rows": test_count
        }

        metadata_path = os.path.join(
            output_path,
            "preprocessing_metadata.json"
        )

        with open(metadata_path, "w") as f:
            json.dump(
                metadata,
                f,
                indent=4
            )

        print(
            f"\nTraining data written to: {train_output}"
        )

        print(
            f"Test data written to: {test_output}"
        )

        print(
            f"Metadata written to: {metadata_path}"
        )

    finally:
        spark.stop()


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Spark preprocessing for "
            "Conditions/Environment track"
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to raw CSV dataset"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for processed data"
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data used for testing"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    preprocess(
        input_path=args.input,
        output_path=args.output,
        test_size=args.test_size,
        seed=args.seed
    )


if __name__ == "__main__":
    main()