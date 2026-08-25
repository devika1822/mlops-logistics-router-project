import argparse
import json
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import min as spark_min, max as spark_max


FEATURES = [
    "weather_impact_index",
    "average_speed_kmph",
    "time_of_day",
    "traffic_density_index",
]

TARGET = "route_reliability_index"


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

        input_rows = df.count()

        print(f"Input rows: {input_rows}")

        required_columns = FEATURES + [TARGET]
        actual_columns = df.columns

        missing_columns = [
            column
            for column in required_columns
            if column not in actual_columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}. "
                f"Available columns: {actual_columns}"
            )

        print("\nRequired columns validated successfully.")


        selected_df = df.select(
            *(FEATURES + [TARGET])
        )


        cleaned_df = selected_df.dropna(
            subset=required_columns
        )

        cleaned_rows = cleaned_df.count()

        print(
            f"Rows after missing-value handling: "
            f"{cleaned_rows}"
        )

        if cleaned_rows == 0:
            raise ValueError(
                "No valid rows remain after removing missing values."
            )


        train_df, test_df = cleaned_df.randomSplit(
            [1.0 - test_size, test_size],
            seed=seed,
        )

        train_count = train_df.count()
        test_count = test_df.count()

        if train_count == 0 or test_count == 0:
            raise ValueError(
                "Train/test split produced an empty dataset. "
                "Please check test_size."
            )

        print(f"Training rows: {train_count}")
        print(f"Test rows: {test_count}")


        target_stats = train_df.agg(
            spark_min(TARGET).alias("target_min"),
            spark_max(TARGET).alias("target_max"),
        ).collect()[0]

        target_min = target_stats["target_min"]
        target_max = target_stats["target_max"]

        print("\nTraining target range:")
        print(f"  {TARGET} min = {target_min}")
        print(f"  {TARGET} max = {target_max}")

        if target_min is None or target_max is None:
            raise ValueError(
                f"Unable to determine target range for {TARGET}."
            )

        if target_min == target_max:
            raise ValueError(
                f"Target {TARGET} has no variation in training data. "
                "Regression model cannot be trained meaningfully."
            )


        train_output = os.path.join(
            output_path,
            "train",
        )

        test_output = os.path.join(
            output_path,
            "test",
        )

        os.makedirs(
            output_path,
            exist_ok=True,
        )


        (
            train_df
            .write
            .mode("overwrite")
            .parquet(train_output)
        )


        (
            test_df
            .write
            .mode("overwrite")
            .parquet(test_output)
        )


        metadata = {
            "features": FEATURES,
            "target": TARGET,
            "target_type": "regression",
            "target_min": float(target_min),
            "target_max": float(target_max),
            "test_size": test_size,
            "random_seed": seed,
            "input_rows": input_rows,
            "cleaned_rows": cleaned_rows,
            "train_rows": train_count,
            "test_rows": test_count,
        }

        metadata_path = os.path.join(
            output_path,
            "preprocessing_metadata.json",
        )

        with open(metadata_path, "w") as f:
            json.dump(
                metadata,
                f,
                indent=4,
            )


        print(
            f"\nTraining data written to: "
            f"{train_output}"
        )

        print(
            f"Test data written to: "
            f"{test_output}"
        )

        print(
            f"Metadata written to: "
            f"{metadata_path}"
        )

        print("\nPreprocessing completed successfully.")

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
        help="Path to raw CSV dataset",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for processed data",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data used for testing",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    if not 0.0 < args.test_size < 1.0:
        raise ValueError(
            "test-size must be between 0 and 1."
        )

    preprocess(
        input_path=args.input,
        output_path=args.output,
        test_size=args.test_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()