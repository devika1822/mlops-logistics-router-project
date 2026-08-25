import argparse
import os

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd

from mlflow.models import infer_signature
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from xgboost import XGBRegressor



FEATURES = [
    "weather_impact_index",
    "average_speed_kmph",
    "time_of_day",
    "traffic_density_index",
]

TARGET = "route_reliability_index"

RANDOM_SEED = 42

MODEL_NAME = "conditions-route-reliability"

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5002",
)

MLFLOW_EXPERIMENT = os.getenv(
    "MLFLOW_EXPERIMENT",
    "conditions_route_reliability_final",
)

MODEL_ALIAS = "champion"




FINAL_MODEL_PARAMS = {
    "n_estimators": 75,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.6,
    "colsample_bytree": 1.0,
    "min_child_weight": 5,
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
}




EXPECTED_MAX_MAE = 0.0100

REFERENCE_TEST_MAE = 0.00454
REFERENCE_TEST_RMSE = 0.00592
REFERENCE_TEST_R2 = 0.99827




def load_data(train_path, test_path):

    train_df = pd.read_parquet(
        train_path
    )

    test_df = pd.read_parquet(
        test_path
    )

    return train_df, test_df




def prepare_data(train_df, test_df):

    required_columns = FEATURES + [TARGET]

    missing_train = [
        column
        for column in required_columns
        if column not in train_df.columns
    ]

    missing_test = [
        column
        for column in required_columns
        if column not in test_df.columns
    ]

    if missing_train:
        raise ValueError(
            "Missing required columns in training data: "
            f"{missing_train}"
        )

    if missing_test:
        raise ValueError(
            "Missing required columns in test data: "
            f"{missing_test}"
        )

    X_train = (
        train_df[FEATURES]
        .copy()
        .astype("float64")
    )

    X_test = (
        test_df[FEATURES]
        .copy()
        .astype("float64")
    )

    y_train = (
        train_df[TARGET]
        .copy()
        .astype("float64")
    )

    y_test = (
        test_df[TARGET]
        .copy()
        .astype("float64")
    )

    if y_train.isna().any():
        raise ValueError(
            f"Missing values found in "
            f"training target '{TARGET}'."
        )

    if y_test.isna().any():
        raise ValueError(
            f"Missing values found in "
            f"test target '{TARGET}'."
        )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )



def create_final_model():

    return XGBRegressor(
        **FINAL_MODEL_PARAMS,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )




def evaluate_model(
    model,
    X_test,
    y_test,
):

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions,
        )
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }




def save_artifacts(
    model,
    X_test,
    y_test,
    output_dir,
):

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    predictions = model.predict(
        X_test
    )



    prediction_df = pd.DataFrame(
        {
            "actual_route_reliability_index":
                y_test.values,
            "predicted_route_reliability_index":
                predictions,
            "absolute_error":
                np.abs(
                    y_test.values
                    - predictions
                ),
        }
    )

    predictions_path = os.path.join(
        output_dir,
        "predictions.csv",
    )

    prediction_df.to_csv(
        predictions_path,
        index=False,
    )



    feature_importance_df = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance": (
                model.feature_importances_
            ),
        }
    ).sort_values(
        by="importance",
        ascending=False,
    )

    feature_importance_path = os.path.join(
        output_dir,
        "feature_importance.csv",
    )

    feature_importance_df.to_csv(
        feature_importance_path,
        index=False,
    )

    return {
        "predictions": predictions_path,
        "feature_importance": feature_importance_path,
    }




def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train and register the final "
            "Conditions XGBoost regression model"
        )
    )

    parser.add_argument(
        "--train",
        required=True,
        help="Path to training Parquet",
    )

    parser.add_argument(
        "--test",
        required=True,
        help="Path to test Parquet",
    )

    args = parser.parse_args()


    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT
    )

    print(
        f"MLflow tracking URI: "
        f"{MLFLOW_TRACKING_URI}"
    )

    print(
        f"Registered model: "
        f"{MODEL_NAME}"
    )



    print(
        "\nLoading processed datasets..."
    )

    train_df, test_df = load_data(
        args.train,
        args.test,
    )


    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = prepare_data(
        train_df,
        test_df,
    )

    train_rows = len(X_train)
    test_rows = len(X_test)

    print(
        f"Training rows: {train_rows}"
    )

    print(
        f"Test rows: {test_rows}"
    )

    print(
        "\nFeatures:"
    )

    for feature in FEATURES:
        print(
            f"  - {feature}"
        )

    print(
        f"\nTarget: {TARGET}"
    )



    print(
        "\nFinal XGBoost parameters:"
    )

    for parameter, value in (
        FINAL_MODEL_PARAMS.items()
    ):
        print(
            f"  {parameter}: {value}"
        )



    model = create_final_model()

    print(
        "\nTraining final model..."
    )

    model.fit(
        X_train,
        y_train,
        verbose=False,
    )


    metrics = evaluate_model(
        model,
        X_test,
        y_test,
    )

    print(
        "\nFinal model performance:"
    )

    print(
        f"MAE  : {metrics['mae']:.4f}"
    )

    print(
        f"RMSE : {metrics['rmse']:.4f}"
    )

    print(
        f"R²   : {metrics['r2']:.4f}"
    )



    print(
        f"\nExpected maximum MAE: "
        f"{EXPECTED_MAX_MAE:.4f}"
    )

    print(
        f"Reference test MAE: "
        f"{REFERENCE_TEST_MAE:.4f}"
    )

    print(
        f"Actual test MAE: "
        f"{metrics['mae']:.4f}"
    )

    if metrics["mae"] > EXPECTED_MAX_MAE:

        raise ValueError(
            f"Performance gate failed. "
            f"Reproduced model MAE "
            f"{metrics['mae']:.4f} "
            f"is above the allowed maximum "
            f"{EXPECTED_MAX_MAE:.4f}. "
            "Model registration aborted."
        )

    print(
        "Performance gate passed."
    )



    with mlflow.start_run(
        run_name="Final_XGBoost"
    ) as run:


        mlflow.log_param(
            "train_rows",
            train_rows,
        )

        mlflow.log_param(
            "test_rows",
            test_rows,
        )

        mlflow.log_param(
            "features",
            ",".join(FEATURES),
        )

        mlflow.log_param(
            "feature_count",
            len(FEATURES),
        )

        mlflow.log_param(
            "target",
            TARGET,
        )

        mlflow.log_param(
            "target_type",
            "regression",
        )

        mlflow.log_param(
            "model_type",
            "XGBRegressor",
        )

        mlflow.log_param(
            "random_seed",
            RANDOM_SEED,
        )

        mlflow.log_param(
            "model_selection",
            "two_stage_tuning",
        )

        mlflow.log_param(
            "selection_metric",
            "mae",
        )

        mlflow.log_param(
            "expected_max_mae",
            EXPECTED_MAX_MAE,
        )



        mlflow.log_metric(
            "reference_test_mae",
            REFERENCE_TEST_MAE,
        )

        mlflow.log_metric(
            "reference_test_rmse",
            REFERENCE_TEST_RMSE,
        )

        mlflow.log_metric(
            "reference_test_r2",
            REFERENCE_TEST_R2,
        )


        for parameter, value in (
            FINAL_MODEL_PARAMS.items()
        ):

            if isinstance(
                value,
                (dict, list, tuple),
            ):
                value = str(value)

            mlflow.log_param(
                parameter,
                value,
            )



        mlflow.log_metric(
            "mae",
            metrics["mae"],
        )

        mlflow.log_metric(
            "rmse",
            metrics["rmse"],
        )

        mlflow.log_metric(
            "r2",
            metrics["r2"],
        )

        mlflow.set_tag(
            "problem_type",
            "regression",
        )

        mlflow.set_tag(
            "target",
            TARGET,
        )

        mlflow.set_tag(
            "final_model_status",
            "selected",
        )

        mlflow.set_tag(
            "performance_gate",
            "passed",
        )


        artifact_dir = (
            "reports/conditions/final_model"
        )

        artifacts = save_artifacts(
            model=model,
            X_test=X_test,
            y_test=y_test,
            output_dir=artifact_dir,
        )

        mlflow.log_artifact(
            artifacts["predictions"]
        )

        mlflow.log_artifact(
            artifacts["feature_importance"]
        )



        input_example = X_train.head(1)

        example_predictions = model.predict(
            input_example
        )

        signature = infer_signature(
            X_train,
            example_predictions,
        )



        model_info = mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            model_format="json",
            registered_model_name=MODEL_NAME,
        )

        print(
            "\nModel logged successfully."
        )

        print(
            f"Run ID     : "
            f"{run.info.run_id}"
        )

        print(
            f"Run URI    : "
            f"{model_info.model_uri}"
        )



    client = mlflow.MlflowClient(
        tracking_uri=MLFLOW_TRACKING_URI
    )

    registered_versions = (
        client.search_model_versions(
            f"name='{MODEL_NAME}'"
        )
    )

    matching_versions = [
        version
        for version in registered_versions
        if version.run_id == run.info.run_id
    ]

    if not matching_versions:

        raise RuntimeError(
            "Model was logged successfully, "
            "but the registered model version "
            "could not be found for the current run."
        )

    registered_version = max(
        matching_versions,
        key=lambda version: int(
            version.version
        ),
    )

    version_number = str(
        registered_version.version
    )



    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=MODEL_ALIAS,
        version=version_number,
    )

    print(
        f"\nAlias '{MODEL_ALIAS}' -> "
        f"version {version_number}"
    )

    print(
        f"Registry URI: "
        f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    )



    print(
        "\nModel registered successfully."
    )

    print(
        f"Registered model: "
        f"{MODEL_NAME}"
    )

    print(
        f"Registered version: "
        f"{version_number}"
    )

    print(
        f"Champion alias: "
        f"{MODEL_ALIAS}"
    )


if __name__ == "__main__":
    main()