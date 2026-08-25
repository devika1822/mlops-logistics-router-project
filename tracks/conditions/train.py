import argparse
import os

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd

from mlflow.models import infer_signature
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.tree import DecisionTreeRegressor
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

MLFLOW_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5002",
)

MLFLOW_EXPERIMENT = os.getenv(
    "MLFLOW_EXPERIMENT",
    "conditions-model-training",
)




def prepare_data(train_df, test_df):
    """
    Validate and prepare training/test data for regression.

    The model directly predicts the numeric
    route_reliability_index target.
    """

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
            f"Missing columns in training data: {missing_train}"
        )

    if missing_test:
        raise ValueError(
            f"Missing columns in test data: {missing_test}"
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

    return X_train, X_test, y_train, y_test




def build_decision_tree():
    return DecisionTreeRegressor(
        max_depth=6,
        min_samples_leaf=5,
        random_state=RANDOM_SEED,
    )


def build_xgboost():
    return XGBRegressor(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.125,
        subsample=0.8,
        colsample_bytree=1.0,
        min_child_weight=3,
        gamma=0.2,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def evaluate_model(model, X_test, y_test):
    """
    Calculate regression metrics.
    """

    predictions = model.predict(X_test)

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




def log_model_run(
    model_name,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
):
    """
    Train, evaluate and log one model
    as a nested MLflow run.
    """

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)


    model.fit(
        X_train,
        y_train,
    )



    metrics = evaluate_model(
        model,
        X_test,
        y_test,
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



    with mlflow.start_run(
        run_name=model_name,
        nested=True,
    ):



        if hasattr(model, "get_params"):

            params = model.get_params()

            for key, value in params.items():

                if value is not None:

                    # Convert complex values to strings
                    # before sending them to MLflow.
                    if isinstance(
                        value,
                        (dict, list, tuple),
                    ):
                        value = str(value)

                    mlflow.log_param(
                        key,
                        value,
                    )



        mlflow.log_metrics(
            metrics
        )



        mlflow.set_tag(
            "model_type",
            model_name,
        )

        mlflow.set_tag(
            "target",
            TARGET,
        )

        mlflow.set_tag(
            "problem_type",
            "regression",
        )



        input_example = X_train.head(1)

        signature = infer_signature(
            X_train,
            model.predict(input_example),
        )



        if isinstance(
            model,
            XGBRegressor,
        ):

            mlflow.xgboost.log_model(
                model,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
                model_format="json",
            )

        else:

            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
            )

        print(
            f"\nMLflow run logged: {model_name}"
        )

    return metrics["mae"], metrics




def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train Conditions regression models"
        )
    )

    parser.add_argument(
        "--train",
        required=True,
        help="Path to processed training Parquet",
    )

    parser.add_argument(
        "--test",
        required=True,
        help="Path to processed test Parquet",
    )

    args = parser.parse_args()


    mlflow.set_tracking_uri(
        MLFLOW_URI
    )

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT
    )

    print(
        f"MLflow tracking URI: {MLFLOW_URI}"
    )

    print(
        f"MLflow experiment: {MLFLOW_EXPERIMENT}"
    )



    print("\nLoading processed datasets...")

    train_df = pd.read_parquet(
        args.train
    )

    test_df = pd.read_parquet(
        args.test
    )

    print(
        f"Training rows: {len(train_df)}"
    )

    print(
        f"Test rows: {len(test_df)}"
    )

    print("\nFeatures:")

    for feature in FEATURES:
        print(f"  - {feature}")

    print(
        f"\nTarget: {TARGET}"
    )


    X_train, X_test, y_train, y_test = prepare_data(
        train_df,
        test_df,
    )



    with mlflow.start_run(
        run_name="Model_Comparison"
    ) as parent_run:

        mlflow.set_tag(
            "problem_type",
            "regression",
        )

        mlflow.set_tag(
            "target",
            TARGET,
        )

        mlflow.log_param(
            "random_seed",
            RANDOM_SEED,
        )

        mlflow.log_param(
            "feature_count",
            len(FEATURES),
        )

        mlflow.log_param(
            "features",
            ",".join(FEATURES),
        )



        decision_tree = build_decision_tree()

        dt_mae, dt_metrics = log_model_run(
            "Decision_Tree",
            decision_tree,
            X_train,
            y_train,
            X_test,
            y_test,
        )



        xgb_model = build_xgboost()

        xgb_mae, xgb_metrics = log_model_run(
            "XGBoost",
            xgb_model,
            X_train,
            y_train,
            X_test,
            y_test,
        )



        comparison = pd.DataFrame(
            [
                {
                    "model": "Decision Tree",
                    "mae": dt_metrics["mae"],
                    "rmse": dt_metrics["rmse"],
                    "r2": dt_metrics["r2"],
                },
                {
                    "model": "XGBoost",
                    "mae": xgb_metrics["mae"],
                    "rmse": xgb_metrics["rmse"],
                    "r2": xgb_metrics["r2"],
                },
            ]
        )

        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)

        print(
            comparison.to_string(
                index=False
            )
        )



        best_index = comparison[
            "mae"
        ].idxmin()

        best_model_name = comparison.loc[
            best_index,
            "model",
        ]

        best_mae = comparison.loc[
            best_index,
            "mae",
        ]

        best_rmse = comparison.loc[
            best_index,
            "rmse",
        ]

        best_r2 = comparison.loc[
            best_index,
            "r2",
        ]

        print(
            f"\nBest model based on MAE: "
            f"{best_model_name}"
        )

        print(
            f"Best MAE : {best_mae:.4f}"
        )

        print(
            f"Best RMSE: {best_rmse:.4f}"
        )

        print(
            f"Best R²  : {best_r2:.4f}"
        )


        mlflow.log_metric(
            "best_model_mae",
            float(best_mae),
        )

        mlflow.log_metric(
            "best_model_rmse",
            float(best_rmse),
        )

        mlflow.log_metric(
            "best_model_r2",
            float(best_r2),
        )

        mlflow.set_tag(
            "best_model",
            best_model_name,
        )

        print(
            f"\nParent MLflow run: "
            f"{parent_run.info.run_id}"
        )


if __name__ == "__main__":
    main()