import argparse
import os
import time
import itertools

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
from sklearn.model_selection import KFold, ParameterSampler
from xgboost import XGBRegressor




FEATURES = [
    "weather_impact_index",
    "average_speed_kmph",
    "time_of_day",
    "traffic_density_index",
]

TARGET = "route_reliability_index"

RANDOM_SEED = 42
N_SPLITS = 5
STAGE2_TRIALS = 40

MLFLOW_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5002",
)

MLFLOW_EXPERIMENT = os.getenv(
    "MLFLOW_EXPERIMENT",
    "conditions-model-tuning",
)



def build_model(params):
    """
    Build XGBoost regression model.
    """

    return XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        **params,
    )




def get_stage1_search_space():
    """
    Broad/global search space.
    """

    return {
        "n_estimators": [50, 75, 100],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.05, 0.1, 0.15],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.8, 0.9, 1.0],
    }



def build_local_search_space(best_params):
    """
    Build a local search space around the best Stage 1
    parameters.

    Stage 2 is sampled randomly using ParameterSampler,
    rather than evaluating the complete Cartesian product.
    """



    learning_rate = best_params["learning_rate"]

    if learning_rate <= 0.05:

        learning_rate_values = [
            0.03,
            0.05,
            0.07,
        ]

    elif learning_rate < 0.15:

        learning_rate_values = [
            max(0.01, learning_rate - 0.025),
            learning_rate,
            learning_rate + 0.025,
        ]

    else:

        learning_rate_values = [
            0.10,
            0.125,
            0.15,
        ]



    n_estimators = best_params["n_estimators"]

    n_estimators_values = sorted(
        set(
            [
                max(25, n_estimators - 25),
                n_estimators,
                n_estimators + 25,
            ]
        )
    )


    max_depth = best_params["max_depth"]

    max_depth_values = sorted(
        set(
            [
                max(2, max_depth - 1),
                max_depth,
                max_depth + 1,
            ]
        )
    )



    subsample = best_params["subsample"]

    if subsample <= 0.8:

        subsample_values = [
            0.6,
            0.7,
            0.8,
        ]

    elif subsample < 1.0:

        subsample_values = [
            0.8,
            0.9,
            1.0,
        ]

    else:

        subsample_values = [
            0.8,
            0.9,
            1.0,
        ]



    colsample_bytree = (
        best_params["colsample_bytree"]
    )

    if colsample_bytree <= 0.8:

        colsample_values = [
            0.7,
            0.8,
            0.9,
        ]

    elif colsample_bytree < 1.0:

        colsample_values = [
            0.8,
            0.9,
            1.0,
        ]

    else:

        colsample_values = [
            0.85,
            0.9,
            1.0,
        ]



    return {
        "n_estimators": n_estimators_values,
        "max_depth": max_depth_values,
        "learning_rate": learning_rate_values,
        "subsample": subsample_values,
        "colsample_bytree": colsample_values,
        "min_child_weight": [
            1,
            3,
            5,
        ],
        "gamma": [
            0.0,
            0.1,
            0.2,
        ],
        "reg_alpha": [
            0.0,
            0.01,
            0.1,
        ],
        "reg_lambda": [
            0.5,
            1.0,
            2.0,
        ],
    }




def parameter_grid(search_space):
    """
    Generate the complete Cartesian product.

    Used only for Stage 1, where the grid size is
    intentionally small.
    """

    keys = list(search_space.keys())

    values = [
        search_space[key]
        for key in keys
    ]

    combinations = []

    for combination in itertools.product(
        *values
    ):

        combinations.append(
            dict(
                zip(
                    keys,
                    combination,
                )
            )
        )

    return combinations



def run_cross_validation(
    X,
    y,
    params,
    cv,
    print_folds=False,
):
    """
    Perform K-fold cross-validation.

    Primary metric:
        MAE - lower is better.

    Additional metrics:
        RMSE
        R²
    """

    fold_mae = []
    fold_rmse = []
    fold_r2 = []

    for fold_number, (
        train_idx,
        val_idx,
    ) in enumerate(
        cv.split(X),
        start=1,
    ):

        X_fold_train = X.iloc[train_idx]
        X_fold_val = X.iloc[val_idx]

        y_fold_train = y.iloc[train_idx]
        y_fold_val = y.iloc[val_idx]

        model = build_model(
            params
        )

        model.fit(
            X_fold_train,
            y_fold_train,
        )

        predictions = model.predict(
            X_fold_val
        )

        mae = mean_absolute_error(
            y_fold_val,
            predictions,
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_fold_val,
                predictions,
            )
        )

        r2 = r2_score(
            y_fold_val,
            predictions,
        )

        fold_mae.append(mae)
        fold_rmse.append(rmse)
        fold_r2.append(r2)

        if print_folds:

            print(
                f"  Fold {fold_number}: "
                f"MAE={mae:.5f}, "
                f"RMSE={rmse:.5f}, "
                f"R²={r2:.5f}"
            )

    return {
        "mae": float(
            np.mean(fold_mae)
        ),
        "rmse": float(
            np.mean(fold_rmse)
        ),
        "r2": float(
            np.mean(fold_r2)
        ),
    }




def prepare_data(
    train_df,
    test_df,
):
    """
    Validate and prepare regression data.
    """

    required_columns = (
        FEATURES + [TARGET]
    )

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
            "Missing columns in training data: "
            f"{missing_train}"
        )

    if missing_test:

        raise ValueError(
            "Missing columns in test data: "
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

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )




def run_stage1(
    X_train,
    y_train,
    cv,
):
    """
    Perform broad global search.
    """

    search_space = (
        get_stage1_search_space()
    )

    stage1_grid = parameter_grid(
        search_space
    )

    # Calculate dynamically so the number is always
    # consistent with the actual search space.
    stage1_trials = len(
        stage1_grid
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "STAGE 1 - GLOBAL SEARCH"
    )

    print(
        "=" * 70
    )

    print(
        f"Stage 1 trials: {stage1_trials}"
    )

    best_params = None
    best_metrics = None
    best_mae = float("inf")

    results = []

    for trial_number, params in enumerate(
        stage1_grid,
        start=1,
    ):

        print(
            f"\nStage 1 Trial "
            f"{trial_number}/{stage1_trials}"
        )

        print(
            f"Parameters: {params}"
        )

        start_time = time.time()

        # Fold-level output is intentionally disabled
        # for Stage 1.
        metrics = run_cross_validation(
            X_train,
            y_train,
            params,
            cv,
            print_folds=False,
        )

        elapsed = (
            time.time()
            - start_time
        )

        print(
            f"CV MAE: "
            f"{metrics['mae']:.5f} | "
            f"RMSE: "
            f"{metrics['rmse']:.5f} | "
            f"R²: "
            f"{metrics['r2']:.5f} | "
            f"Time: "
            f"{elapsed:.2f}s"
        )

        result = {
            **params,
            "stage": "stage1_global",
            "trial": trial_number,
            "cv_mae": metrics["mae"],
            "cv_rmse": metrics["rmse"],
            "cv_r2": metrics["r2"],
            "training_time_sec": elapsed,
        }

        results.append(
            result
        )

        if metrics["mae"] < best_mae:

            best_mae = (
                metrics["mae"]
            )

            best_params = (
                params.copy()
            )

            best_metrics = (
                metrics.copy()
            )

            print(
                "  *** New best "
                "Stage 1 model ***"
            )

    return (
        best_params,
        best_metrics,
        results,
        stage1_trials,
    )




def run_stage2(
    X_train,
    y_train,
    best_stage1_params,
    cv,
):
    """
    Perform local random refinement.

    The complete Stage 2 search space contains:

        9 parameters × 3 values each
        = 19,683 combinations.

    We intentionally sample only 40 combinations.
    """

    search_space = (
        build_local_search_space(
            best_stage1_params
        )
    )

    stage2_samples = list(
        ParameterSampler(
            search_space,
            n_iter=STAGE2_TRIALS,
            random_state=RANDOM_SEED,
        )
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "STAGE 2 - LOCAL REFINEMENT"
    )

    print(
        "=" * 70
    )

    print(
        f"Stage 2 trials: "
        f"{len(stage2_samples)}"
    )

    best_params = None
    best_metrics = None
    best_mae = float("inf")

    results = []

    for trial_number, params in enumerate(
        stage2_samples,
        start=1,
    ):

        print(
            f"\nStage 2 Trial "
            f"{trial_number}/"
            f"{len(stage2_samples)}"
        )

        print(
            f"Parameters: {params}"
        )

        start_time = time.time()

        # Fold-level output is enabled for Stage 2.
        metrics = run_cross_validation(
            X_train,
            y_train,
            params,
            cv,
            print_folds=True,
        )

        elapsed = (
            time.time()
            - start_time
        )

        print(
            f"CV MAE: "
            f"{metrics['mae']:.5f} | "
            f"RMSE: "
            f"{metrics['rmse']:.5f} | "
            f"R²: "
            f"{metrics['r2']:.5f} | "
            f"Time: "
            f"{elapsed:.2f}s"
        )

        result = {
            **params,
            "stage": "stage2_local",
            "trial": trial_number,
            "cv_mae": metrics["mae"],
            "cv_rmse": metrics["rmse"],
            "cv_r2": metrics["r2"],
            "training_time_sec": elapsed,
        }

        results.append(
            result
        )

        if metrics["mae"] < best_mae:

            best_mae = (
                metrics["mae"]
            )

            best_params = (
                params.copy()
            )

            best_metrics = (
                metrics.copy()
            )

            print(
                "  *** New best "
                "Stage 2 model ***"
            )

    return (
        best_params,
        best_metrics,
        results,
    )




def train_final_model(
    params,
    X_train,
    y_train,
    X_test,
    y_test,
):
    """
    Train the selected model on all training data
    and evaluate on untouched test data.
    """

    print(
        "\n" + "=" * 70
    )

    print(
        "Training final selected model "
        "on all training data..."
    )

    print(
        "=" * 70
    )

    model = build_model(
        params
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "\nEvaluating final model "
        "on untouched test data..."
    )

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

    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }

    print(
        f"\nTest MAE  : {mae:.5f}"
    )

    print(
        f"Test RMSE : {rmse:.5f}"
    )

    print(
        f"Test R²   : {r2:.5f}"
    )

    return (
        model,
        metrics,
    )




def main():

    parser = argparse.ArgumentParser(
        description=(
            "Two-stage XGBoost tuning for "
            "route_reliability_index"
        )
    )

    parser.add_argument(
        "--train",
        required=True,
        help=(
            "Path to processed training "
            "Parquet dataset"
        ),
    )

    parser.add_argument(
        "--test",
        required=True,
        help=(
            "Path to processed test "
            "Parquet dataset"
        ),
    )

    args = parser.parse_args()



    mlflow.set_tracking_uri(
        MLFLOW_URI
    )

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT
    )

    print(
        f"MLflow tracking URI: "
        f"{MLFLOW_URI}"
    )

    print(
        f"MLflow experiment: "
        f"{MLFLOW_EXPERIMENT}"
    )



    print(
        "\nLoading processed datasets..."
    )

    train_df = pd.read_parquet(
        args.train
    )

    test_df = pd.read_parquet(
        args.test
    )

    print(
        f"Training rows: "
        f"{len(train_df)}"
    )

    print(
        f"Test rows: "
        f"{len(test_df)}"
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


    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = prepare_data(
        train_df,
        test_df,
    )



    cv = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )



    with mlflow.start_run(
        run_name="XGBoost_Tuning_Final"
    ) as parent_run:

        mlflow.set_tag(
            "problem_type",
            "regression",
        )

        mlflow.set_tag(
            "target",
            TARGET,
        )

        mlflow.set_tag(
            "tuning_method",
            "two_stage_search",
        )

        mlflow.set_tag(
            "stage2_sampling",
            "ParameterSampler",
        )

        mlflow.log_param(
            "random_seed",
            RANDOM_SEED,
        )

        mlflow.log_param(
            "cv_folds",
            N_SPLITS,
        )

        mlflow.log_param(
            "stage2_trials",
            STAGE2_TRIALS,
        )

        mlflow.log_param(
            "features",
            ",".join(FEATURES),
        )



        (
            stage1_best_params,
            stage1_best_metrics,
            stage1_results,
            stage1_trials,
        ) = run_stage1(
            X_train,
            y_train,
            cv,
        )

        print(
            "\n" + "=" * 70
        )

        print(
            "STAGE 1 RESULT"
        )

        print(
            "=" * 70
        )

        print(
            f"Stage 1 trials: "
            f"{stage1_trials}"
        )

        print(
            f"Best Stage 1 CV MAE: "
            f"{stage1_best_metrics['mae']:.5f}"
        )

        print(
            f"Best Stage 1 CV RMSE: "
            f"{stage1_best_metrics['rmse']:.5f}"
        )

        print(
            f"Best Stage 1 CV R²: "
            f"{stage1_best_metrics['r2']:.5f}"
        )

        print(
            f"Best parameters: "
            f"{stage1_best_params}"
        )



        (
            stage2_best_params,
            stage2_best_metrics,
            stage2_results,
        ) = run_stage2(
            X_train,
            y_train,
            stage1_best_params,
            cv,
        )



        if (
            stage2_best_metrics["mae"]
            <= stage1_best_metrics["mae"]
        ):

            best_stage = (
                "stage2_local"
            )

            best_params = (
                stage2_best_params
            )

            best_cv_metrics = (
                stage2_best_metrics
            )

        else:

            best_stage = (
                "stage1_global"
            )

            best_params = (
                stage1_best_params
            )

            best_cv_metrics = (
                stage1_best_metrics
            )



        print(
            "\n" + "=" * 70
        )

        print(
            "FINAL TUNING RESULT"
        )

        print(
            "=" * 70
        )

        print(
            f"Best stage: "
            f"{best_stage}"
        )

        print(
            f"Best CV MAE: "
            f"{best_cv_metrics['mae']:.5f}"
        )

        print(
            f"Best CV RMSE: "
            f"{best_cv_metrics['rmse']:.5f}"
        )

        print(
            f"Best CV R²: "
            f"{best_cv_metrics['r2']:.5f}"
        )

        print(
            f"Best parameters: "
            f"{best_params}"
        )


        (
            final_model,
            test_metrics,
        ) = train_final_model(
            best_params,
            X_train,
            y_train,
            X_test,
            y_test,
        )


        mlflow.log_metric(
            "best_cv_mae",
            best_cv_metrics["mae"],
        )

        mlflow.log_metric(
            "best_cv_rmse",
            best_cv_metrics["rmse"],
        )

        mlflow.log_metric(
            "best_cv_r2",
            best_cv_metrics["r2"],
        )

        mlflow.log_metric(
            "test_mae",
            test_metrics["mae"],
        )

        mlflow.log_metric(
            "test_rmse",
            test_metrics["rmse"],
        )

        mlflow.log_metric(
            "test_r2",
            test_metrics["r2"],
        )

        mlflow.set_tag(
            "best_stage",
            best_stage,
        )



        for key, value in best_params.items():

            if isinstance(
                value,
                (dict, list, tuple),
            ):
                value = str(value)

            mlflow.log_param(
                f"final_{key}",
                value,
            )



        all_results = (
            stage1_results
            + stage2_results
        )

        results_df = pd.DataFrame(
            all_results
        )

        results_path = (
            "conditions_tuning_results.csv"
        )

        results_df.to_csv(
            results_path,
            index=False,
        )

        mlflow.log_artifact(
            results_path
        )



        input_example = (
            X_train.head(1)
        )

        signature = infer_signature(
            X_train,
            final_model.predict(
                input_example
            ),
        )

        mlflow.xgboost.log_model(
            final_model,
            artifact_path="final_model",
            signature=signature,
            input_example=input_example,
            model_format="json",
        )



        print(
            "\n" + "=" * 70
        )

        print(
            "TUNING COMPLETED"
        )

        print(
            "=" * 70
        )

        print(
            f"Best Stage: "
            f"{best_stage}"
        )

        print(
            f"Best CV MAE: "
            f"{best_cv_metrics['mae']:.5f}"
        )

        print(
            f"Final Test MAE: "
            f"{test_metrics['mae']:.5f}"
        )

        print(
            f"Final Test RMSE: "
            f"{test_metrics['rmse']:.5f}"
        )

        print(
            f"Final Test R²: "
            f"{test_metrics['r2']:.5f}"
        )

        print(
            f"MLflow Parent Run: "
            f"{parent_run.info.run_id}"
        )


if __name__ == "__main__":
    main()