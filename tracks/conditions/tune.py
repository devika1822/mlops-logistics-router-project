import argparse
import json
import os
import time

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd

from mlflow.models import infer_signature

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier



FEATURES = [
    "weather_impact_index",
    "time_of_day",
    "traffic_density_index",
]

TARGET = "reliability_class"

LABEL_MAPPING = {
    "Low": 0,
    "Medium": 1,
    "High": 2,
}

LABEL_NAMES = [
    "Low",
    "Medium",
    "High",
]

RANDOM_SEED = 42

CV_FOLDS = 5

# Stage 1: original V1 search
STAGE1_TRIALS = 108

# Stage 2: localized V2 refinement
STAGE2_TRIALS = 40

MLFLOW_EXPERIMENT = (
    "conditions_route_reliability_tuning_final"
)



def load_data(train_path, test_path):
    """Load the Spark-generated Parquet datasets."""

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    return train_df, test_df


def prepare_data(train_df, test_df):

    X_train = train_df[FEATURES].copy()
    X_test = test_df[FEATURES].copy()

    y_train = train_df[TARGET].map(LABEL_MAPPING)
    y_test = test_df[TARGET].map(LABEL_MAPPING)

    if y_train.isna().any():
        raise ValueError(
            "Unknown or missing class found in training data."
        )

    if y_test.isna().any():
        raise ValueError(
            "Unknown or missing class found in test data."
        )

    return X_train, X_test, y_train, y_test



def build_model(params):
    """Create an XGBoost multiclass classifier."""

    return XGBClassifier(
        objective="multi:softmax",
        num_class=3,
        eval_metric="mlogloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        **params,
    )




def run_cross_validation(
    X_train,
    y_train,
    params,
    cv,
    print_folds=True,
):


    fold_f1 = []
    fold_accuracy = []

    for fold_number, (train_idx, val_idx) in enumerate(
        cv.split(X_train, y_train),
        start=1,
    ):

        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]

        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]

        model = build_model(params)

        model.fit(
            X_fold_train,
            y_fold_train,
            verbose=False,
        )

        predictions = model.predict(
            X_fold_val
        )

        fold_f1_value = f1_score(
            y_fold_val,
            predictions,
            average="macro",
            zero_division=0,
        )

        fold_accuracy_value = accuracy_score(
            y_fold_val,
            predictions,
        )

        fold_f1.append(
            fold_f1_value
        )

        fold_accuracy.append(
            fold_accuracy_value
        )

        if print_folds:
            print(
                f"    Fold {fold_number}: "
                f"F1={fold_f1_value:.4f}, "
                f"Accuracy={fold_accuracy_value:.4f}"
            )

    return {
        "cv_macro_f1": float(
            np.mean(fold_f1)
        ),
        "cv_macro_f1_std": float(
            np.std(fold_f1)
        ),
        "cv_accuracy": float(
            np.mean(fold_accuracy)
        ),
    }



def evaluate_test_model(
    model,
    X_test,
    y_test,
):
    """Evaluate the selected model on the untouched test set."""

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        y_test,
        predictions,
        target_names=LABEL_NAMES,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "macro_f1": macro_f1,
        "report": report,
        "matrix": matrix,
        "predictions": predictions,
    }




def get_stage1_search_space():
    """
    Original V1 search space.

    3 x 3 x 3 x 2 x 2 = 108 configurations.
    """

    return {
        "n_estimators": [
            50,
            100,
            200,
        ],
        "max_depth": [
            3,
            5,
            7,
        ],
        "learning_rate": [
            0.05,
            0.1,
            0.2,
        ],
        "subsample": [
            0.8,
            1.0,
        ],
        "colsample_bytree": [
            0.8,
            1.0,
        ],
    }



def build_local_search_space(best_params):
    """
    Build a localized Stage-2 search around the best
    Stage-1 configuration.

    Additional regularization parameters are introduced
    here, based on the V2 strategy.
    """

    n_estimators = best_params["n_estimators"]
    max_depth = best_params["max_depth"]
    learning_rate = best_params["learning_rate"]
    subsample = best_params["subsample"]
    colsample_bytree = best_params["colsample_bytree"]



    estimator_values = sorted(
        {
            max(50, n_estimators - 50),
            max(50, n_estimators - 25),
            n_estimators,
            n_estimators + 25,
            n_estimators + 50,
        }
    )



    depth_values = sorted(
        {
            max(3, max_depth - 1),
            max_depth,
            min(10, max_depth + 1),
        }
    )



    learning_rate_values = sorted(
        {
            max(0.025, round(
                learning_rate * 0.5,
                3,
            )),
            max(0.025, round(
                learning_rate * 0.75,
                3,
            )),
            round(learning_rate, 3),
            round(
                learning_rate * 1.25,
                3,
            ),
        }
    )

    learning_rate_values = [
        value
        for value in learning_rate_values
        if value <= 0.3
    ]



    if subsample <= 0.8:
        subsample_values = [
            0.7,
            0.8,
            0.9,
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



    if colsample_bytree <= 0.8:
        colsample_values = [
            0.8,
            0.9,
            1.0,
        ]
    else:
        colsample_values = [
            0.8,
            0.9,
            1.0,
        ]

    return {
        "n_estimators": estimator_values,
        "max_depth": depth_values,
        "learning_rate": learning_rate_values,
        "subsample": subsample_values,
        "colsample_bytree": colsample_values,

        # New regularization parameters in Stage 2
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
            1.0,
            2.0,
            3.0,
        ],
    }



def log_trial(
    stage_name,
    trial_number,
    params,
    cv_results,
    elapsed_seconds,
):
    """Log one tuning trial to MLflow."""

    with mlflow.start_run(
        run_name=(
            f"{stage_name}_trial_"
            f"{trial_number:03d}"
        ),
        nested=True,
    ):

        mlflow.log_param(
            "stage",
            stage_name,
        )

        for key, value in params.items():
            mlflow.log_param(
                key,
                value,
            )

        mlflow.log_metric(
            "cv_macro_f1",
            cv_results["cv_macro_f1"],
        )

        mlflow.log_metric(
            "cv_macro_f1_std",
            cv_results["cv_macro_f1_std"],
        )

        mlflow.log_metric(
            "cv_accuracy",
            cv_results["cv_accuracy"],
        )

        mlflow.log_metric(
            "training_time_seconds",
            elapsed_seconds,
        )



def save_results(
    results,
    output_dir,
):
    """Save tuning results as CSV."""

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    results_df = pd.DataFrame(
        results
    )

    path = os.path.join(
        output_dir,
        "xgboost_tuning_final_results.csv",
    )

    results_df.to_csv(
        path,
        index=False,
    )

    return path


def save_best_parameters(
    best_params,
    output_dir,
):
    """Save the selected parameters as JSON."""

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    path = os.path.join(
        output_dir,
        "xgboost_best_parameters.json",
    )

    with open(
        path,
        "w",
    ) as file:

        json.dump(
            best_params,
            file,
            indent=4,
        )

    return path


def save_final_artifacts(
    model,
    metrics,
    output_dir,
):
    """Save final test-evaluation artifacts."""

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    report_path = os.path.join(
        output_dir,
        "xgboost_final_classification_report.txt",
    )

    with open(
        report_path,
        "w",
    ) as file:
        file.write(
            metrics["report"]
        )

    confusion_path = os.path.join(
        output_dir,
        "xgboost_final_confusion_matrix.txt",
    )

    with open(
        confusion_path,
        "w",
    ) as file:

        file.write(
            "Labels: Low, Medium, High\n\n"
        )

        file.write(
            str(metrics["matrix"])
        )

    feature_importance_df = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance": model.feature_importances_,
        }
    ).sort_values(
        by="importance",
        ascending=False,
    )

    feature_importance_path = os.path.join(
        output_dir,
        "xgboost_final_feature_importance.csv",
    )

    feature_importance_df.to_csv(
        feature_importance_path,
        index=False,
    )

    return {
        "report": report_path,
        "confusion": confusion_path,
        "feature_importance": feature_importance_path,
    }




def main():

    parser = argparse.ArgumentParser(
        description=(
            "Two-stage XGBoost tuning for "
            "Conditions/Environment track"
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

    parser.add_argument(
        "--mlruns",
        default="mlruns",
        help="MLflow tracking directory",
    )

    args = parser.parse_args()



    tracking_path = os.path.abspath(
        args.mlruns
    )

    mlflow.set_tracking_uri(
        f"file://{tracking_path}"
    )

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT
    )



    print(
        "Loading processed datasets..."
    )

    train_df, test_df = load_data(
        args.train,
        args.test,
    )

    X_train, X_test, y_train, y_test = prepare_data(
        train_df,
        test_df,
    )

    print(
        f"Training rows: {len(X_train)}"
    )

    print(
        f"Test rows: {len(X_test)}"
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



    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    all_results = []

    best_cv_score = -1.0
    best_params = None
    best_stage = None



    with mlflow.start_run(
        run_name="XGBoost_Tuning_Final"
    ) as parent_run:

        mlflow.log_param(
            "search_strategy",
            "broad_then_local_refinement",
        )

        mlflow.log_param(
            "stage1_trials",
            STAGE1_TRIALS,
        )

        mlflow.log_param(
            "stage2_trials",
            STAGE2_TRIALS,
        )

        mlflow.log_param(
            "cv_folds",
            CV_FOLDS,
        )

        mlflow.log_param(
            "optimization_metric",
            "cv_macro_f1",
        )

        mlflow.log_param(
            "train_rows",
            len(X_train),
        )

        mlflow.log_param(
            "test_rows",
            len(X_test),
        )

        mlflow.log_param(
            "random_seed",
            RANDOM_SEED,
        )

        mlflow.log_param(
            "features",
            ",".join(FEATURES),
        )


        print(
            "\n"
            + "=" * 70
        )

        print(
            "STAGE 1 - BROAD XGBOOST SEARCH"
        )

        print(
            "=" * 70
        )

        stage1_grid = list(
            ParameterGrid(
                get_stage1_search_space()
            )
        )

        print(
            f"Stage 1 configurations: "
            f"{len(stage1_grid)}"
        )

        for trial_number, params in enumerate(
            stage1_grid,
            start=1,
        ):

            print(
                "\n"
                + "-" * 60
            )

            print(
                f"Stage 1 Trial "
                f"{trial_number}/{STAGE1_TRIALS}"
            )

            print(
                f"Parameters: {params}"
            )

            start_time = time.time()

            cv_results = run_cross_validation(
                X_train,
                y_train,
                params,
                cv,
            )

            elapsed_seconds = (
                time.time() - start_time
            )

            log_trial(
                stage_name="stage1_broad",
                trial_number=trial_number,
                params=params,
                cv_results=cv_results,
                elapsed_seconds=elapsed_seconds,
            )

            result = {
                "stage": "stage1_broad",
                "trial": trial_number,
                **params,
                **cv_results,
                "training_time_seconds": (
                    elapsed_seconds
                ),
            }

            all_results.append(
                result
            )

            print(
                f"    CV Macro F1: "
                f"{cv_results['cv_macro_f1']:.4f}"
            )

            if (
                cv_results["cv_macro_f1"]
                > best_cv_score
            ):
                best_cv_score = (
                    cv_results["cv_macro_f1"]
                )

                best_params = params.copy()

                best_stage = "stage1_broad"

                print(
                    "\n    New overall best!"
                )

        print(
            "\n"
            + "=" * 70
        )

        print(
            "BEST STAGE 1 CONFIGURATION"
        )

        print(
            "=" * 70
        )

        print(
            f"CV Macro F1: "
            f"{best_cv_score:.4f}"
        )

        print(
            f"Parameters: "
            f"{best_params}"
        )



        print(
            "\n"
            + "=" * 70
        )

        print(
            "STAGE 2 - LOCAL REFINEMENT"
        )

        print(
            "=" * 70
        )

        stage2_search_space = (
            build_local_search_space(
                best_params
            )
        )

        stage2_samples = list(
            ParameterSampler(
                stage2_search_space,
                n_iter=STAGE2_TRIALS,
                random_state=RANDOM_SEED,
            )
        )

        print(
            f"Stage 2 configurations: "
            f"{len(stage2_samples)}"
        )

        for trial_number, params in enumerate(
            stage2_samples,
            start=1,
        ):

            print(
                "\n"
                + "-" * 60
            )

            print(
                f"Stage 2 Trial "
                f"{trial_number}/{STAGE2_TRIALS}"
            )

            print(
                f"Parameters: {params}"
            )

            start_time = time.time()

            cv_results = run_cross_validation(
                X_train,
                y_train,
                params,
                cv,
            )

            elapsed_seconds = (
                time.time() - start_time
            )

            log_trial(
                stage_name="stage2_local",
                trial_number=trial_number,
                params=params,
                cv_results=cv_results,
                elapsed_seconds=elapsed_seconds,
            )

            result = {
                "stage": "stage2_local",
                "trial": trial_number,
                **params,
                **cv_results,
                "training_time_seconds": (
                    elapsed_seconds
                ),
            }

            all_results.append(
                result
            )

            print(
                f"    CV Macro F1: "
                f"{cv_results['cv_macro_f1']:.4f}"
            )

            if (
                cv_results["cv_macro_f1"]
                > best_cv_score
            ):
                best_cv_score = (
                    cv_results["cv_macro_f1"]
                )

                best_params = params.copy()

                best_stage = "stage2_local"

                print(
                    "\n    New overall best!"
                )



        results_path = save_results(
            all_results,
            "reports/conditions",
        )

        mlflow.log_artifact(
            results_path
        )

        best_params_path = save_best_parameters(
            best_params,
            "reports/conditions",
        )

        mlflow.log_artifact(
            best_params_path
        )


        print(
            "\n"
            + "=" * 70
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
            f"Best CV Macro F1: "
            f"{best_cv_score:.4f}"
        )

        print(
            f"Best parameters: "
            f"{best_params}"
        )

        mlflow.log_param(
            "best_stage",
            best_stage,
        )

        mlflow.log_metric(
            "best_cv_macro_f1",
            best_cv_score,
        )

        for key, value in best_params.items():

            mlflow.log_param(
                f"best_{key}",
                value,
            )



        print(
            "\nTraining final selected model "
            "on all training data..."
        )

        final_model = build_model(
            best_params
        )

        final_model.fit(
            X_train,
            y_train,
            verbose=False,
        )


        print(
            "\nEvaluating final model on "
            "untouched test data..."
        )

        test_metrics = evaluate_test_model(
            final_model,
            X_test,
            y_test,
        )

        print(
            f"\nTest Accuracy : "
            f"{test_metrics['accuracy']:.4f}"
        )

        print(
            f"Test Precision: "
            f"{test_metrics['precision']:.4f}"
        )

        print(
            f"Test Recall   : "
            f"{test_metrics['recall']:.4f}"
        )

        print(
            f"Test Macro F1 : "
            f"{test_metrics['macro_f1']:.4f}"
        )



        artifacts = save_final_artifacts(
            final_model,
            test_metrics,
            "reports/conditions",
        )

        mlflow.log_artifact(
            artifacts["report"]
        )

        mlflow.log_artifact(
            artifacts["confusion"]
        )

        mlflow.log_artifact(
            artifacts["feature_importance"]
        )


        input_example = X_train.head(3)

        example_predictions = (
            final_model.predict(
                input_example
            )
        )

        signature = infer_signature(
            input_example,
            example_predictions,
        )

        mlflow.xgboost.log_model(
            final_model,
            artifact_path="final_model",
            signature=signature,
            input_example=input_example,
        )


        mlflow.log_metric(
            "test_accuracy",
            test_metrics["accuracy"],
        )

        mlflow.log_metric(
            "test_macro_precision",
            test_metrics["precision"],
        )

        mlflow.log_metric(
            "test_macro_recall",
            test_metrics["recall"],
        )

        mlflow.log_metric(
            "test_macro_f1",
            test_metrics["macro_f1"],
        )

        print(
            "\n"
            + "=" * 70
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
            f"Best CV Macro F1: "
            f"{best_cv_score:.4f}"
        )

        print(
            f"Final Test Macro F1: "
            f"{test_metrics['macro_f1']:.4f}"
        )

        print(
            f"MLflow Parent Run: "
            f"{parent_run.info.run_id}"
        )


if __name__ == "__main__":
    main()