import argparse
import os

import mlflow
import mlflow.xgboost
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

MODEL_NAME = "conditions-route-reliability"

MLFLOW_TRACKING_URI = "http://127.0.0.1:5002"

MLFLOW_EXPERIMENT = (
    "conditions_route_reliability_final"
)



def load_data(train_path, test_path):
    """Load processed Parquet datasets."""

    train_df = pd.read_parquet(
        train_path
    )

    test_df = pd.read_parquet(
        test_path
    )

    return train_df, test_df


def prepare_data(train_df, test_df):
    """Prepare features and encode target classes."""

    X_train = train_df[
        FEATURES
    ].copy()

    X_test = test_df[
        FEATURES
    ].copy()

    y_train = train_df[
        TARGET
    ].map(LABEL_MAPPING)

    y_test = test_df[
        TARGET
    ].map(LABEL_MAPPING)

    if y_train.isna().any():
        raise ValueError(
            "Unknown or missing target class in training data."
        )

    if y_test.isna().any():
        raise ValueError(
            "Unknown or missing target class in test data."
        )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )




def create_final_model():
    """Create the selected final XGBoost model."""

    return XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.125,
        subsample=0.8,
        colsample_bytree=1.0,
        min_child_weight=3,
        gamma=0.2,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="multi:softmax",
        num_class=3,
        eval_metric="mlogloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )



def evaluate_model(
    model,
    X_test,
    y_test,
):
    """Evaluate the final model."""

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
    }




def save_artifacts(
    model,
    metrics,
    output_dir,
):
    """Save evaluation and feature-importance files."""

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    # Classification report
    report_path = os.path.join(
        output_dir,
        "classification_report.txt",
    )

    with open(
        report_path,
        "w",
    ) as file:
        file.write(
            metrics["report"]
        )

    # Confusion matrix
    confusion_path = os.path.join(
        output_dir,
        "confusion_matrix.txt",
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

    # Feature importance
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
        "report": report_path,
        "confusion": confusion_path,
        "feature_importance": (
            feature_importance_path
        ),
    }



def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train and register the final "
            "Conditions XGBoost model"
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
        f"MLflow server: "
        f"{MLFLOW_TRACKING_URI}"
    )

    print(
        f"Model name: {MODEL_NAME}"
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

    train_rows = len(
        X_train
    )

    test_rows = len(
        X_test
    )

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


    model = create_final_model()

    print(
        "\nFinal XGBoost parameters:"
    )

    for parameter, value in model.get_params().items():

        if parameter in {
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "min_child_weight",
            "gamma",
            "reg_alpha",
            "reg_lambda",
        }:
            print(
                f"  {parameter}: {value}"
            )


    print(
        "\nTraining final model"
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
        f"Accuracy       : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Macro Precision: "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Macro Recall   : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"Macro F1       : "
        f"{metrics['macro_f1']:.4f}"
    )



    with mlflow.start_run(
        run_name="Final_XGBoost"
    ) as run:

        # Dataset information
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
            "target",
            TARGET,
        )

        mlflow.log_param(
            "random_seed",
            RANDOM_SEED,
        )

        mlflow.log_param(
            "model_selection",
            "V2 localized tuning",
        )

        mlflow.log_param(
            "cv_folds",
            5,
        )

        # Model parameters
        model_params = model.get_params()

        selected_parameters = [
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "min_child_weight",
            "gamma",
            "reg_alpha",
            "reg_lambda",
        ]

        for parameter in selected_parameters:

            mlflow.log_param(
                parameter,
                model_params[parameter],
            )

        # Metrics
        mlflow.log_metric(
            "accuracy",
            metrics["accuracy"],
        )

        mlflow.log_metric(
            "macro_precision",
            metrics["precision"],
        )

        mlflow.log_metric(
            "macro_recall",
            metrics["recall"],
        )

        mlflow.log_metric(
            "macro_f1",
            metrics["macro_f1"],
        )

        # Artifacts
        artifact_dir = (
            "reports/conditions/final_model"
        )

        artifacts = save_artifacts(
            model=model,
            metrics=metrics,
            output_dir=artifact_dir,
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

        # Model signature
        input_example = X_train.head(
            3
        )

        example_predictions = (
            model.predict(
                input_example
            )
        )

        signature = infer_signature(
            input_example,
            example_predictions,
        )

        model_info = (
            mlflow.xgboost.log_model(
                model,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
                registered_model_name=MODEL_NAME,
            )
        )

        print(
            "\nModel logged successfully."
        )

        print(
            f"Run ID: {run.info.run_id}"
        )

        print(
            f"Model URI: "
            f"{model_info.model_uri}"
        )

        print(
            "\nModel registered successfully."
        )

        print(
            f"Registered model: "
            f"{MODEL_NAME}"
        )


if __name__ == "__main__":
    main()