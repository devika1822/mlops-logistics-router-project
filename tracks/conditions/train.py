import argparse
import os

import mlflow
import mlflow.sklearn
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
from sklearn.tree import DecisionTreeClassifier

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


def load_data(train_path, test_path):

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    return train_df, test_df


def prepare_data(train_df, test_df):

    X_train = train_df[FEATURES].copy()
    X_test = test_df[FEATURES].copy()

    y_train = train_df[TARGET].map(LABEL_MAPPING)
    y_test = test_df[TARGET].map(LABEL_MAPPING)

    if y_train.isna().any() or y_test.isna().any():
        raise ValueError(
            "Unknown or missing target class found in dataset."
        )

    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

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


def get_feature_importance(model):

    importance_df = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance": model.feature_importances_,
        }
    )

    return (
        importance_df
        .sort_values(
            by="importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def save_artifacts(
    model_name,
    metrics,
    model,
    output_dir,
):

    model_dir = os.path.join(
        output_dir,
        model_name,
    )

    os.makedirs(
        model_dir,
        exist_ok=True,
    )

    # Classification report
    report_path = os.path.join(
        model_dir,
        "classification_report.txt",
    )

    with open(
        report_path,
        "w",
    ) as file:
        file.write(metrics["report"])

    # Confusion matrix
    confusion_matrix_path = os.path.join(
        model_dir,
        "confusion_matrix.txt",
    )

    with open(
        confusion_matrix_path,
        "w",
    ) as file:
        file.write(
            "Labels: Low, Medium, High\n\n"
        )
        file.write(
            str(metrics["matrix"])
        )

    # Feature importance
    feature_importance_df = get_feature_importance(
        model
    )

    feature_importance_path = os.path.join(
        model_dir,
        "feature_importance.csv",
    )

    feature_importance_df.to_csv(
        feature_importance_path,
        index=False,
    )

    return {
        "report": report_path,
        "confusion_matrix": confusion_matrix_path,
        "feature_importance": feature_importance_path,
    }


def log_model_run(
    model_name,
    model,
    metrics,
    X_train,
    train_rows,
    test_rows,
    output_dir,
):
    """Log one model run to MLflow."""

    with mlflow.start_run(
        run_name=model_name
    ):

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

        # Model parameters
        model_params = model.get_params()

        for parameter, value in model_params.items():

            if value is not None:

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

        # Model signature
        input_example = X_train.head(3)

        example_predictions = model.predict(
            input_example
        )

        signature = infer_signature(
            input_example,
            example_predictions,
        )

        artifacts = save_artifacts(
            model_name=model_name,
            metrics=metrics,
            model=model,
            output_dir=output_dir,
        )

        mlflow.log_artifact(
            artifacts["report"]
        )

        mlflow.log_artifact(
            artifacts["confusion_matrix"]
        )

        mlflow.log_artifact(
            artifacts["feature_importance"]
        )

        if model_name == "XGBoost":

            mlflow.xgboost.log_model(
                model,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
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

        print(
            f"Accuracy : {metrics['accuracy']:.4f}"
        )

        print(
            f"Precision: {metrics['precision']:.4f}"
        )

        print(
            f"Recall   : {metrics['recall']:.4f}"
        )

        print(
            f"Macro F1 : {metrics['macro_f1']:.4f}"
        )

        return metrics["macro_f1"]


def train_models(
    X_train,
    y_train,
    X_test,
    y_test,
    train_rows,
    test_rows,
    output_dir,
):
    """Train and evaluate Decision Tree and XGBoost."""

    results = []



    decision_tree = DecisionTreeClassifier(
        max_depth=5,
        random_state=RANDOM_SEED,
    )

    decision_tree.fit(
        X_train,
        y_train,
    )

    dt_metrics = evaluate_model(
        decision_tree,
        X_test,
        y_test,
    )

    print(
        "\n" + "=" * 60
    )
    print("Decision Tree")
    print("=" * 60)

    print(
        f"Accuracy : {dt_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: {dt_metrics['precision']:.4f}"
    )

    print(
        f"Recall   : {dt_metrics['recall']:.4f}"
    )

    print(
        f"Macro F1 : {dt_metrics['macro_f1']:.4f}"
    )

    dt_f1 = log_model_run(
        model_name="Decision_Tree",
        model=decision_tree,
        metrics=dt_metrics,
        X_train=X_train,
        train_rows=train_rows,
        test_rows=test_rows,
        output_dir=output_dir,
    )

    results.append(
        {
            "model": "Decision Tree",
            "accuracy": dt_metrics["accuracy"],
            "precision": dt_metrics["precision"],
            "recall": dt_metrics["recall"],
            "f1": dt_f1,
        }
    )



    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax",
        num_class=3,
        eval_metric="mlogloss",
        random_state=RANDOM_SEED,
    )

    xgb_model.fit(
        X_train,
        y_train,
    )

    xgb_metrics = evaluate_model(
        xgb_model,
        X_test,
        y_test,
    )

    print(
        "\n" + "=" * 60
    )
    print("XGBoost")
    print("=" * 60)

    print(
        f"Accuracy : {xgb_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: {xgb_metrics['precision']:.4f}"
    )

    print(
        f"Recall   : {xgb_metrics['recall']:.4f}"
    )

    print(
        f"Macro F1 : {xgb_metrics['macro_f1']:.4f}"
    )

    xgb_f1 = log_model_run(
        model_name="XGBoost",
        model=xgb_model,
        metrics=xgb_metrics,
        X_train=X_train,
        train_rows=train_rows,
        test_rows=test_rows,
        output_dir=output_dir,
    )

    results.append(
        {
            "model": "XGBoost",
            "accuracy": xgb_metrics["accuracy"],
            "precision": xgb_metrics["precision"],
            "recall": xgb_metrics["recall"],
            "f1": xgb_f1,
        }
    )

    return results


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train Conditions/Environment "
            "classification models with MLflow"
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

    # MLflow configuration
    tracking_path = os.path.abspath(
        args.mlruns
    )

    mlflow.set_tracking_uri(
        f"file://{tracking_path}"
    )

    mlflow.set_experiment(
        "conditions_route_reliability"
    )

    # Load data
    print(
        "Loading processed datasets..."
    )

    train_df, test_df = load_data(
        args.train,
        args.test,
    )

    train_rows = len(train_df)
    test_rows = len(test_df)

    print(
        f"Training rows: {train_rows}"
    )

    print(
        f"Test rows: {test_rows}"
    )

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(
        train_df,
        test_df,
    )

    print("\nFeatures:")

    for feature in FEATURES:
        print(
            f"  - {feature}"
        )

    print(
        f"\nTarget: {TARGET}"
    )

    # Train models
    results = train_models(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        train_rows=train_rows,
        test_rows=test_rows,
        output_dir="reports/conditions",
    )

    # Model comparison
    print(
        "\n" + "=" * 60
    )
    print("MODEL COMPARISON")
    print("=" * 60)

    results_df = pd.DataFrame(
        results
    )

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    best_model = results_df.loc[
        results_df["f1"].idxmax()
    ]

    print(
        f"\nBest model based on Macro F1: "
        f"{best_model['model']}"
    )

    print(
        f"Best Macro F1: "
        f"{best_model['f1']:.4f}"
    )


if __name__ == "__main__":
    main()