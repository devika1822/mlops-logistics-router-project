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
    "average_speed_kmph",
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

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5002",
)

MLFLOW_EXPERIMENT = (
    "conditions_route_reliability_final"
)



FINAL_MODEL_PARAMS = {
    "n_estimators": 75,
    "max_depth": 4,
    "learning_rate": 0.125,
    "subsample": 0.7,
    "colsample_bytree": 1.0,
    "min_child_weight": 1,
    "gamma": 0.2,
    "reg_alpha": 0.01,
    "reg_lambda": 1.0,
}


EXPECTED_MIN_F1 = 0.95
EXPECTED_REFERENCE_F1 = 0.9629




def load_data(train_path, test_path):


    train_df = pd.read_parquet(
        train_path
    )

    test_df = pd.read_parquet(
        test_path
    )

    return train_df, test_df


def prepare_data(train_df, test_df):


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

    y_train = train_df[TARGET].map(
        LABEL_MAPPING
    )

    y_test = test_df[TARGET].map(
        LABEL_MAPPING
    )

    if y_train.isna().any():
        raise ValueError(
            "Unknown or missing target class "
            "in training data."
        )

    if y_test.isna().any():
        raise ValueError(
            "Unknown or missing target class "
            "in test data."
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
        **FINAL_MODEL_PARAMS,
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

    os.makedirs(
        output_dir,
        exist_ok=True,
    )



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

    for parameter, value in (
        FINAL_MODEL_PARAMS.items()
    ):
        print(
            f"  {parameter}: {value}"
        )



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



    print(
        f"\nExpected minimum Macro F1: "
        f"{EXPECTED_MIN_F1:.4f}"
    )

    print(
        f"Reference Macro F1: "
        f"{EXPECTED_REFERENCE_F1:.4f}"
    )

    if (
        metrics["macro_f1"]
        < EXPECTED_MIN_F1
    ):
        raise ValueError(
            f"Reproduced model F1 "
            f"{metrics['macro_f1']:.4f} "
            f"is below threshold "
            f"{EXPECTED_MIN_F1:.4f}. "
            "Registration aborted."
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
            "random_seed",
            RANDOM_SEED,
        )


        mlflow.log_param(
            "model_selection",
            "V2 localized tuning",
        )

        mlflow.log_param(
            "selection_metric",
            "test_macro_f1",
        )

        mlflow.log_param(
            "cv_folds",
            5,
        )

        mlflow.log_metric(
            "reference_macro_f1",
            EXPECTED_REFERENCE_F1,
        )

        mlflow.set_tag(
            "final_model_status",
            "selected",
        )



        for parameter, value in (
            FINAL_MODEL_PARAMS.items()
        ):
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



        input_example = X_train.head(3)

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
                model_format="json",
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
            f"Run URI: "
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

        if not registered_versions:
            raise RuntimeError(
                "Model was logged, but no registered "
                "model versions were found."
            )

        registered_version = max(
            registered_versions,
            key=lambda version: int(
                version.version
            ),
        )

        version_number = str(
            registered_version.version
        )

  

        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="champion",
            version=version_number,
        )

        print(
            f"\nAlias 'champion' -> "
            f"version {version_number}"
        )

        print(
            f"Registry URI: "
            f"models:/{MODEL_NAME}@champion"
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


if __name__ == "__main__":
    main()