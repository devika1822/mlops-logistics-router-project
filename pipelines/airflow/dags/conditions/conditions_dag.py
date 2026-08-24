from datetime import datetime

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount


import os

PROJECT_PATH = os.environ["PROJECT_PATH"]

RAW_DATA_PATH = (
    "/opt/project/data/raw/"
    "ecommerce_logistics_route_planning_dataset.csv"
)

PROCESSED_DATA_PATH = (
    "/opt/project/data/processed/conditions"
)

TRAIN_DATA_PATH = (
    "/opt/project/data/processed/conditions/train"
)

TEST_DATA_PATH = (
    "/opt/project/data/processed/conditions/test"
)


with DAG(
    dag_id="conditions_pipeline",
    description=(
        "Conditions track preprocessing "
        "and final model registration"
    ),
    start_date=datetime(2026, 8, 18),
    schedule=None,
    catchup=False,
    tags=["conditions"],
) as dag:

    spark_preprocessing = DockerOperator(
        task_id="spark_preprocessing",
        image="apache/spark:4.2.0-scala2.13-java21-python3-ubuntu",
        command=(
            "/opt/spark/bin/spark-submit "
            "--master local[*] "
            "/opt/project/pipelines/spark/conditions/"
            "spark_preprocessing.py "
            f"--input {RAW_DATA_PATH} "
            f"--output {PROCESSED_DATA_PATH} "
            "--test-size 0.2 "
            "--seed 42"
        ),
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        user="root",
        auto_remove=True,
        mount_tmp_dir=False,
        mounts=[
            Mount(
                source=PROJECT_PATH,
                target="/opt/project",
                type="bind",
            )
        ],
    )

    register_final_model = DockerOperator(
        task_id="register_final_model",
        image="conditions-trainer:latest",
        command=(
            "python "
            "/opt/project/tracks/conditions/register_model.py "
            f"--train {TRAIN_DATA_PATH} "
            f"--test {TEST_DATA_PATH}"
        ),
        docker_url="unix://var/run/docker.sock",
        network_mode="mlops-logistics-router-project_default",
        environment={
            "MLFLOW_TRACKING_URI": "http://mlflow:5000",
        },
        auto_remove=True,
        mount_tmp_dir=False,
        mounts=[
            Mount(
                source=PROJECT_PATH,
                target="/opt/project",
                type="bind",
            )
        ],
    )

    spark_preprocessing >> register_final_model