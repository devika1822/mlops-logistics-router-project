from datetime import datetime

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount


import os

PROJECT_PATH = os.environ["PROJECT_PATH"]

with DAG(
    dag_id="geography_pipeline",
    description="Geography track preprocessing and model training pipeline",
    start_date=datetime(2026, 8, 18),
    schedule=None,
    catchup=False,
    tags=["geography"],
) as dag:

    spark_preprocessing = DockerOperator(
        task_id="spark_preprocessing",
        image="apache/spark:4.2.0-scala2.13-java21-python3-ubuntu",
        command=(
            "/opt/spark/bin/spark-submit "
            "--master local[*] "
            "/opt/project/pipelines/spark/geography/spark_preprocessing.py"
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

    train_final_model = DockerOperator(
        task_id="train_final_model",
        image="geography-trainer:latest",
        command="python /opt/project/tracks/geography/train_final.py",
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

    spark_preprocessing >> train_final_model