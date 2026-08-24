from datetime import datetime

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount


import os

PROJECT_PATH = os.environ["PROJECT_PATH"]


with DAG(
    dag_id="cost_pipeline",
    description="Cost track preprocessing and final model training pipeline",
    start_date=datetime(2026, 8, 18),
    schedule=None,
    catchup=False,
    tags=["cost"],
) as dag:

    spark_preprocessing = DockerOperator(
        task_id="spark_preprocessing",
        image="cost-spark:latest",
        command=(
            "/opt/spark/bin/spark-submit "
            "--master local[*] "
            "/opt/project/pipelines/spark/cost/spark_preprocessing.py"
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
        image="cost-trainer:latest",
        command=(
            "python "
            "/opt/project/tracks/cost/train.py"
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

    spark_preprocessing >> train_final_model