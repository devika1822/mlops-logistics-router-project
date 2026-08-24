from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount


import os

PROJECT_PATH = os.environ["PROJECT_PATH"]

default_args = {
    "owner": "payload_track",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "email_on_failure": False,
}


with DAG(
    dag_id="payload_pipeline",
    default_args=default_args,
    description="Payload track: Kafka -> Spark -> MLflow",
    start_date=datetime(2026, 8, 18),
    schedule=None,
    catchup=False,
    tags=["vehicle", "payload", "mlops"],
) as dag:

    consume_kafka_batch = DockerOperator(
        task_id="consume_kafka_batch",
        image="vehicle-pipeline:latest",
        command=(
            "python "
            "/opt/project/pipelines/airflow/scripts/"
            "consume_payload_kafka.py"
        ),
        docker_url="unix://var/run/docker.sock",
        network_mode="mlops-logistics-router-project_default",
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

    spark_clean_payload = DockerOperator(
        task_id="spark_clean_payload",
        image="apache/spark:4.2.0-scala2.13-java21-python3-ubuntu",
        command=(
            "bash -c 'cd /opt/project && "
            "/opt/spark/bin/spark-submit "
            "--master local[*] "
            "/opt/project/pipelines/spark/"
            "payload_cleaning.py'"
        ),
        docker_url="unix://var/run/docker.sock",
        network_mode="mlops-logistics-router-project_default",
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

    train_payload_model = DockerOperator(
        task_id="train_payload_model",
        image="vehicle-pipeline:latest",
        command=(
            "python "
            "/opt/project/tracks/vehicle/train.py"
        ),
        docker_url="unix://var/run/docker.sock",
        network_mode="mlops-logistics-router-project_default",
        environment={
            "MLFLOW_TRACKING_URI": "http://mlflow:5000",
            "PAYLOAD_DATA_PATH": (
                "/opt/project/data/processed/"
                "payload_clean.parquet"
            ),
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

    (
        consume_kafka_batch
        >> spark_clean_payload
        >> train_payload_model
    )