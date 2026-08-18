from datetime import datetime

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount


with DAG(
    dag_id="geography_pipeline",
    description="Geography track preprocessing pipeline",
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
                source=r"C:\Users\HP\mlops-logistics-router-project",
                target="/opt/project",
                type="bind",
            )
        ],
    )