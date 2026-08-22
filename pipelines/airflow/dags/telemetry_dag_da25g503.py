import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_ROOT = os.environ.get(
    "MLOPS_PROJECT_ROOT", 
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

default_args = {
    "owner": "mlops_telemetry",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    "truck_telemetry_mlops_pipeline",
    default_args=default_args,
    description="Automated orchestration workflow for vehicle threat analytics",
    schedule_interval=None,
    catchup=False,
) as dag:

    clean_telemetry_task = BashOperator(
        task_id="pyspark_clean_telemetry",
        bash_command=f"python {os.path.join(PROJECT_ROOT, 'pipelines', 'spark', 'clean_spark.py')}",
    )

    train_models_task = BashOperator(
        task_id="mlflow_train_all_features",
        bash_command=f"python {os.path.join(PROJECT_ROOT, 'train_telemetry_da25g503.py')}",
    )

    clean_telemetry_task >> train_models_task
