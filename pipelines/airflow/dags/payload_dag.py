"""
Cargo Payload Track – Airflow DAG

Pipeline stages:
  t1 → consume_kafka_batch  : Pull payload messages from Kafka → raw parquet
  t2 → spark_clean_payload  : Run Spark job to standardise & flag overloads
  t3 → train_payload_model  : Train RF + XGBoost, register best in MLflow

Schedule: every hour (@hourly)
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

default_args = {
    "owner":        "payload_track",
    "retries":      2,
    "retry_delay":  timedelta(minutes=5),
    "email_on_failure": False,
}

# Absolute paths inside the Airflow container (volumes mounted in docker-compose)
SPARK_SCRIPT = "/opt/airflow/spark/payload_cleaning.py"
TRAIN_SCRIPT = "/opt/airflow/tracks/cost/train.py"


# ---------------------------------------------------------------------------
# Task 1: Consume a batch from Kafka and persist as raw parquet
# ---------------------------------------------------------------------------
def consume_kafka_batch(**kwargs) -> None:
    import json
    import os
    import pandas as pd
    from kafka import KafkaConsumer

    consumer = KafkaConsumer(
        "fleet-telemetry",
        bootstrap_servers="kafka:9092",
        group_id="payload_dag_consumer",
        auto_offset_reset="earliest",
        consumer_timeout_ms=15_000,          # stop after 15 s of no messages
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )

    records = [
        msg.value for msg in consumer
        if msg.value.get("source") == "payload_track"
    ]
    consumer.close()

    if not records:
        print("[DAG] No new payload records in Kafka topic.")
        return

    os.makedirs("data/raw", exist_ok=True)
    out_path = "data/raw/payload_raw.parquet"
    pd.DataFrame(records).to_parquet(out_path, index=False)
    print(f"[DAG] Saved {len(records)} records → {out_path}")


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------
with DAG(
    dag_id="payload_pipeline",
    default_args=default_args,
    description="Cargo Payload Track: Kafka → Spark → MLflow",
    schedule_interval="@hourly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["payload", "cargo", "mlops"],
) as dag:

    t1_consume = PythonOperator(
        task_id="consume_kafka_batch",
        python_callable=consume_kafka_batch,
    )

    t2_clean = BashOperator(
        task_id="spark_clean_payload",
        bash_command=(
            f"spark-submit --master local[*] {SPARK_SCRIPT}"
        ),
    )

    t3_train = BashOperator(
        task_id="train_payload_model",
        bash_command=f"python {TRAIN_SCRIPT}",
        env={"MLFLOW_TRACKING_URI": "http://mlflow:5000"},
    )

    t1_consume >> t2_clean >> t3_train
