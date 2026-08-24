import json
from pathlib import Path

import pandas as pd
from kafka import KafkaConsumer


PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "payload_raw.parquet"
)

consumer = KafkaConsumer(
    "fleet-telemetry",
    bootstrap_servers="kafka:9092",
    group_id="payload_dag_consumer",
    auto_offset_reset="earliest",
    consumer_timeout_ms=15000,
    value_deserializer=lambda m: json.loads(
        m.decode("utf-8")
    ),
)

records = [
    msg.value
    for msg in consumer
    if msg.value.get("source") == "payload_track"
]

consumer.close()

if not records:
    raise RuntimeError(
        "No payload records received from Kafka."
    )

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

pd.DataFrame(records).to_parquet(
    OUTPUT_PATH,
    index=False,
)

print(
    f"[KAFKA] Saved {len(records)} records "
    f"to {OUTPUT_PATH}"
)