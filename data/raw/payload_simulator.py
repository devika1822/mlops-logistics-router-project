"""
Cargo Payload Track – Kafka Producer / Data Simulator
Reads the Kaggle logistics CSV row-by-row and streams
cargo/payload-related fields to the 'fleet-telemetry' Kafka topic.

Usage:
    python payload_simulator.py --csv data/raw/logistics_dataset.csv
"""

import csv
import json
import time
import argparse
from kafka import KafkaProducer

# Actual columns from ecommerce_logistics.csv belonging to the Cargo Payload track
PAYLOAD_FIELDS = [
    "order_weight_kg",          # cargo weight (kg) – renamed/standardised by Spark
    "vehicle_capacity_kg",      # vehicle's rated max capacity (kg)
    "vehicle_utilization_ratio",# load ratio 0-1; Spark converts to capacity_pct
    "delivery_time_window_hrs", # hours available for delivery
    "distance_km",              # trip distance
    "traffic_density_index",    # 0-1 traffic stress (used to derive engine wear)
    "weather_impact_index",     # 0-1 weather stress (used to derive engine wear)
]


def build_producer(broker: str) -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=broker,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks="all",
        retries=3,
    )


def stream_csv(csv_path: str, broker: str, topic: str, delay: float) -> None:
    producer = build_producer(broker)

    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        sent = 0
        for row in reader:
            record = {k: row[k] for k in PAYLOAD_FIELDS if k in row}
            record["source"] = "payload_track"
            # Skip rows that are missing any required field
            if len(record) < len(PAYLOAD_FIELDS):
                continue

            producer.send(topic, value=record)
            sent += 1
            print(f"[PRODUCER] #{sent:>5} Sent: {record}")
            time.sleep(delay)

    producer.flush()
    print(f"\n[PRODUCER] Done. Total records sent: {sent}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Payload track Kafka producer")
    parser.add_argument("--csv",    default="data/raw/ecommerce_logistics.csv",
                        help="Path to the Kaggle CSV dataset")
    parser.add_argument("--broker", default="localhost:29092",
                        help="Kafka broker address")
    parser.add_argument("--topic",  default="fleet-telemetry",
                        help="Kafka topic name")
    parser.add_argument("--delay",  type=float, default=0.5,
                        help="Seconds between each message (simulates live stream)")
    args = parser.parse_args()

    stream_csv(args.csv, args.broker, args.topic, args.delay)
