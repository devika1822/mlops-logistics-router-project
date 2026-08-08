# E-Commerce Logistics — Delivery Efficiency Prediction (MLOps Course Project)

## Project Overview
Predicts `delivery_efficiency_score` (bucketed: Low / Medium / High) for e-commerce
delivery orders, using an end-to-end MLOps pipeline: Airflow-orchestrated ingestion,
Spark-based cleaning, MLflow experiment tracking, DVC dataset/model versioning,
FastAPI + Docker deployment, and monitoring/drift detection.

Dataset: [E-commerce Logistics Route Planning Dataset](https://www.kaggle.com/datasets/zara2099/e-commerce-logistics-route-planning-dataset)
(1,000 rows, 18 columns).

## Team & Ownership
| Track | Owner | Feature columns |
|---|---|---|
| Geography & Route | _TBD_ | order_latitude, order_longitude, distance_km, delivery_time_window_hrs, order_priority |
| Vehicle & Load | _TBD_ | vehicle_capacity_kg, order_weight_kg, vehicle_utilization_ratio |
| Traffic & Environment | _TBD_ | traffic_density_index, average_speed_kmph, weather_impact_index, time_of_day |
| Cost & Outcome | _TBD_ | fuel_cost_per_km, driver_cost_per_hour, optimized_route_time_min, optimized_route_cost |

Shared target: `delivery_efficiency_score` (all tracks predict this).
Join key: dataframe row index, used as `order_id` (dataset has no native ID column).

## Repo Structure
```
data/raw/              # original Kaggle CSV, DVC-tracked
data/processed/        # cleaned per-track and merged feature tables
tracks/<name>/         # each person's Spark cleaning + practice model notebook/script
pipelines/airflow/     # DAGs for ingestion + retraining
pipelines/spark/       # Spark cleaning scripts per track
models/mlflow/         # MLflow tracking config / mlruns (gitignored, DVC or MLflow server instead)
api/                    # FastAPI app; api/routers/ has one router per track
docker/                # Dockerfile(s), docker-compose.yml
monitoring/             # drift detection / logging scripts
.github/workflows/      # CI/CD pipeline
docs/                   # architecture diagram, setup notes
reports/                # final 5-10 page technical report
```

## Setup
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Running the pipeline
_TBD as components are built — Airflow DAG trigger, Spark job commands, FastAPI run command, Docker build/run commands go here._

## API Usage
_TBD — document each track's endpoint (/api/v1/geography, /api/v1/vehicle, /api/v1/conditions, /api/v1/cost) once built._

## Contract (agreed Day 1 — do not change without team sync)
- **Target column:** `delivery_efficiency_score`, bucketed into `Low` / `Medium` / `High`
  (bucket boundaries: _TBD, agree on Day 1 based on the actual distribution_)
- **Row key:** dataframe index → `order_id`
- **API response schema:** `{"order_id": int, "track": str, "prediction": str, "confidence": float}`
- **Docker base image:** `python:3.11-slim`
