# RouteSense: MLOps-Driven Predictive Analytics for Logistics Delivery Performance

## 1. Project Overview

This project implements an end-to-end MLOps system for fleet logistics and route planning. The system is organized into multiple machine learning tracks that analyze different aspects of logistics operations and expose trained models through independent FastAPI services.

The project integrates data preprocessing, model development, experiment tracking, model registration, API-based inference, workflow orchestration, containerization, streaming infrastructure, and model monitoring into a common repository.

Dataset: [E-commerce Logistics Route Planning Dataset](https://www.kaggle.com/datasets/zara2099/e-commerce-logistics-route-planning-dataset)
(1,000 rows, 18 columns).

The major prediction tracks are:

| Track | Purpose |
|---|---|
| Geography | Predict optimized route/travel time using geographical and traffic-related features |
| Vehicle / Telemetry | Analyze vehicle telemetry and predict vehicle-related operational metrics |
| Cost | Predict optimized route cost |
| Conditions | Predict route reliability based on weather, traffic, speed, and time-related conditions |

The tracks are developed independently but integrated through a shared MLOps architecture using **Apache Spark, MLflow, FastAPI, Apache Airflow, Docker Compose, and Kafka**.

---

## 2. System Architecture

The project follows a modular MLOps architecture.

### Processing Flow

1. Raw logistics and telemetry data are ingested from the `data/raw` directory.
2. Apache Spark performs distributed preprocessing and feature preparation.
3. Processed datasets are stored under `data/processed`.
4. Each track performs model experimentation, tuning, and final model training.
5. MLflow records model parameters, metrics, artifacts, and registered model versions.
6. FastAPI services expose trained models through REST prediction endpoints.
7. Docker packages the individual services and their dependencies into reproducible containers.
8. Docker Compose provides a common environment for MLflow, APIs, Airflow, Kafka, and supporting services.
9. Apache Airflow orchestrates pipeline stages through track-specific DAGs.
10. Monitoring components can collect prediction information and generate model monitoring reports.

The modular design allows each machine learning track to be developed and tested independently while still operating as part of a common logistics MLOps platform.

---

## 3. System Architecture Diagram

```text
                                                 +-----------------------+
                         |    Raw Data Sources   |
                         +-----------+-----------+
                                     |
                                     v
                         +-----------------------+
                         |    Apache Spark       |
                         | Data Preprocessing    |
                         +-----------+-----------+
                                     |
                                     v
                         +-----------------------+
                         |   Processed Datasets  |
                         +-----------+-----------+
                                     |
        +-------------+--------------+--------------+-------------+
        |             |              |              |
        v             v              v              v
+---------------+ +---------------+ +---------------+ +---------------+
|   Geography   | |   Vehicle /   | |     Cost      | |  Conditions   |
|     Track     | |   Telemetry   | |     Track     | |     Track     |
+-------+-------+ +-------+-------+ +-------+-------+ +-------+-------+
        |                 |                 |                 |
        +-----------------+-----------------+-----------------+
                                  |
                                  v
                         +-----------------------+
                         |        MLflow         |
                         | Experiments / Models  |
                         |   Model Registry      |
                         +-----------+-----------+
                                     |
        +-------------+--------------+--------------+-------------+
        |             |              |              |
        v             v              v              v
+---------------+ +---------------+ +---------------+ +---------------+
| Geography API | | Vehicle API   | |   Cost API    | | Conditions API|
|   FastAPI     | |   FastAPI     | |   FastAPI     | |   FastAPI     |
+---------------+ +---------------+ +---------------+ +---------------+
                                     |
                                     v
                         +-----------------------+
                         |    Docker Compose     |
                         | Integrated Execution  |
                         +-----------------------+

      Apache Airflow → Pipeline orchestration
      Apache Kafka  → Streaming infrastructure
      Monitoring    → Drift / prediction monitoring
```

---

## 4. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Distributed Processing | Apache Spark / PySpark |
| Machine Learning | Scikit-learn / track-specific ML libraries |
| Experiment Tracking | MLflow |
| Model Registry | MLflow Model Registry |
| API Layer | FastAPI |
| API Server | Uvicorn |
| Workflow Orchestration | Apache Airflow |
| Streaming | Apache Kafka |
| Containerization | Docker |
| Multi-service Execution | Docker Compose |
| Data Versioning | DVC |
| Data Processing | Pandas / NumPy |
| Model Monitoring | Track-specific monitoring scripts |
| Version Control | Git / GitHub |

---

## 5. Repository Structure

```text
mlops-logistics-router-project/
│
├── api/
│   ├── main.py
│   └── routers/
│       ├── geography.py
│       ├── conditions.py
│       ├── app_cost_da25g503.py
│       └── payload.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docker/
│   ├── geography/
│   │   └── Dockerfile
│   ├── vehicle/
│   │   └── Dockerfile
│   ├── cost/
│   │   └── Dockerfile
│   └── conditions/
│       └── Dockerfile
│
├── models/
│   └── mlflow/
│       ├── mlflow.db
│       └── artifacts/
│
├── monitoring/
│   └── geography/
│
├── pipelines/
│   ├── airflow/
│   │   └── dags/
│   │       ├── geography/
│   │       ├── cost/
│   │       ├── vehicle/
│   │       └── conditions/
│   │
│   └── spark/
│       ├── geography/
│       ├── cost/
│       ├── payload_cleaning/
│       └── conditions/
│
├── tracks/
│   ├── geography/
│   ├── vehicle/
│   ├── cost/
│   └── conditions/
│
├── reports/
├── docs/
├── .dvc/
├── .dockerignore
├── .gitignore
├── docker-compose.yml
├── requirements.txt
└── README.md
```

The exact contents of individual track directories may vary because each track implements its own preprocessing, training, tuning, inference, and monitoring requirements.

---

## 6. Setup and Installation

### Prerequisites

The following software should be installed before running the project:

- Python 3.12 or compatible Python version
- Git
- Docker Desktop
- Docker Compose
- Java JDK for Apache Spark
- Apache Spark / PySpark dependencies
- DVC

### Clone the Repository

```bash
git clone <repository-url>
cd mlops-logistics-router-project
```

### Create a Python Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Retrieve Versioned Data

Where the raw dataset is managed using DVC:

```bash
dvc pull
```

---

## 7. Running the MLOps Pipeline

The recommended execution order is:

```text
Raw Data
   ↓
Spark Preprocessing
   ↓
Processed Data
   ↓
Model Experimentation
   ↓
Hyperparameter Tuning
   ↓
Final Model Training
   ↓
MLflow Logging / Model Registration
   ↓
FastAPI Model Serving
   ↓
Dockerized Services
   ↓
Monitoring
```

### Step 1 – Run Spark Preprocessing

Run the Spark preprocessing script corresponding to the required track.

Example for Geography:

```bash
python pipelines/spark/geography/spark_preprocessing.py
```

The preprocessing stage generates the processed data required by downstream model training and inference components.

Other tracks should execute their corresponding preprocessing scripts under:

```text
pipelines/spark/<track>/
```

### Step 2 – Run Model Training

Each track contains its own model development scripts under:

```text
tracks/<track>/
```

Depending on the track, these may include:

```text
model_experiment.py
model_tuning.py
train.py
train_final.py
```

The general model development sequence is:

```text
Model Experimentation
        ↓
Model Selection
        ↓
Hyperparameter Tuning
        ↓
Final Training
        ↓
MLflow Registration
```

### Step 3 – Start MLflow

MLflow can be started through Docker Compose:

```bash
docker compose up -d mlflow
```

The MLflow UI is available at:

```text
http://localhost:5000
```

MLflow is used for:

- experiment tracking
- parameter logging
- metric logging
- artifact storage
- model versioning
- model registration

### Step 4 – Run FastAPI Services

Once the required models have been trained and registered, the APIs can be started using Docker Compose.

---

## 8. API Usage

The integrated system exposes separate APIs for the machine learning tracks.

| Service | Host Port | Purpose |
|---|---:|---|
| Geography API | 8001 | Geography / route-time prediction |
| Vehicle / Telemetry API | 8002 | Vehicle telemetry prediction |
| Cost API | 8003 | Route-cost prediction |
| Conditions API | 8004 | Route-reliability prediction |

FastAPI automatically provides interactive Swagger documentation.

Examples:

```text
http://localhost:8001/docs
http://localhost:8002/docs
http://localhost:8003/docs
http://localhost:8004/docs
```

Prediction requests are submitted as JSON to the prediction endpoint exposed by the respective service.

Example request structure:

```json
{
  "feature_1": 10.5,
  "feature_2": 0.4,
  "feature_3": 25
}
```

The exact input features and validation constraints depend on the selected prediction track and can be viewed through its `/docs` endpoint.

Example response structure:

```json
{
  "status": "success",
  "prediction": 42.5
}
```

---

## 9. Docker Execution

Docker is used to provide reproducible execution environments for the project services.

### Validate Docker Compose Configuration

```bash
docker compose config
```

### Build Individual APIs

```bash
docker compose build geography-api
docker compose build telemetry-api
docker compose build cost-api
docker compose build conditions-api
```

### Build Multiple APIs

```bash
docker compose build geography-api telemetry-api cost-api conditions-api
```

### Start MLflow and APIs

```bash
docker compose up -d mlflow geography-api telemetry-api cost-api conditions-api
```

### Check Running Containers

```bash
docker compose ps
```

### View Service Logs

```bash
docker compose logs geography-api
docker compose logs telemetry-api
docker compose logs cost-api
docker compose logs conditions-api
```

### Stop Services

```bash
docker compose down
```

### Rebuild a Service After Code Changes

```bash
docker compose build --no-cache geography-api
docker compose up -d --force-recreate geography-api
```

---

## 10. Airflow Orchestration

Apache Airflow is used to orchestrate pipeline tasks and track dependencies between stages.

Track-specific DAGs are maintained under:

```text
pipelines/airflow/dags/
```

The Airflow environment includes:

- PostgreSQL metadata database
- Airflow webserver
- Airflow scheduler

Start the Airflow services using:

```bash
docker compose up -d airflow-db airflow-init airflow-webserver airflow-scheduler
```

The Airflow UI is available at:

```text
http://localhost:8080
```

The DAGs coordinate pipeline stages such as preprocessing, training, validation, and other track-specific workflow operations.

---

## 11. Kafka Streaming

Apache Kafka is included as the streaming component of the MLOps architecture and can be used to simulate or process continuously arriving logistics and telemetry events.

Start Kafka and ZooKeeper using:

```bash
docker compose up -d zookeeper kafka
```

The Docker Compose environment exposes Kafka for communication between streaming producers, consumers, and downstream pipeline components.

---

## 12. MLflow Model Management

MLflow provides centralized experiment tracking and model management across the different project tracks.

The shared MLflow service is configured through `docker-compose.yml`.

Model artifacts and metadata are maintained under:

```text
models/mlflow/
```

Typical tracked information includes:

- model type
- hyperparameters
- evaluation metrics
- training runs
- model artifacts
- model versions
- registered production candidates

This enables the API layer to load registered models instead of relying only on manually stored model files.

---

## 13. Monitoring

Monitoring components are maintained under:

```text
monitoring/
```

Monitoring scripts can be used to:

- generate test prediction traffic
- capture inference information
- analyze prediction distributions
- produce monitoring reports
- identify potential changes in model behavior or input data

Monitoring is maintained independently for tracks where monitoring functionality has been implemented.

---

## 14. Dependencies

The major Python dependencies include:

```text
fastapi
uvicorn
mlflow
pandas
numpy
scikit-learn
pyspark
pyarrow
apache-airflow
```

Additional dependencies may be required by individual tracks, for example XGBoost or monitoring libraries.

For the complete dependency specification, refer to:

```text
requirements.txt
```

Docker images additionally provide isolated runtime dependencies for individual services.

---

## 15. Integrated Execution

After preprocessing and model training have been completed for all required tracks, the integrated environment can be started using Docker Compose.

```bash
docker compose up -d
```

The expected integrated environment consists of:

```text
Apache Spark        → distributed preprocessing
Apache Airflow      → workflow orchestration
MLflow              → experiment tracking and model registry
FastAPI             → prediction services
Docker              → service containerization
Docker Compose      → multi-service integration
Apache Kafka        → streaming infrastructure
Monitoring          → model/prediction monitoring
```

The final system therefore demonstrates the complete MLOps lifecycle from raw data processing and model development through model registration, deployment, orchestration, and monitoring.

---

## 16. Service Endpoints

After successful deployment, the primary local interfaces are:

| Component | URL |
|---|---|
| MLflow UI | `http://localhost:5000` |
| Airflow UI | `http://localhost:8080` |
| Geography API Docs | `http://localhost:8001/docs` |
| Vehicle / Telemetry API Docs | `http://localhost:8002/docs` |
| Cost API Docs | `http://localhost:8003/docs` |
| Conditions API Docs | `http://localhost:8004/docs` |

> **Note:** Processed datasets and registered MLflow models must be available before model-serving APIs that depend on them can start successfully.
- **Row key:** dataframe index → `order_id`
- **API response schema:** `{"order_id": int, "track": str, "prediction": str, "confidence": float}`
- **Docker base image:** `python:3.11-slim`
