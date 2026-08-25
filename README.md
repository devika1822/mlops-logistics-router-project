# RouteSense: MLOps-Driven Predictive Analytics for Logistics Delivery Performance

### Project Report

[📄 View Final Project Report](doc/MLOPS_RouteSense_Report.pdf)

### Configure Environment Variables

Create a local `.env` file from the provided `.env.example` template.

**Windows PowerShell:**

```powershell
Copy-Item .env.example .env
```

**Linux / macOS:**

```bash
cp .env.example .env
```

Open `.env` and configure the required values for your local environment:

```env
PROJECT_PATH=C:/path/to/mlops-logistics-router-project
DOCKER_NETWORK=mlops-logistics-router-project_default
```

- `PROJECT_PATH` — absolute path to the cloned project on the host machine. Airflow uses this to mount the project into DockerOperator task containers.
- `DOCKER_NETWORK` — Docker Compose network used by Airflow-launched containers to communicate with services such as MLflow.

The `.env` file contains machine-specific configuration and should not be committed to Git.

## 1. Project Overview

RouteSense is an end-to-end MLOps system for fleet logistics and route-planning analytics. The project is organized into four independent machine-learning tracks, each addressing a different aspect of logistics operations.

| Track | Purpose |
|---|---|
| **Geography** | Predict optimized route/travel time using geographical and traffic-related features |
| **Conditions** | Predict route reliability based on weather, traffic, speed, and time-related conditions |
| **Cost** | Predict optimized route cost |
| **Vehicle / Telemetry** | Analyze vehicle/payload characteristics and predict vehicle-related operational metrics |

Each track maintains its own preprocessing, model-development and serving components while sharing common MLOps infrastructure.


The project integrates:

- distributed preprocessing with Apache Spark;
- model experimentation and tuning;
- MLflow experiment tracking and model registration;
- workflow orchestration with Apache Airflow;
- model serving through FastAPI;
- multi-model integration through a unified API;
- containerization with Docker and Docker Compose;
- Kafka-based streaming infrastructure;
- prediction and drift monitoring; and
- CI validation using GitHub Actions.

**Dataset:** E-commerce Logistics Route Planning Dataset — 1,000 rows and 18 columns.

---

## 2. System Architecture

The four prediction tracks operate **in parallel**. Each track performs its own preprocessing and model-development workflow before exposing its trained model through an independent API.

```text
                                                  Raw Logistics Data
                                |
                                v
                      +--------------------+
                      | Apache Spark       |
                      | Preprocessing      |
                      +---------+----------+
                                |
                                v
                       Processed Datasets
                                |
        +-----------------------+-----------------------+
        |                       |                       |
        v                       v                       v
 +-------------+         +-------------+         +-------------+
 |  Geography  |         | Conditions  |         |    Cost     |
 |    Track    |         |    Track    |         |    Track    |
 +------+------+         +------+------+         +------+------+
        |                       |                       |
        |                       |                       |
        |                 +-----+-----------------------+
        |                 |
        |                 |             +-----------------+
        |                 |             | Vehicle/Payload |
        |                 |             |      Track      |
        |                 |             +--------+--------+
        |                 |                      |
        +-----------------+----------------------+
                          |
                          v
                Model Training / Tuning
                          |
                          v
                +---------------------+
                |       MLflow        |
                | Experiment Tracking |
                |   Model Registry    |
                +----------+----------+
                           |
       +-------------------+-------------------+-------------------+
       |                   |                   |                   |
       v                   v                   v                   v
+---------------+   +---------------+   +---------------+   +---------------+
| Geography API |   | Conditions API|   |    Cost API   |   | Vehicle API   |
|    FastAPI    |   |    FastAPI    |   |    FastAPI    |   |    FastAPI    |
+-------+-------+   +-------+-------+   +-------+-------+   +-------+-------+
        |                   |                   |                   |
        +-------------------+-------------------+-------------------+
                                    |
                                    v
                         +----------------------+
                         |   Integration API    |
                         | Unified REST Layer   |
                         +----------------------+

       Airflow   → Pipeline orchestration
       Docker    → Containerized execution
       Kafka     → Vehicle/Payload streaming ingestion
       Evidently → Data-drift monitoring

```

> The four tracks are logically parallel. Their exact preprocessing, training and monitoring implementations differ according to the requirements of each track.

---

## 3. Processing Flow

The general MLOps lifecycle is:

```text
Raw / Streaming Data
        |
        v
Data Validation & Preprocessing
        |
        v
Feature Engineering
        |
        v
Model Experimentation
        |
        v
Model Comparison / Tuning
        |
        v
Final Model Training
        |
        v
MLflow Tracking & Model Registration
        |
        v
FastAPI Model Serving
        |
        v
Integrated Logistics API
        |
        v
Prediction / Drift Monitoring
```

Apache Airflow orchestrates track-specific production workflows, while Docker provides reproducible runtime environments.

---

## 4. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python 3.12 |
| Distributed Processing | Apache Spark / PySpark |
| Machine Learning | Scikit-learn / track-specific ML libraries |
| Data Processing | Pandas / NumPy |
| Experiment Tracking | MLflow |
| Model Registry | MLflow Model Registry |
| API Framework | FastAPI |
| API Server | Uvicorn |
| Workflow Orchestration | Apache Airflow |
| Streaming | Apache Kafka / ZooKeeper |
| Containerization | Docker |
| Multi-service Execution | Docker Compose |
| Monitoring | Evidently / track-specific monitoring |
| Data Versioning | DVC |
| Version Control | Git / GitHub |
| Continuous Integration | GitHub Actions |

---

## 5. Repository Structure

```text
mlops-logistics-router-project/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── api/
│   ├── integration.py
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
│   ├── conditions/
│   ├── cost/
│   ├── vehicle/
│   └── integration/
│
├── models/
│   └── mlflow/
│
├── monitoring/
│
├── pipelines/
│   ├── airflow/
│   │   └── dags/
│   └── spark/
│
├── tracks/
│   ├── geography/
│   ├── conditions/
│   ├── cost/
│   └── vehicle/
│
├── reports/
├── docs/
├── .dvc/
├── .dockerignore
├── .dvcignore
├── .env.example
├── .gitignore
├── docker-compose.yml
├── requirements.txt
└── README.md
```

Each track maintains its own implementation according to its preprocessing, experimentation, tuning, training, inference and monitoring requirements.

---

## 6. Setup and Installation

### Prerequisites

Install the following before running the project:

- Python 3.12
- Git
- Docker Desktop
- Docker Compose
- Java JDK
- DVC

### Clone the Repository

```bash
git clone -b integration/final https://github.com/devika1822/mlops-logistics-router-project.git
cd mlops-logistics-router-project
```

### Create a Virtual Environment

#### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Retrieve Versioned Data

Where required, retrieve DVC-managed data using:

```bash
dvc pull
```

---

## 7. Running the Complete Environment

The complete multi-service environment is defined in `docker-compose.yml`.

### Validate Docker Compose

```bash
docker compose config
```

### Start the Complete Stack

```bash
docker compose up -d
```

### Check Running Containers

```bash
docker compose ps
```

### Rebuild After Code Changes

```bash
docker compose up -d --build
```

### Stop the Environment

```bash
docker compose down
```

The Compose environment includes:

- ZooKeeper
- Kafka
- Airflow PostgreSQL database
- Airflow initialization service
- Airflow webserver
- Airflow scheduler
- MLflow
- Geography API
- Conditions API
- Cost API
- Vehicle / Telemetry API
- Integration API
- track-specific training and pipeline images

---

## 8. Service Endpoints

After successful startup:

| Service | Endpoint |
|---|---|
| **Integration API** | `http://localhost:8000/docs` |
| **Geography API** | `http://localhost:8001/docs` |
| **Vehicle / Telemetry API** | `http://localhost:8002/docs` |
| **Cost API** | `http://localhost:8003/docs` |
| **Conditions API** | `http://localhost:8004/docs` |
| **MLflow UI** | `http://localhost:5000` |
| **Airflow UI** | `http://localhost:8080` |

FastAPI automatically exposes interactive Swagger documentation through the `/docs` endpoint.

> Processed datasets and required registered models must be available for model-serving APIs that depend on them.

---

## 9. Geography Track

### Objective

The Geography track predicts:

```text
optimized_route_time_min
```

using geographical, delivery and traffic-related information.

The original candidate features included:

```text
order_latitude
order_longitude
distance_km
delivery_time_window_hrs
order_priority
traffic_density_index
average_speed_kmph
```

### Leakage Analysis

Exploratory analysis identified `average_speed_kmph` as a potential target-leakage source.

A domain-derived route-time relationship combining:

```text
distance_km
average_speed_kmph
traffic_density_index
```

was found to have an almost perfect correlation with the target.

When average speed was retained, model performance became extremely high. Random Forest, for example, achieved an R² of approximately **0.98**.

To avoid an unrealistically optimistic model that could effectively reconstruct route time, `average_speed_kmph` was excluded from the final Geography feature set.

The final base features are:

```text
order_latitude
order_longitude
distance_km
delivery_time_window_hrs
order_priority
traffic_density_index
```

---

## 10. Geography Spark Preprocessing

Geography preprocessing is implemented using PySpark.

```bash
python pipelines/spark/geography/spark_preprocessing.py
```

The preprocessing stage performs:

- required-column validation;
- missing-value checks;
- removal of rows containing nulls in required fields;
- duplicate handling;
- latitude and longitude range validation;
- positive distance validation;
- positive delivery-window validation;
- traffic-density validation;
- order-priority validation;
- positive target validation;
- IQR-based outlier detection; and
- feature engineering.

Statistical outliers are **flagged instead of automatically deleted**, since unusual long-distance or long-duration deliveries may still be valid logistics observations.

### Engineered Features

#### Distance-Traffic Interaction

```text
distance_traffic_interaction
    = distance_km × traffic_density_index
```

This represents the interaction between route length and traffic conditions.

#### Distance per Delivery Window

```text
distance_per_delivery_window
    = distance_km / delivery_time_window_hrs
```

This represents distance pressure relative to the promised delivery window.

#### Distance Outlier Flag

An IQR-based binary feature identifies unusually short or long delivery distances while retaining those records.

Processed Geography data is stored under:

```text
data/processed/geography/
```

---

## 11. Geography Model Development

The Geography track evaluates two regression algorithms:

- Linear Regression
- Random Forest Regressor

Both base and engineered feature sets are evaluated during experimentation.

After removing the leakage-prone average-speed feature, the initial comparison produced approximately:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 34.72 | 51.55 | 0.460 |
| Random Forest | 34.73 | 51.42 | 0.463 |

Random Forest is subsequently subjected to hyperparameter tuning, while both model candidates are evaluated using **5-fold cross-validation**.

The tuning process evaluates Random Forest parameters including:

```text
n_estimators
max_depth
min_samples_split
min_samples_leaf
```

The final production model selected for the Geography track is:

```text
Linear Regression + StandardScaler
```

The selected model is validated using 5-fold cross-validation and then fitted using the complete processed Geography dataset.

---

## 12. MLflow Experiment Tracking and Model Registry

MLflow provides centralized experiment tracking and model management.

Start MLflow independently using:

```bash
docker compose up -d mlflow
```

The UI is available at:

```text
http://localhost:5000
```

MLflow tracks information such as:

- model type;
- feature set;
- hyperparameters;
- evaluation metrics;
- cross-validation results;
- model artifacts; and
- registered model versions.

The final Geography model is registered as:

```text
geography_route_time_model
```

The Geography FastAPI service loads this registered model from MLflow for inference.

MLflow metadata and artifacts are persisted through:

```text
./models/mlflow:/mlflow
```

The MLflow service uses port `5000` both inside the Docker network and on the host.

---

## 13. Airflow Orchestration

Apache Airflow orchestrates track-specific production workflows.

The Airflow environment consists of:

- PostgreSQL metadata database;
- initialization service;
- Airflow webserver; and
- Airflow scheduler.

Start Airflow using:

```bash
docker compose up -d airflow-db airflow-init airflow-webserver airflow-scheduler
```

Airflow UI:

```text
http://localhost:8080
```

Track-specific DAGs are maintained under:

```text
pipelines/airflow/dags/
```

### Geography Pipeline

The Geography production workflow is:

```text
Spark Preprocessing
        |
        v
Final Model Training
        |
        v
MLflow Logging / Registration
```

The preprocessing stage is executed in a Spark container.

The final Geography training stage uses:

```text
geography-trainer:latest
```

Airflow uses `DockerOperator` to execute pipeline tasks in isolated containers.

---

## 14. Geography Model Serving

The Geography model is exposed through a FastAPI service.

The prediction request accepts six base inputs:

```text
order_latitude
order_longitude
distance_km
delivery_time_window_hrs
order_priority
traffic_density_index
```

Pydantic validates the request types.

The API recreates the engineered features required by the trained model:

```text
distance_traffic_interaction
distance_per_delivery_window
distance_outlier_flag
```

The registered model is loaded from MLflow and used to generate the final prediction.

Example response:

```json
{
  "predicted_route_time_min": 123.45
}
```

The API also logs prediction inputs, the predicted route time and a timestamp for downstream monitoring.

---

## 15. Integrated Logistics Router API

The Integration API provides a single entry point for all four prediction tracks.

Swagger UI:

```text
http://localhost:8000/docs
```

Main endpoint:

```text
POST /route-analysis
```

The integrated API:

1. receives one combined logistics request;
2. validates it using Pydantic;
3. creates a Geography-specific payload;
4. creates a Conditions-specific payload;
5. creates a Cost-specific payload;
6. creates a Vehicle/Payload-specific payload;
7. sends each payload to the corresponding FastAPI service;
8. checks the downstream responses; and
9. combines the four successful predictions into one JSON response.

Inside Docker, the Integration API communicates with the individual services using Docker service names:

```text
http://geography-api:8000/predict
http://conditions-api:8000/api/v1/conditions/predict
http://cost-api:8000/predict
http://telemetry-api:8000/api/v1/payload
```

The Integration API therefore acts as an aggregation layer rather than containing another ML model.

---

## 16. Docker Architecture

Docker provides isolated and reproducible environments for the different components of the system.

### Geography Serving Image

The Geography serving image is built using:

```text
docker/geography/Dockerfile
```

It provides the dependencies required to load the MLflow model and serve predictions through FastAPI and Uvicorn.

### Geography Training Image

The training environment is defined separately using:

```text
docker/geography/Dockerfile.train
```

and is built as:

```text
geography-trainer:latest
```

This separates model-training requirements from inference requirements.

### Integration Image

The Integration API is built from:

```text
docker/integration/Dockerfile
```

Its runtime includes FastAPI, Uvicorn, HTTPX and Pydantic.

HTTPX is used to communicate with the four downstream APIs.

### Additional Pipeline Images

Docker Compose also defines dedicated images including:

```text
conditions-trainer:latest
cost-spark:latest
cost-trainer:latest
vehicle-pipeline:latest
```

These provide isolated runtime environments for track-specific pipeline operations.

---

## 17. Docker Volumes

Docker volumes and bind mounts provide persistent storage and host-container file sharing.

### MLflow

```text
./models/mlflow:/mlflow
```

This persists the MLflow database and artifacts outside the container.

### Geography Monitoring

```text
./monitoring/geography:/opt/project/monitoring/geography
```

This makes prediction logs generated by the Geography API container available to monitoring scripts on the host.

### Airflow Database

Airflow PostgreSQL data is persisted using the named volume:

```text
airflow-db-data
```

This prevents metadata from being lost simply because the database container is recreated.

---

## 18. Kafka Streaming

Apache Kafka and ZooKeeper provide the streaming infrastructure for the project.

Start the streaming services using:

```bash
docker compose up -d zookeeper kafka
```

The Compose configuration exposes Kafka for both internal Docker communication and host-based clients.

Kafka is available internally through:

```text
kafka:9092
```

and host clients can use the configured host listener.

The streaming infrastructure supports continuously arriving logistics and telemetry data, particularly for the Vehicle / Payload workflow.

---

## 19. Monitoring and Data Drift

Monitoring components are maintained under:

```text
monitoring/
```

Track-specific monitoring implementations collect inference information and generate monitoring reports.

### Geography Monitoring

Every Geography prediction request records:

- model input features;
- predicted route time; and
- prediction timestamp.

Prediction logs are written to:

```text
monitoring/geography/prediction_logs.csv
```

### Generate Test Traffic

For monitoring demonstrations:

```bash
python monitoring/geography/generate_test_traffic.py
```

This sends test prediction requests to the running Geography API.

### Generate Geography Drift Report

```bash
python monitoring/geography/monitor.py
```

The monitoring process compares:

```text
Reference data:
data/processed/geography/
```

with:

```text
Current inference data:
monitoring/geography/prediction_logs.csv
```

Evidently's `DataDriftPreset` is used to compare feature distributions and generate an HTML monitoring report.

The report is written to:

```text
monitoring/geography/reports/data_drift_report.html
```

On Windows PowerShell:

```powershell
Start-Process monitoring\geography\reports\data_drift_report.html
```

Detected data drift indicates that the distribution of current model inputs differs statistically from the training/reference distribution. Drift does not automatically mean that the model has failed, but it provides a signal for further investigation and possible retraining.

---

## 20. Continuous Integration

The project contains a GitHub Actions workflow at:

```text
.github/workflows/ci.yml
```

The **RouteSense CI Pipeline** runs on:

- pushes to `main`;
- pushes to `integration/final`; and
- pull requests targeting `main`.

The CI job runs on Ubuntu and uses Python 3.12.

The workflow performs:

### Python Syntax Validation

```bash
python -m compileall tracks pipelines api monitoring
```

This validates Python source files across the major implementation directories.

### Docker Compose Validation

```bash
docker compose config
```

This verifies the Compose configuration.

### Docker Image Build Validation

The CI pipeline builds:

```text
Geography API
Conditions API
Cost API
Payload API
Integration API
```

This verifies that all five API Docker images can be successfully constructed from the repository.

---

## 21. Useful Commands

### Start Complete Stack

```bash
docker compose up -d
```

### Start and Rebuild

```bash
docker compose up -d --build
```

### Check Services

```bash
docker compose ps
```

### Validate Compose

```bash
docker compose config
```

### View Geography API Logs

```bash
docker compose logs geography-api
```

### Follow Geography Logs

```bash
docker compose logs -f geography-api
```

### View Integration API Logs

```bash
docker compose logs integration-api
```

### Start Only MLflow

```bash
docker compose up -d mlflow
```

### Start Kafka

```bash
docker compose up -d zookeeper kafka
```

### Stop Services

```bash
docker compose down
```

---

## 22. End-to-End MLOps Lifecycle

RouteSense demonstrates the following end-to-end lifecycle:

```text
                Raw / Streaming Data
                         |
                         v
              Distributed Processing
                    with Spark
                         |
                         v
                Feature Engineering
                         |
                         v
               Model Experimentation
                         |
                         v
              Model Tuning / Selection
                         |
                         v
               Cross-Validation
                         |
                         v
               Final Model Training
                         |
                         v
                MLflow Tracking
                         |
                         v
             MLflow Model Registry
                         |
                         v
              Dockerized FastAPI
                  Model Services
                         |
          +--------------+--------------+--------------+
          |              |              |              |
          v              v              v              v
      Geography      Conditions        Cost      Vehicle/Payload
          \              |              /            /
           \             |             /            /  
            +------ -------------------------------+
                         |
                         v
                 Integration API
                         |
                         v
              Prediction Monitoring
                  & Drift Analysis
```

The four modelling tracks remain independently maintainable while sharing common infrastructure for orchestration, model management, containerization, serving, integration and monitoring.

---

## 23. Key Design Principles

RouteSense follows several core MLOps principles:

- **Modularity** — Geography, Conditions, Cost and Vehicle/Telemetry are implemented as independent tracks.
- **Reproducibility** — Docker provides consistent runtime environments.
- **Distributed processing** — Spark is used for scalable preprocessing and feature preparation.
- **Workflow orchestration** — Airflow manages pipeline execution and task dependencies.
- **Experiment traceability** — MLflow records parameters, metrics and artifacts.
- **Model versioning** — registered models can be loaded by serving applications.
- **Service separation** — individual models are exposed through independent FastAPI services.
- **API integration** — a dedicated Integration API aggregates all four model outputs.
- **Monitoring** — inference information can be compared with reference data to detect distribution changes.
- **Leakage prevention** — potentially target-reconstructing features are identified and excluded where required.
- **Continuous integration** — GitHub Actions validates Python syntax, Docker Compose configuration and API image builds.
