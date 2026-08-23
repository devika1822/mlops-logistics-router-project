import json
import os
import pickle
from datetime import datetime, timedelta
from airflow.decorators import dag, task
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

@dag(
    dag_id='cost_route_model_training_pipeline',
    default_args=default_args,
    description='Automated pipeline to track, compare, and register route cost models using MLflow',
    schedule_interval=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['mlflow', 'cost_prediction', 'regression'],
)
def cost_pipeline():

    @task
    def train_and_compare_candidates():
        mlflow.set_tracking_uri("http://localhost:5000")
        
        df = pd.read_csv("data/processed/clean_cost_da25g503.csv")
        
        primary_target = "optimized_route_cost"
        competing_targets = ["delivery_efficiency_score", "optimized_route_time_min"]
        
        y = df[primary_target]
        X = df.drop(columns=[primary_target] + [col for col in competing_targets if col in df.columns], errors='ignore')
        X = X.select_dtypes(include=['number'])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        feature_list_str = ",".join(X.columns.tolist())
        feature_count = len(X.columns)
        training_rows = len(X_train)

        mlflow.set_experiment("Cost Track")

        with mlflow.start_run(run_name="cost_candidate_linear_regression"):
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            lr_preds = lr.predict(X_test)
            lr_mse = mean_squared_error(y_test, lr_preds)
            lr_r2 = r2_score(y_test, lr_preds)
            
            mlflow.log_param("model_name", "linear_regression")
            mlflow.log_param("feature_set", "engineered")
            mlflow.log_param("feature_count", feature_count)
            mlflow.log_param("features", feature_list_str)
            mlflow.log_param("average_speed_excluded", True)
            mlflow.log_param("training_rows", training_rows)

            mlflow.log_metric("cv_mse", lr_mse)
            mlflow.log_metric("cv_r2", lr_r2)
            mlflow.sklearn.log_model(lr, artifact_path="model")

        config_path = "config/best_params_da25g503.json"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                best_params = json.load(f)
        else:
            best_params = {"n_estimators": 100, "max_depth": None, "random_state": 42}

        best_params["random_state"] = 42

        with mlflow.start_run(run_name="cost_candidate_random_forest"):
            rf = RandomForestRegressor(**best_params)
            rf.fit(X_train, y_train)
            rf_preds = rf.predict(X_test)
            rf_mse = mean_squared_error(y_test, rf_preds)
            rf_r2 = r2_score(y_test, rf_preds)
            
            mlflow.log_param("model_name", "random_forest")
            mlflow.log_param("feature_set", "engineered")
            mlflow.log_param("feature_count", feature_count)
            mlflow.log_param("features", feature_list_str)
            mlflow.log_param("average_speed_excluded", True)
            mlflow.log_param("training_rows", training_rows)

            mlflow.log_metric("cv_mse", rf_mse)
            mlflow.log_metric("cv_r2", rf_r2)
            mlflow.sklearn.log_model(rf, artifact_path="model")

        if rf_r2 > lr_r2:
            winning_type = "random_forest"
            best_mse = rf_mse
            best_r2 = rf_r2
            winning_model = rf
        else:
            winning_type = "linear_regression"
            best_mse = lr_mse
            best_r2 = lr_r2
            winning_model = lr

        os.makedirs("models", exist_ok=True)
        with open("models/temp_best_model.pkl", "wb") as f:
            pickle.dump(winning_model, f)

        return {
            "winning_type": winning_type,
            "best_mse": best_mse,
            "best_r2": best_r2,
            "feature_count": feature_count,
            "feature_list_str": feature_list_str,
            "training_rows": training_rows
        }

    @task
    def register_production_model(metrics_dict: dict):
        mlflow.set_tracking_uri("http://localhost:5000")
        winning_type = metrics_dict["winning_type"]
        best_mse = metrics_dict["best_mse"]
        best_r2 = metrics_dict["best_r2"]
        
        with open("models/temp_best_model.pkl", "rb") as f:
            best_model = pickle.load(f)

        mlflow.set_experiment("Cost Production")
        production_run_name = f"cost_final_{winning_type}"

        with mlflow.start_run(run_name=production_run_name):
            mlflow.log_param("model_name", winning_type)
            mlflow.log_param("feature_set", "engineered")
            mlflow.log_param("feature_count", metrics_dict["feature_count"])
            mlflow.log_param("features", metrics_dict["feature_list_str"])
            mlflow.log_param("average_speed_excluded", True)
            mlflow.log_param("training_rows", metrics_dict["training_rows"])

            mlflow.log_metric("final_mse", best_mse)
            mlflow.log_metric("final_r2", best_r2)
            
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="model",
                registered_model_name="cost_route_cost_model"
            )

            with open("models/truck_cost_model_all_features.pkl", "wb") as f:
                pickle.dump(best_model, f)

    eval_results = train_and_compare_candidates()
    register_production_model(eval_results)

dag_instance = cost_pipeline()
