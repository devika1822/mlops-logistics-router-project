import os
import json
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_final_model():
    print("=== Phase 3: Commencing Cost Candidate Tracking & Model Selection ===")
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

    print("\n--- Evaluating Linear Regression Candidate ---")
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
        print(f"Linear Regression -> MSE: {lr_mse:.4f} | R2: {lr_r2:.4f}")

    print("\n--- Evaluating Random Forest Candidate ---")
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
        print(f"Random Forest -> MSE: {rf_mse:.4f} | R2: {rf_r2:.4f}")

    if rf_r2 > lr_r2:
        winning_type = "random_forest"
        best_model = rf
        best_mse = rf_mse
        best_r2 = rf_r2
    else:
        winning_type = "linear_regression"
        best_model = lr
        best_mse = lr_mse
        best_r2 = lr_r2

    print(f"\nSelection Complete -> Winner: {winning_type} with R2: {best_r2:.4f}")

    mlflow.set_experiment("Cost Production")
    production_run_name = f"cost_final_{winning_type}"

    with mlflow.start_run(run_name=production_run_name):
        mlflow.log_param("model_name", winning_type)
        mlflow.log_param("feature_set", "engineered")
        mlflow.log_param("feature_count", feature_count)
        mlflow.log_param("features", feature_list_str)
        mlflow.log_param("average_speed_excluded", True)
        mlflow.log_param("training_rows", training_rows)

        mlflow.log_metric("final_mse", best_mse)
        mlflow.log_metric("final_r2", best_r2)
        
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            registered_model_name="cost_route_cost_model"
        )

        os.makedirs("models", exist_ok=True)
        with open("models/truck_cost_model_all_features.pkl", "wb") as f:
            pickle.dump(best_model, f)
            
        print(f"\nSuccessfully logged '{production_run_name}' under 'Cost Production' and registered 'cost_route_cost_model'!")

if __name__ == "__main__":
    train_final_model()
