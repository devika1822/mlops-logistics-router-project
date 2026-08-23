import os
import json
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_final_model():
    print("=== Phase 3: Commencing Final Production Model Training Loops ===")
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Truck_Route_Cost_Regression")
    
    df = pd.read_csv("data/processed/clean_cost_da25g503.csv")
    
    primary_target = "optimized_route_cost"
    competing_targets = ["delivery_efficiency_score", "optimized_route_time_min"]
    
    y = df[primary_target]
    X = df.drop(columns=[primary_target] + [col for col in competing_targets if col in df.columns], errors='ignore')
    X = X.select_dtypes(include=['number']) # Explicitly maintain clean numerical feature boundaries

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    config_path = "config/best_params_da25g503.json"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            best_params = json.load(f)
        print(f"Successfully loaded optimized hyperparameters from tuning file: {best_params}")
    else:
        best_params = {"n_estimators": 100, "max_depth": None, "random_state": 42}
        print("Tuning parameter config file missing. Falling back to default baseline configurations.")

    best_params["random_state"] = 42

    with mlflow.start_run(run_name="Final_Champion_Model_Training"):
        rf = RandomForestRegressor(**best_params)
        rf.fit(X_train, y_train)
        
        preds = rf.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        mlflow.log_metrics({"Final_MSE": mse, "Final_R2_Score": r2})
        
        mlflow.sklearn.log_model(
            sk_model=rf, 
            artifact_path="model", 
            registered_model_name="Optimized_Route_Cost_Model"
        )
        
        os.makedirs("models", exist_ok=True)
        with open("models/truck_cost_model_all_features.pkl", "wb") as f:
            pickle.dump(rf, f)
            
        print("\n Production Model Training and Registry Success!")
        print(f" Final Evaluation Metrics -> Test MSE: {mse:.4f} | Test R2 Score: {r2:.4f}")

if __name__ == "__main__":
    train_final_model()
