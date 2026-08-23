import os
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_and_compare_models():
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Truck_Telemetry_Regression")
    
    df = pd.read_csv("data/processed/clean_telemetery_da25g503.csv")
    
    primary_target = "delivery_efficiency_score"
    competing_targets = ["breakdown_risk_level", "optimized_route_cost", "optimized_route_time_min"]
    
    y = df[primary_target]
    X = df.drop(columns=[primary_target] + [col for col in competing_targets if col in df.columns])
    X = X.select_dtypes(include=['number'])  # Keep only clean numerical features

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    best_r2 = -1.0
    best_model = None

    # --- MODEL 1: Baseline Linear Regression ---
    with mlflow.start_run(run_name="Baseline_Linear_Regression"):
        print("\n--- Training Model 1: Baseline Linear Regression ---")
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        
        preds = lr.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("R2_Score", r2)
        mlflow.sklearn.log_model(lr, "baseline_lr_model")
        print(f"Baseline LR -> MSE: {mse:.4f} | R2: {r2:.4f}")
        
        if r2 > best_r2:
            best_r2, best_model = r2, lr

    # --- MODEL 2: Advanced Random Forest Regressor ---
    with mlflow.start_run(run_name="Random_Forest_Regressor"):
        print("\n--- Training Model 2: Advanced Random Forest Regressor ---")
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        preds_rf = rf.predict(X_test)
        mse_rf = mean_squared_error(y_test, preds_rf)
        r2_rf = r2_score(y_test, preds_rf)
        
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("MSE", mse_rf)
        mlflow.log_metric("R2_Score", r2_rf)
        mlflow.sklearn.log_model(rf, "advanced_rf_model")
        print(f"Advanced RF -> MSE: {mse_rf:.4f} | R2: {r2_rf:.4f}")
        
        if r2_rf > best_r2:
            best_r2, best_model = r2_rf, rf

    # --- MODEL REGISTRATION ---
    winner_name = "RandomForest" if isinstance(best_model, RandomForestRegressor) else "LinearRegression"
    print(f"\n🏆 Champion Model Selected: {winner_name} (R2: {best_r2:.4f})")
    
    with mlflow.start_run(run_name="Final_Model_Registration"):
        mlflow.sklearn.log_model(best_model, "model", registered_model_name="Delivery_Efficiency_Model")
        
    os.makedirs("models", exist_ok=True)
    with open("models/truck_telemetry_model_all_features.pkl", "wb") as f:
        pickle.dump(best_model, f)
    print("Model successfully saved inside models/ directory.")

if __name__ == "__main__":
    train_and_compare_models()
