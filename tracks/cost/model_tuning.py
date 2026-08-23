import os
import json
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

def tune_hyperparameters():
    print("=== Phase 2: Starting Automated Hyperparameter Optimization Sweeps ===")
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Truck_Cost_Hyperparameter_Tuning")
    
    df = pd.read_csv("data/processed/clean_cost_da25g503.csv")
    
    primary_target = "optimized_route_cost"
    competing_targets = ["delivery_efficiency_score", "optimized_route_time_min"]
    
    y = df[primary_target]
    X = df.drop(columns=[primary_target] + [col for col in competing_targets if col in df.columns], errors='ignore')
    X = X.select_dtypes(include=['number'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_distributions = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }


    mlflow.sklearn.autolog(log_models=False)

    with mlflow.start_run(run_name="Random_Search_Tuning_Wave"):
        base_rf = RandomForestRegressor(random_state=42)
        
        search = RandomizedSearchCV(
            estimator=base_rf, 
            param_distributions=param_distributions, 
            n_iter=8, 
            cv=3, 
            scoring='r2', 
            random_state=42,
            n_jobs=-1
        )
        
        search.fit(X_train, y_train)
        
        os.makedirs("config", exist_ok=True)
        with open("config/best_params_da25g503.json", "w") as f:
            json.dump(search.best_params_, f, indent=4)
            
        print("\n Tuning Search Wave Complete!")
        print(f" Optimized Param Matrix Found: {search.best_params_}")
        print(f" Top Optimization R2 Score: {search.best_score_:.4f}")

if __name__ == "__main__":
    tune_hyperparameters()
