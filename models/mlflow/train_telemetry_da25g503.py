import os
import pickle
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

def train_on_all_features():
    mlflow.set_experiment("Truck_Telemetry_Full_Features")
    
    data_path = "data/processed/clean_telemtery_da25g503.csv"
    if not os.path.exists(data_path):
        if os.path.exists("clean_telemtery_da25g503.csv"):
            data_path = "clean_telemtery_da25g503.csv"
        else:
            raise FileNotFoundError("Error: Clean telemetry file not found. Please check path.")

    df = pd.read_csv(data_path)
    
    target_column = "breakdown_risk_level"
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' missing from dataset.")
        
    X = df.drop(columns=[target_column])
    X = X.select_dtypes(include=['number'])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name="Random_Forest_All_Features"):
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        predictions = rf.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted')
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        
        importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        
        print("\n--- Mathematical Feature Importances ---")
        for feature, score in importances.items():
            print(f"{feature}: {score:.4f}")
            mlflow.log_metric(f"importance_{feature}", float(score))
            
        # FIX 1: Moved OUTSIDE the loop so it logs only ONCE per run
        # FIX 2: Changed 'sk_model=model' to 'sk_model=rf' to reference the trained variable
        mlflow.sklearn.log_model(
            sk_model=rf, 
            artifact_path="model",
            registered_model_name="Truck_Telemetry_Model"
        )
            
    os.makedirs("models", exist_ok=True)
    model_output_path = os.path.join("models", "truck_telemetry_model_all_features.pkl")
    
    with open(model_output_path, "wb") as f:
        pickle.dump(rf, f)
        
    print(f"\nModel trained on all features and saved successfully to: {model_output_path}")

if __name__ == "__main__":
    train_on_all_features()
