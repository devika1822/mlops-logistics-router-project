import os
import pickle
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

def train_and_compare_models():
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

    best_f1 = -1.0
    best_model = None

    # =========================================================================
    #  MODEL 1: Logistic Regression
    # =========================================================================
    with mlflow.start_run(run_name="Baseline_Logistic_Regression"):
        print("\n--- Training Model 1: Baseline Logistic Regression ---")
        
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        
        lr_predictions = lr.predict(X_test)
        lr_acc = accuracy_score(y_test, lr_predictions)
        lr_f1 = f1_score(y_test, lr_predictions, average='weighted')
        
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_metric("accuracy", lr_acc)
        mlflow.log_metric("f1_score", lr_f1)
        
        print(f"Baseline Accuracy: {lr_acc:.4f} | Weighted F1: {lr_f1:.4f}")
        
        mlflow.sklearn.log_model(sk_model=lr, artifact_path="baseline_model")
        
        if lr_f1 > best_f1:
            best_f1 = lr_f1
            best_model = lr

    # =========================================================================
    # MODEL 2: Random Forest
    # =========================================================================
    with mlflow.start_run(run_name="Random_Forest_All_Features"):
        print("\n--- Training Model 2: Advanced Random Forest Classifier ---")
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        rf_predictions = rf.predict(X_test)
        rf_acc = accuracy_score(y_test, rf_predictions)
        rf_f1 = f1_score(y_test, rf_predictions, average='weighted')
        
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", rf_acc)
        mlflow.log_metric("f1_score", rf_f1)
        
        print(f"Advanced Accuracy: {rf_acc:.4f} | Weighted F1: {rf_f1:.4f}")
        
        importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        print("\n--- Mathematical Feature Importances ---")
        for feature, score in importances.items():
            print(f"{feature}: {score:.4f}")
            mlflow.log_metric(f"importance_{feature}", float(score))
            
        mlflow.sklearn.log_model(sk_model=rf, artifact_path="advanced_model")
        
        if rf_f1 > best_f1:
            best_f1 = rf_f1
            best_model = rf

    # =========================================================================
    # CENTRALIZED REGISTRY & MODEL EXPORT
    # =========================================================================
    print("\n--- Selecting Final Pipeline Winner ---")
    winner_name = "RandomForestClassifier" if isinstance(best_model, RandomForestClassifier) else "LogisticRegression"
    print(f"Winning model based on Weighted F1-Score: {winner_name} ({best_f1:.4f})")

    with mlflow.start_run(run_name="Final_Model_Registration"):
        mlflow.log_param("selected_winner", winner_name)
        mlflow.log_metric("final_winning_f1", best_f1)
        
        mlflow.sklearn.log_model(
            sk_model=best_model, 
            artifact_path="model",
            registered_model_name="Truck_Telemetry_Model"
        )
            
    os.makedirs("models", exist_ok=True)
    model_output_path = os.path.join("models", "truck_telemetry_model_all_features.pkl")
    
    with open(model_output_path, "wb") as f:
        pickle.dump(best_model, f)
        
    print(f"\nChampion model configuration saved successfully for production to: {model_output_path}")

if __name__ == "__main__":
    train_and_compare_models()
