from fastapi import FastAPI, HTTPException, Body
import pandas as pd
import numpy as np
import mlflow.sklearn

app = FastAPI(
    title="Truck Route Cost Prediction API",
    description="Serves predictions from MLflow Model Registry",
    version="1.0"
)

mlflow.set_tracking_uri("http://localhost:5000")
MODEL_URI = "models:/cost_route_cost_model/latest"

def load_model():
    try:
        return mlflow.sklearn.load_model(MODEL_URI)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = load_model()

@app.get("/")
def read_root():
    return {
        "status": "Active",
        "model_loaded": model is not None,
        "model_uri": MODEL_URI
    }

@app.post("/predict")
def predict_cost(payload: dict = Body(...)):
    global model
    if model is None:
        model = load_model()
        if model is None:
            raise HTTPException(status_code=500, detail="Model not found in MLflow Registry.")
    
    try:
        input_df = pd.DataFrame([payload])
        
        if hasattr(model, "feature_names_in_"):
            for col in model.feature_names_in_:
                if col not in input_df.columns:
                    input_df[col] = 0.0 # Default missing values to 0
            input_df = input_df[model.feature_names_in_]

        prediction = model.predict(input_df)
        
        return {
            "status": "success",
            "predicted_optimized_route_cost": float(prediction[0])
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
