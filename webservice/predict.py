import os

import mlflow
import pandas as pd


def load_model(model_name):
    stage = "Production"
    model_uri = f"models:/{model_name}/{stage}"
    model = mlflow.pyfunc.load_model(model_uri)
    return model


def predict(model_name, data):
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print("Data input:", data)
    model_input = pd.DataFrame([data.dict()])
    print("Load model...")
    model = load_model(model_name)
    print("Making prediction with data: ", model_input.head())
    prediction = model.predict(model_input)
    return float(prediction[0])
