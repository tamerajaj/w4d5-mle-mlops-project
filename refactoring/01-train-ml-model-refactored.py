#!/usr/bin/env python
# coding: utf-8

import os

import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow.tracking.client import MlflowClient
from rich import print
from sklearn.linear_model import LinearRegression

# import randomforest regressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def download_data(year=2021, month=2, color="green"):
    # Download the data

    if not os.path.exists(f"./data/{color}_tripdata_{year}-{month:02d}.parquet"):
        print(f"Downloading {color} taxi data for {year}-{month:02d}")
        os.system(
            f"wget -P ./data https://d37ci6vzurychx.cloudfront.net/trip-data/{color}_tripdata_{year}-{month:02d}.parquet"
        )
    print(f"Data for {color} taxi in {year}-{month:02d} is ready")


load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

mlflow.set_experiment("green-taxi-monitoring")

features = [
    "PULocationID",
    "DOLocationID",
    "trip_distance",
    "passenger_count",
    "fare_amount",
    "total_amount",
]
target = "duration"


# calculate the trip duration in minutes and drop trips that are less than 1 minute and more than 2 hours
def calculate_trip_duration_in_minutes(df):
    print("Calculating trip duration in minutes")
    df["duration"] = (
        df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    df = df[(df["duration"] >= 1) & (df["duration"] <= 60)]
    df = df[(df["passenger_count"] > 0) & (df["passenger_count"] < 8)]
    df = df[features + [target]]
    return df


def preprocess(df):
    print("Preprocessing the data")
    df_processed = calculate_trip_duration_in_minutes(df)

    y = df_processed["duration"]
    X = df_processed.drop(columns=["duration"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.2
    )
    return X_train, X_test, y_train, y_test


def train_model(df):
    print("Training the model")
    load_dotenv()
    X_train, X_test, y_train, y_test = preprocess(df)
    SA_KEY = os.getenv("SA_KEY")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SA_KEY

    with mlflow.start_run():
        tags = {
            "model": "linear regression",
            "developer": "<your name>",
            "dataset": f"{color}-taxi",
            "year": year,
            "month": month,
            "features": features,
            "target": target,
        }
        mlflow.set_tags(tags)

        lr = LinearRegression()
        lr.fit(X_train, y_train)
        print("Model training completed")

        y_pred = lr.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        print(f"RMSE: {rmse:.2f}")
        mlflow.log_metric("rmse", rmse)

        mlflow.sklearn.log_model(lr, "model")
        run_id = mlflow.active_run().info.run_id

        model_uri = f"runs:/{run_id}/model"
        model_name = "green-taxi-ride-duration"
        mlflow.register_model(model_uri=model_uri, name=model_name)
        print(f"Model saved as {model_name}")
        model_version = 1
        new_stage = "Production"
        client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage=new_stage,
            archive_existing_versions=False,
        )


if __name__ == "__main__":
    year = 2021
    month = 2
    color = "green"
    download_data(year, month, color)
    df = pd.read_parquet(f"./data/{color}_tripdata_{year}-{month:02d}.parquet")
    train_model(df)
