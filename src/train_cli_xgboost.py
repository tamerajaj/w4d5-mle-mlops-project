import argparse

# import base64
import os

import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

parser = argparse.ArgumentParser()
parser.add_argument(
    "--cml_run", default=False, action=argparse.BooleanOptionalAction, required=True
)
args = parser.parse_args()
cml_run = args.cml_run

GOOGLE_APPLICATION_CREDENTIALS = "./credentials.json"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS


# Set up the connection to MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Setup the MLflow experiment
mlflow.set_experiment("green-taxi-trip-duration-xgb")

# Set variables
color = "green"
features = ["PULocationID", "DOLocationID", "trip_distance"]
target = "duration"
model_name = "green-taxi-trip-duration-xgb"

df = pd.read_parquet("data/green_tripdata_2021-01.parquet")


def calculate_trip_duration_in_minutes(df):
    df["trip_duration_minutes"] = (
        df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    df = df[(df["trip_duration_minutes"] >= 1) & (df["trip_duration_minutes"] <= 60)]
    return df


def preprocess(df):
    df = df.copy()
    df = calculate_trip_duration_in_minutes(df)
    categorical_features = ["PULocationID", "DOLocationID"]
    df[categorical_features] = df[categorical_features].astype(str)
    df["trip_route"] = df["PULocationID"] + "_" + df["DOLocationID"]
    df = df[["trip_route", "trip_distance", "trip_duration_minutes"]]
    return df


df_processed = preprocess(df)

y = df_processed["trip_duration_minutes"]
X = df_processed.drop(columns=["trip_duration_minutes"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2
)

X_train = X_train.to_dict(orient="records")
X_test = X_test.to_dict(orient="records")

with mlflow.start_run() as run:
    tags = {
        "model": "xgboost pipeline",
        "developer": "<your name>",
        "dataset": f"{color}-taxi",
        "features": features,
        "target": target,
    }
    mlflow.set_tags(tags)
    pipeline = make_pipeline(DictVectorizer(), xgb.XGBRegressor())
    pipeline.fit(X_train, y_train)

    y_pred_train = pipeline.predict(X_train)
    rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
    mlflow.log_metric("rmse train", rmse_train)

    y_pred_test = pipeline.predict(X_test)
    rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
    mlflow.log_metric("rmse test", rmse_test)

    mlflow.sklearn.log_model(pipeline, "model")

    run_id = run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri=model_uri, name=model_name)


if cml_run:
    with open("metrics.txt", "w") as f:
        f.write(f"RMSE on the Train Set: {rmse_train}")
        f.write(f"RMSE on the Test Set: {rmse_test}")
