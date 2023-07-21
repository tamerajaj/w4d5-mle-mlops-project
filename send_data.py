import time

import pandas as pd
import requests

df = pd.read_parquet("data/green_tripdata_2022-02.parquet")

features = [
    "PULocationID",
    "DOLocationID",
    "trip_distance",
    "passenger_count",
    "fare_amount",
    "total_amount",
]


# calculate the trip duration in minutes and drop trips that are less than 1 minute and more than 2 hours
def calculate_trip_duration_in_minutes(df):
    df["duration"] = (
        df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    df = df[(df["duration"] >= 1) & (df["duration"] <= 60)]
    df = df[(df["passenger_count"] > 0) & (df["passenger_count"] < 8)]
    df = df[features]
    return df


df = calculate_trip_duration_in_minutes(df)

df = df[29000:]

for idx, row in df.sample(frac=1).iterrows():
    time.sleep(1)
    url = "http://<external-ip>:8080/predict"
    headers = {"accept": "application/json", "Content-Type": "application/json"}
    print(row.to_json())
    response = requests.post(url, headers=headers, data=row.to_json())
    print(response)
