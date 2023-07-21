#!/usr/bin/env python
# coding: utf-8

# # Train a Regression Model
#
# In this notebook we will train a Linear Regression Model on the Green Taxi Dataset. We will only use one month for the training. And keep only a small number of features.
#
# We want the model to predict the duration of a trip. This can be useful for the taxi drivers to plan their trips, for the customers to know how long a trip will take but also for the taxi companies to plan their fleet. The first two predictions would need real time predictions because the duration of a trip is not known in advance. The last one could be done in batch mode, as it is more a analytical task that doesn't need to be done in real time.
#
# Additionally, we will use MLFlow to track the model training and log the model artifacts.

# In[115]:


import os

import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow.tracking.client import MlflowClient
from sklearn.linear_model import LinearRegression

# import randomforest regressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split

# In[1]:


a = 5


# In[ ]:


get_ipython().system(" jupyter nbconvert --to script 01-train-ml-model.ipynb")


# In[116]:


year = 2021
month = 2
color = "green"


# In[117]:


# Download the data
if not os.path.exists(f"./data/{color}_tripdata_{year}-{month:02d}.parquet"):
    os.system(
        f"wget -P ./data https://d37ci6vzurychx.cloudfront.net/trip-data/{color}_tripdata_{year}-{month:02d}.parquet"
    )


# In[118]:


# Load the data

df = pd.read_parquet(f"./data/{color}_tripdata_{year}-{month:02d}.parquet")


# In[119]:


df.shape


# Now we will set up the connection to MLFlow. For that we have to create a `.env` file with the URI to the MLFlow Server in gcp (this will be `http://<external-ip>:5000`). You can simply run:
#
# ```bash
# echo "MLFLOW_TRACKING_URI=http://<external-ip>:5000" > .env
# ```
#
# We also will create an experiment to track the model and the metrics.

# In[120]:


load_dotenv()


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")


# In[121]:


# Set up the connection to MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
# Setup the MLflow experiment
mlflow.set_experiment("green-taxi-monitoring")


# If everything went well, you should be able to see the experiment now in the MLFlow UI at `http://<external-ip>:5000`.

# Let's start now with looking at the data a bit:

# In[122]:


df.head()


# In[123]:


df.info()


# In[124]:


# Look for missing values
df.isnull().sum()


# Nearly all features seem to be in the correct type and we have only missings in features that we will not use for the model training. For predicting the duration of a trip, we will use the following features:
#
# - `PULocationID`: The pickup location ID
# - `DOLocationID`: The dropoff location ID
# - `trip_distance`: The distance of the trip in miles
# - `fare_amount`: The fare amount in USD
# - `total_amount`: The total amount in USD
# - `passenger_count`: The number of passengers
#
# But first we have to calculate the duration of the trip in minutes because it is our target. For that we will use the `tpep_pickup_datetime` and `tpep_dropoff_datetime` columns. We will also remove all trips that have a duration of 0 and that are longer than 1 hours to remove outliers.

# In[125]:


features = [
    "PULocationID",
    "DOLocationID",
    "trip_distance",
    "passenger_count",
    "fare_amount",
    "total_amount",
]
target = "duration"


# In[126]:


# calculate the trip duration in minutes and drop trips that are less than 1 minute and more than 2 hours
def calculate_trip_duration_in_minutes(df):
    df["duration"] = (
        df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    df = df[(df["duration"] >= 1) & (df["duration"] <= 60)]
    df = df[(df["passenger_count"] > 0) & (df["passenger_count"] < 8)]
    df = df[features + [target]]
    return df


# In[127]:


df_processed = calculate_trip_duration_in_minutes(df)


# In[128]:


df_processed.head()


# In[129]:


df_processed.duration.describe()


# In[130]:


df_processed.duration.hist()


# Now that we have the dataframe that we want to train our model on. We need to split it into a train and test set. We will use 80% of the data for training and 20% for testing.

# In[131]:


y = df_processed["duration"]
X = df_processed.drop(columns=["duration"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2
)


# And now we can train the model and track the experiment with MLFlow. We will set tags to the experiment to make it easier to find it later.
#
# - `model`: `linear-regression`
# - `dataset`: `yellow-taxi`
# - `developer`: `your-name`
# - `train_size`: The size of the train set
# - `test_size`: The size of the test set
# - `features`: The features that we used for training
# - `target`: The target that we want to predict
# - `year`: The year of the data
# - `month`: The month of the data
#
# We could also log the model parameters but Linear Regression doesn't have any.
#
# And finally we will log the metrics:
#
# - `rmse`: The root mean squared error
#
# We will also log the model artifacts. For that we will need to set the `service account json` that we downloaded earlier as the environment variable `GOOGLE_APPLICATION_CREDENTIALS`.

# In[132]:


load_dotenv()
SA_KEY = os.getenv("SA_KEY")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SA_KEY


# In[133]:


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

    y_pred = lr.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mlflow.log_metric("rmse", rmse)

    mlflow.sklearn.log_model(lr, "model")
    run_id = mlflow.active_run().info.run_id

    model_uri = f"runs:/{run_id}/model"
    model_name = "green-taxi-ride-duration"
    mlflow.register_model(model_uri=model_uri, name=model_name)

    model_version = 1
    new_stage = "Production"
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage=new_stage,
        archive_existing_versions=False,
    )


# You should now see your run in the MLFlow UI. Under the created experiment. You can also see the logged tags, the metric and the saved model.
#
# ![mlflow-ui](./images/mlflow-run.png)
#
# And you can see what you need to do to load the model in an API or script in the UI as long as the application has access to MLFlow.
#
