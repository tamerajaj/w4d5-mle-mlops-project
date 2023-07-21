#!/usr/bin/env python3

"""
This is a demo service for Evidently metrics integration with Prometheus and Grafana.

Read `README.md` for proper setup and installation.

The service gets a reference dataset from reference.csv file and process current data with HTTP API.

Metrics calculation results are available with `GET /metrics` HTTP method in Prometheus compatible format.
"""
import dataclasses
import datetime
import hashlib
import logging
import os
from typing import Dict, List, Optional

import flask
import pandas as pd
import prometheus_client
import yaml
from evidently.model_monitoring import (  # CatTargetDriftMonitor,; ClassificationPerformanceMonitor,; NumTargetDriftMonitor,; ProbClassificationPerformanceMonitor,
    DataDriftMonitor,
    DataQualityMonitor,
    ModelMonitoring,
    RegressionPerformanceMonitor,
)
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.runner.loader import DataLoader, DataOptions
from flask import Flask
from werkzeug.middleware.dispatcher import DispatcherMiddleware

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

# Add prometheus wsgi middleware to route /metrics requests
app.wsgi_app = DispatcherMiddleware(
    app.wsgi_app, {"/metrics": prometheus_client.make_wsgi_app()}
)


def getDriftMonitoringService(config):
    loader = DataLoader()
    logging.info(f"config: {config}")
    options = MonitoringServiceOptions(**config["service"])

    reference_data = loader.load(
        options.reference_path,
        DataOptions(
            date_column=config["data_format"].get("date_column", None),
            separator=config["data_format"]["separator"],
            header=config["data_format"]["header"],
        ),
    )
    logging.info(f"reference dataset loaded: {len(reference_data)} rows")
    svc = MonitoringService(
        reference_data,
        options=options,
        column_mapping=ColumnMapping(**config["column_mapping"]),
    )
    return svc


@dataclasses.dataclass
class MonitoringServiceOptions:
    reference_path: str
    min_reference_size: int
    use_reference: bool
    moving_reference: bool
    window_size: int
    calculation_period_sec: int
    monitors: List[str]


@dataclasses.dataclass
class LoadedDataset:
    name: str
    references: pd.DataFrame
    monitors: List[str]
    column_mapping: ColumnMapping


EVIDENTLY_MONITORS_MAPPING = {
    "data_drift": DataDriftMonitor,
    "data_quality": DataQualityMonitor,
    "regression_performance": RegressionPerformanceMonitor,
}


class MonitoringService:
    metric: Dict[str, prometheus_client.Gauge]
    last_run: Optional[datetime.datetime]

    def __init__(
        self,
        reference: pd.DataFrame,
        options: MonitoringServiceOptions,
        column_mapping: ColumnMapping = None,
    ):
        self.monitoring = ModelMonitoring(
            monitors=[EVIDENTLY_MONITORS_MAPPING[k]() for k in options.monitors],
            options=[],
        )

        if options.use_reference:
            self.reference = reference.iloc[: -options.window_size, :].copy()
            self.current = pd.DataFrame()
        else:
            self.reference = reference.copy()
            self.current = pd.DataFrame().reindex_like(reference).dropna()
        self.column_mapping = column_mapping
        self.options = options
        self.metrics = {}
        self.next_run_time = None
        self.new_rows = 0
        self.hash = hashlib.sha256(
            pd.util.hash_pandas_object(self.reference).values
        ).hexdigest()
        self.hash_metric = prometheus_client.Gauge(
            "evidently:reference_dataset_hash", "", labelnames=["hash"]
        )

    def iterate(self, new_rows: pd.DataFrame):
        rows_count = new_rows.shape[0]

        self.current = self.current.append(new_rows, ignore_index=True)
        self.new_rows += rows_count
        current_size = self.current.shape[0]
        if self.new_rows < self.options.window_size < current_size:
            self.current.drop(
                index=list(range(0, current_size - self.options.window_size)),
                inplace=True,
            )
            self.current.reset_index(drop=True, inplace=True)

        if current_size < self.options.window_size:
            logging.info(
                f"Not enough data for measurement: {current_size} of {self.options.window_size}."
                f" Waiting more data"
            )
            return
        if (
            self.next_run_time is not None
            and self.next_run_time > datetime.datetime.now()
        ):
            logging.info(f"Next run at {self.next_run_time}")
            return
        self.next_run_time = datetime.datetime.now() + datetime.timedelta(
            seconds=self.options.calculation_period_sec
        )
        self.monitoring.execute(self.reference, self.current, self.column_mapping)
        self.hash_metric.labels(hash=self.hash).set(1)
        for metric, value, labels in self.monitoring.metrics():
            metric_key = f"evidently:{metric.name}"
            found = self.metrics.get(metric_key)
            if not found:
                found = prometheus_client.Gauge(
                    metric_key,
                    "",
                    () if labels is None else list(sorted(labels.keys())),
                )
                self.metrics[metric_key] = found
            if labels is None:
                found.set(value)
            else:
                found.labels(**labels).set(value)


SERVICE: Optional[MonitoringService] = None


@app.before_first_request
def startup_event():
    # pylint: disable=global-statement
    global SERVICE
    config_file_name = "config.yaml"
    # try to find a config file, it should be generated via a data preparation script
    if not os.path.exists(config_file_name):
        exit(
            "Cannot find config file for the metrics service. Try to check README.md for setup instructions."
        )

    with open(config_file_name, "rb") as config_file:
        config = yaml.safe_load(config_file)

    SERVICE = getDriftMonitoringService(config)


@app.route("/iterate/<dataset>", methods=["POST"])
def iterate(dataset: str):
    item = flask.request.json

    global SERVICE
    if SERVICE is None:
        return "Internal Server Error: service not found", 500
    logging.info(f"Got Data: {item}")
    data = pd.DataFrame([item])
    logging.info(f"Dataframe: {data.head()}")
    SERVICE.iterate(new_rows=data)
    return "ok"


if __name__ == "__main__":
    app.run(debug=True)
