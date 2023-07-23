# MLOps Project

## Summary:
This MLOps project showcases a structured workflow for model training, deployment, and monitoring using GitHub Actions, Docker, and Google Cloud Platform. The model is deployed using FastAPI and monitored using Prometheus, Evidently, and Grafana.

## Structure:

### 1. Refactoring:
In [01-train-ml-model-refactored.py](refactoring/01-train-ml-model-refactored.py) file you can find the initial refactored code of the project. The refactored code is based on the notebook [01-train-ml-model.py](refactoring/01-train-ml-model.py).
### 2. Model training:
* The refactored code is copied to [src/train.py](src/train.py) file. The code is refactored to be able to run it from the command line. It also has MLflow tracking implemented.
* A GitHub action is created to run the model training on every pull request to the main branch. The action is defined in [.github/workflows/cml.yaml](.github/workflows/cml.yaml) file.
* The GitHub action runs the model training on a GitHub with DVC implemented for data versioning.
* The GitHub action also saves the model using MLflow and stages the model in MLflow registry.
### 3. Model serving:
* The model is served using FastAPI. The code is in [webservice/app.py](webservice/app.py) file. The service is Dockerized and deployable to GCP.

### 4. Model monitoring:
* The model is monitored using [Prometheus](deployment/prometheus_deployment),  [Evidently](evidently_service), and  [Grafana](deployment/grafana_deployment).
* The monitoring services are Dockerized and deployable to GCP.


### 5. Monitoring services deployment:
* A GitHub action is created and is triggered with a push to the api-tracking branch (can be changed).
* The action builds and pushes the Docker image of the model serving and monitoring services to GCP. The action is defined in [.github/workflows/deploy.yaml](.github/workflows/deploy.yaml) file.

### 6. Running the Dockerized services:
* The bash file [docker_clean_pull_run.sh](docker_clean_pull_run.sh) is created to run the Dockerized services locally.
* It can be run on the VM instance like this:
```bash
./docker_pull_cleanup.sh --user <user_name> --project <project_name> --internal-ip <internal_ip_VM> --mlflow-tracking <tracking_ip_mlflow>
```
Note: Don't forget the port number for the MLflow tracking server, e.g. ```<mlflow-ip>:5000```.
### 7. Environment variables:
The GitHub actions use the following environment variables:
* ```BUCKET_NAME```: GCP bucket name.
* ```GAR_LOCATION```: GCP region.
* ```GOOGLE_APPLICATION_CREDENTIALS```: GCP service account key.
* ```INTERNAL_IP```: Internal IP of the VM instance.
* ```MLFLOW_TRACKING_URI```: MLflow tracking server IP with port. 
* ```PROJECT_ID```: GCP project ID.
* ```REPOSITORY```: Docker registry name. [needs to be changed].

### 8. Sending data to the model:
You can send data to the model using the [send_data.py](send_data.py) script. The script sends the data to the model using the FastAPI service. You need to add the external IP of the VM instance to the script.

### 9. Links:
If everything is deployed correctly, you can access the following services:
* MLflow tracking server: ```<mlflow-ip>:5000```
* FastAPI service: ```<monitoring-ip>:8080```
* Grafana: ```<monitoring-ip>:3000```
* Prometheus: ```<monitoring-ip>:9090```
* Evidently: ```<monitoring-ip>:8085```

---
## Environment

You need the google cloud sdk installed and configured. If you don't have it installed follow the instructions here or use homebrew on mac:
```bash
brew install --cask google-cloud-sdk
```

Python environment:

```bash
pyenv local 3.11.3
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```
---
## Prerequisites on GCP:
1. Two running VM instance on GCP:
   1. VM instance for MLflow tracking server.
   2. VM instance for the FastAPI server, Grafana, Evidently and Prometheus.
3. Additionally for MLflow tracking server:
   1. A PostgreSQL database for MLFLow experiments.
   2. A GCS bucket for MLFlow artifacts.
---
## Pre-commit hooks:
This repo contains pre-commit hooks for Python projects, integrating Black, Isort, and Flake8. These hooks ensure code formatting, proper import sorting (using Black profile), and adherence to Flake8 rules. Maintaining code quality and consistency before each commit.