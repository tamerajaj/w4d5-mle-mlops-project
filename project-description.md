# MLOps Project

## Project Description

In this project you will create a Github Action that trains the model we trained on Wednesday during the intro to evidently and uses DVC for the versioning of the train data. This model should be used in an API you can take the one from yesterday. 
You should monitor the API and the model using evidently, prometheus and grafana.

## Project Structure

The project structure is as follows:

- Take the code from the training notebook and put it into a python file 
- Create a Github Action that trains the model and tracks it with MLflow and uses DVC for the versioning of the train data
- Update the API that uses this model and deploy the API
- Monitor the API and the model using evidently, prometheus and grafana


Bonus: Send data to the API and monitor the data with evidently
Bonus2: Use the Github Action to retrain the model with new data if you see data drift 

## How to submit your project

Upload your project on GitHub and send us the link. Answer the questions above in the README.md file.
