#!/bin/bash

# Default values
USERNAME=""
PROJECT_ID=""
INTERNAL_IP=""
MLFLOW_TRACKER=""

# Parse command-line options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -u|--user) USERNAME="$2"; shift ;;
        -p|--project) PROJECT_ID="$2"; shift ;;
        -i|--internal-ip) INTERNAL_IP="$2"; shift ;;
        -t|--mlflow-tracking) MLFLOW_TRACKER="$2"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# Check if all required options are provided
if [ -z "$USERNAME" ] || [ -z "$PROJECT_ID" ] || [ -z "$INTERNAL_IP" ] || [ -z "$MLFLOW_TRACKER" ]; then
    echo "Usage: $0 -u <username> -p <project_id> -i <internal_ip> -t <mlflow_tracker>"
    exit 1
fi

# Rest of the script remains the same...


sudo docker stop $(sudo docker ps -aq) # Stop all running Docker containers

sudo docker rm $(sudo docker ps -aq) # Remove all Docker containers

## Remove all Docker images
#docker rmi $(docker images -aq)

# Pull the Docker images
sudo docker --config /home/"$USERNAME"/.docker pull europe-west3-docker.pkg.dev/"$PROJECT_ID"/docker-registry/ml-service:latest
sudo docker --config /home/"$USERNAME"/.docker pull europe-west3-docker.pkg.dev/"$PROJECT_ID"/docker-registry/grafana:latest
sudo docker --config /home/"$USERNAME"/.docker pull europe-west3-docker.pkg.dev/"$PROJECT_ID"/docker-registry/evidently_service:latest
sudo docker --config /home/"$USERNAME"/.docker pull europe-west3-docker.pkg.dev/"$PROJECT_ID"/docker-registry/prometheus:latest

# Run the Docker containers

# Run ml-service with environment variables INTERNAL_IP and MLFLOW_TRACKING_URI
sudo docker run -d -p 8080:8080 -e INTERNAL_IP="$INTERNAL_IP" -e MLFLOW_TRACKING_URI="$MLFLOW_TRACKER" --name=webservice europe-west3-docker.pkg.dev/"$PROJECT_ID"/docker-registry/ml-service

# Run prometheus with environment variable INTERNAL_IP
sudo docker run -d -p 9090:9090 -e INTERNAL_IP="$INTERNAL_IP" --name=prometheus --network=monitoring europe-west3-docker.pkg.dev/"$PROJECT_ID"/docker-registry/prometheus

# Run evidently_service
sudo docker run -d -p 8085:8085 --name=evidently_service --network=monitoring europe-west3-docker.pkg.dev/"$PROJECT_ID"/docker-registry/evidently_service

# Run grafana
sudo docker run -d -p 3000:3000 --name=grafana --network=monitoring europe-west3-docker.pkg.dev/"$PROJECT_ID"/docker-registry/grafana
