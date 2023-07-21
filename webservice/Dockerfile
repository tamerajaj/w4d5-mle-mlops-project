# Use the official Python image as the base image
FROM python:3.10-slim-buster

# Set the working directory
WORKDIR /app

# Copy the application code to the working directory
COPY . /app

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port
EXPOSE 8080

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]