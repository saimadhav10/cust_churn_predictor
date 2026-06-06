# Customer Churn Prediction Microservice (MLOps)

An end-to-end, production-ready machine learning API designed to identify high-risk customers before they cancel their subscriptions. 

This project demonstrates a complete MLOps lifecycle: from synthetic data engineering and automated model training to building a high-performance REST API and containerizing the application for cloud deployment.

## Project Architecture & Tech Stack

* **Data Engineering:** `pandas`, `numpy` (Synthetic data generation mimicking telecom telemetry)
* **Machine Learning:** `scikit-learn` (Random Forest Classifier, robust preprocessing pipelines)
* **API & Serving:** `FastAPI`, `pydantic`, `uvicorn` (Asynchronous REST API with strict schema validation)
* **Deployment & Ops:** `Docker`, `Conda`, `Git` (Containerization and environment isolation)

## How to Run Locally

### Option 1: Run via Docker (Recommended)
You do not need Python installed locally to run this, just Docker.

1. Build the container image:
   docker build -t churn-api-service .
2. Run the container:
   docker run -p 8000:8000 churn-api-service
3. Access the API Documentation: Open http://localhost:8000/docs in your browser to test the interactive Swagger UI.

### Option 2: Run via Conda Environment
Run the pipeline natively on your machine to retrain the model or modify the data source.

1. Initialize the environment:
   conda create -n churn_env python=3.12 -y
   conda activate churn_env
   pip install -r requirements.txt
2. Generate data and train the model:
   python data/generate_data.py
   python src/train.py
3. Start the API server:
   python src/app.py