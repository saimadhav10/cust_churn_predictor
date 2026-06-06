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
   ```bash
   docker build -t churn-api-service .