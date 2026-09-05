📘 Sentiment Analysis with MLOps (FastAPI • MLflow • DVC • Docker • Neon Postgres)

An end-to-end Sentiment Analysis system built using modern MLOps practices, secure user authentication, MLflow model tracking, FastAPI APIs, Docker containerization, and a cloud Postgres database.

🚀 Features
1. Complete MLOps Pipeline

DVC-tracked dataset (raw + processed)

Data ingestion, transformation, and model training modules

Modular, production-style component architecture

Centralized logging and exception handling

2. MLflow Integration

Experiment tracking (parameters, metrics, artifacts)

Model versioning using MLflow Model Registry

During prediction, model is downloaded directly from MLflow

3. FastAPI Backend

/register → Register user

/login → Login with JWT

/predict → Sentiment prediction

Pydantic schemas for request/response validation

4. User Authentication System

Secure password hashing

JWT-based authentication

User predictions saved in database

5. Neon Postgres Cloud Database

Online cloud Postgres

Stores:

User accounts

Prediction logs

Integrated with SQLAlchemy ORM

6. Docker Support

Dockerfile for full containerization

Image published on Docker Hub

Run anywhere with one command

📂 Project Structure
.
├── Data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── api/
│   │   ├── routes/
│   │   ├── schemas/
│   │   ├── models/
│   │   ├── database/
│   │   ├── utils/
│   │   ├── app.py
│   │   └── main.py
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── model/
│   ├── pipeline/
│   │   ├── train_pipeline.py
│   │   └── predict_pipeline.py
│   ├── utils.py
│   └── exception.py
├── Dockerfile
├── docker-compose.yml
├── .dvc/
├── .env
└── README.md

🧠 How the System Works
Training Workflow

Ingest data via DVC

Apply transformations

Train ML model

Log metrics and model to MLflow

Register final model in MLflow Registry

Prediction Workflow

User sends text to FastAPI

API downloads model from MLflow (if needed)

Preprocess text

Predict sentiment

Save results to Neon Postgres

🐳 Running With Docker
Build
docker build -t sentiment-mlops .

Run
docker run -p 8000:8000 sentiment-mlops

Pull from Docker Hub
docker pull <your-dockerhub-username>/<image-name>

▶️ Running Without Docker
Install Packages
pip install -r requirements.txt

Start Application
uvicorn src.api.main:app --reload

🔐 API Endpoints
Auth
Method	Endpoint	Description
POST	/register	Register user
POST	/login	Login & get JWT
Prediction
Method	Endpoint	Description
POST	/predict	Predict sentiment
📊 Tech Stack

Python

FastAPI

MLflow

DVC

Docker

Neon Postgres

Scikit-learn

SQLAlchemy

Pydantic

## Maintainer

Maintained by [Muhammad Haseeb](https://github.com/mhaseeb04).
