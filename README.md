🌫️ AQI Forecasting System — 3-Day Air Quality Prediction with Automated Data + ML Pipelines

🌫️ Air Quality Index (AQI) Prediction System
End-to-End Machine Learning Pipeline | Feast Feature Store | S3 Automation | FastAPI + Streamlit

This project is a complete AQI forecasting system designed to fetch, clean, store, and model air-quality data using a modern ML pipeline.
It includes data ingestion, feature engineering, model training, automated pipelines, and a web interface for real-time predictions.

🚀 Project Overview

The AQI Prediction System uses hourly environmental data (PM2.5, PM10, CO, SO₂, NO₂, temperature, humidity, wind, etc.) to predict future AQI levels for Karachi.

The system includes:

Automated data ingestion (live data fetched hourly)

Feature engineering pipeline using Feast Feature Store

ML model training & evaluation (RandomForest / XGBoost)

S3 integration for storing raw data, cleaned features, and trained models

FastAPI backend for predictions

Streamlit frontend for visualization

Full automation pipeline for daily updates

🏗️ Architecture
Live API → fetch_live_khi.py → Raw CSV → S3
                          ↓
            automations/automate_pipeline_khi.py
                          ↓
     clean_feature_engineering.py → Parquet features → S3
                          ↓
             Feast Registry → Model Training
                          ↓
        Trained Model.pkl → S3 → FastAPI Endpoint
                          ↓
                   Streamlit Dashboard
