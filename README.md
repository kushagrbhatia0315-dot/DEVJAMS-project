# 🔥 Wildfire & Storms AI Predictor

An end-to-end Machine Learning pipeline and interactive web application that predicts the causes of US wildfires by merging historical fire databases with extreme weather and storm records. 

This project features an automated ML pipeline, a FastAPI backend for data serving, and a Streamlit frontend for interactive geographic mapping and predictive forecasting.

---

## 🌟 Key Features

* **Automated Data Pipeline:** Extracts, cleans, and merges 24 years of SQLite wildfire records with localized CSV storm data.
* **Advanced Feature Engineering:** Utilizes spatial geo-binning, cyclic temporal encoding (sine/cosine for seasonality), and weekend-effect tracking to translate raw coordinates and dates into mathematical patterns.
* **Machine Learning Forecaster:** Uses an XGBoost Classification algorithm optimized with sample weighting to predict rare fire causes (e.g., Arson, Lightning, Campfires) based on weather and human activity trends.
* **Interactive UI (Streamlit):** Features a live geographic disaster map, dynamic confusion matrices, and interactive data tables.
* **RESTful API (FastAPI):** Exposes endpoints to retrieve generated data, evaluation metrics, and rendered charts.
* **Future Scenario Modeling:** Generates synthesized predictive scenarios for Fall 2026 based on state-by-state historical medians.

---

## 🗂️ Project Structure

```text
project-devjams/
│
├── FPA_FOD_20170508.sqlite      # SQLite Wildfire Database (Must be in root!)
├── app.py                       # Streamlit interactive web application
├── main.py                      # Master execution script for the ML pipeline
├── api.py                       # FastAPI server for endpoints & chart serving
├── requirements.txt             # Python dependencies
│
├── data/                      
│   └── Storms 1996-2019/        # Folder containing NOAA Storm CSV files
│
├── outputs/                     # Generated charts and artifacts
│
├── models/                      # Saved ML models (e.g., model.pkl, encoders)
│
└── src/                         # Core Pipeline Modules
    ├── config.py                # Global variables and file paths
    ├── ingestor.py              # Data extraction and SQL queries
    ├── engineer.py              # Data cleaning and feature engineering
    ├── modeler.py               # XGBoost model training and forecasting
    └── evaluator.py             # Accuracy metrics and visualization plotting
