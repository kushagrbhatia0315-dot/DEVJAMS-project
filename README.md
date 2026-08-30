# 🔥 Wildfire & Storms AI Predictor

An end-to-end Machine Learning pipeline and interactive web application that predicts the causes of US wildfires by merging 24 years of historical fire records with extreme weather and storm data.

The project combines an automated ML pipeline, a **FastAPI** backend for serving data and metrics, and a **Streamlit** frontend for interactive geographic mapping and predictive forecasting.

---

## 🌟 Key Features

- **Automated Data Pipeline** — Extracts, cleans, and merges 24 years of SQLite wildfire records with localized NOAA storm CSV data.
- **Advanced Feature Engineering** — Spatial geo-binning, cyclic temporal encoding (sine/cosine for seasonality), and weekend-effect tracking to turn raw coordinates and dates into meaningful ML features.
- **Machine Learning Forecaster** — XGBoost classifier optimized with sample weighting to predict rare fire causes (e.g., Arson, Lightning, Campfires) from weather and human activity trends.
- **Interactive UI (Streamlit)** — Live geographic disaster map, dynamic confusion matrices, and interactive data tables.
- **RESTful API (FastAPI)** — Endpoints to retrieve generated data, evaluation metrics, and rendered charts.
- **Future Scenario Modeling** — Synthesized predictive scenarios for Fall 2026 based on state-by-state historical medians.

---

## 🗂️ Project Structure

```
project-devjams/
│
├── FPA_FOD_20170508.sqlite      # SQLite Wildfire Database (must be in root!)
├── app.py                       # Streamlit interactive web application
├── main.py                      # Master execution script for the ML pipeline
├── api.py                       # FastAPI server for endpoints & chart serving
├── setup_data.py                # Data setup / download helper script
├── requirements.txt             # Python dependencies
│
├── data/
│   └── Storms 1996-2019/        # NOAA Storm CSV files
│
├── outputs/                     # Generated charts and artifacts
│
├── models/                      # Saved ML models (e.g., model.pkl, encoders)
│
└── src/                         # Core pipeline modules
    ├── config.py                # Global variables and file paths
    ├── ingestor.py               # Data extraction and SQL queries
    ├── engineer.py                # Data cleaning and feature engineering
    ├── modeler.py                # XGBoost model training and forecasting
    └── evaluator.py               # Accuracy metrics and visualization plotting
```

---

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone [https://github.com/kushagrbhatia0315-dot/DEVJAMS-project.git](https://github.com/kushagrbhatia0315-dot/DEVJAMS-project.git)
cd DEVJAMS-project
```

**2. Create a virtual environment (recommended for local use)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
pip install huggingface_hub
```

**4. Download required data (Automated)**
No manual downloading needed! Run this python snippet to safely fetch the datasets directly from Hugging Face into the correct folders:
```python
from huggingface_hub import hf_hub_download, snapshot_download

hf_hub_download(repo_id="kushagrbhatia03/wildfire-storm-data", repo_type="dataset", filename="FPA_FOD_20170508.sqlite", local_dir=".")
snapshot_download(repo_id="kushagrbhatia03/wildfire-storm-data", repo_type="dataset", allow_patterns="Storms 1996-2019/*", local_dir="./data")
```
---
## 🚀 Usage & Judge Evaluation (Google Colab)

For hackathon judges, the easiest way to evaluate this project end-to-end is via Google Colab. Open a blank [Google Colab Notebook](https://colab.research.google.com/) and run this **all-in-one cell** to download the code, train the XGBoost model, and expose the Streamlit UI to the web:

```python
# 1. Clone Repo & Install
!git clone https://github.com/kushagrbhatia0315-dot/DEVJAMS-project.git
%cd /content/DEVJAMS-project
!pip install -r requirements.txt huggingface_hub

# 2. Download Data
import os
from huggingface_hub import hf_hub_download, snapshot_download
hf_hub_download(repo_id="kushagrbhatia03/wildfire-storm-data", repo_type="dataset", filename="FPA_FOD_20170508.sqlite", local_dir=".")
snapshot_download(repo_id="kushagrbhatia03/wildfire-storm-data", repo_type="dataset", allow_patterns="Storms 1996-2019/*", local_dir="./data")

# 3. Train the Model
!python main.py

# 4. Launch the App
import urllib
print("\n🚨 LOCALTUNNEL PASSWORD:", urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip("\n"))
!npm install -g localtunnel
!streamlit run app.py &>/content/logs.txt &
!npx localtunnel --port 8501
```
---

## 🧠 Methodology

The model merges two data sources — historical wildfire records and NOAA storm events — using spatial and temporal alignment (geo-bins and date windows). Weather conditions around each fire event (e.g., nearby storm activity, seasonality, day-of-week patterns) are engineered into features that feed an XGBoost classifier trained to predict the most likely **cause** of a fire (lightning, arson, campfire, equipment, etc.).

Sample weighting is used to address class imbalance, since some fire causes (e.g., arson) are far rarer in the historical record than others.

---

## 📊 Model Performance

"Live model evaluation metrics, including real-time accuracy scores and an interactive confusion matrix, are generated dynamically and visualized directly within the Streamlit dashboard."
*(Populate this table with your actual evaluator.py output, and consider embedding a confusion matrix image from `outputs/`.)*

---

## 🔮 Future Scenario Modeling

The pipeline includes a module to generate synthesized fire-risk scenarios for **Fall 2026**, based on state-by-state historical medians of weather and fire-cause patterns — useful for early-warning style forecasting rather than only historical analysis.

---

## 🛣️ Roadmap

- [ ] Add automated tests (`pytest`) for pipeline stages
- [ ] Pin dependency versions in `requirements.txt`
- [ ] Add live weather API integration for real-time predictions
- [ ] Deploy Streamlit app (e.g., Streamlit Community Cloud) for public demo
- [ ] Expand scenario modeling to storm severity, not just fire cause

---

## 👥 Team / Contributor
 Kushagr Bhatia   (26BIT0109)
 Prince Choudhary (26BIT0107)
-
-
