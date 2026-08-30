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

```text
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
    ├── ingestor.py              # Data extraction and SQL queries
    ├── engineer.py              # Data cleaning and feature engineering
    ├── modeler.py               # XGBoost model training and forecasting
    └── evaluator.py             # Accuracy metrics and visualization plotting
```

---

## 🚀 Usage & Judge Evaluation (Google Colab)

 Open a blank [Google Colab Notebook](https://colab.research.google.com/) and run this **all-in-one cell** to download the code, train the XGBoost model, and expose the Streamlit UI to the web:

```python
# 1. Clone Repo & Install
!rm -rf /content/DEVJAMS-project
!git clone https://github.com/kushagrbhatia0315-dot/DEVJAMS-project.git
%cd /content/DEVJAMS-project
!pip install -r requirements.txt

# 2. Download Data
!python setup_data.py

# 3. Train the Model
!python main.py

# 4. Launch the App (Cloudflare Tunnel - NO PASSWORD NEEDED)
!wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
!chmod +x cloudflared-linux-amd64

# Start Streamlit with WebSocket security disabled so data can flow!
!streamlit run app.py --server.enableCORS false --server.enableXsrfProtection false --server.enableWebsocketCompression false &>/content/logs.txt &

# Start Cloudflare Tunnel and grab the URL
import time
!nohup ./cloudflared-linux-amd64 tunnel --url http://localhost:8501 &> /content/tunnel.txt &
time.sleep(5)
print("\n🚨 CLICK THIS LINK TO VIEW YOUR APP:\n")
!grep -o 'https://.*\.trycloudflare.com' /content/tunnel.txt
```

*(Once the pipeline finishes, click the generated `loca.lt` link at the bottom and paste the password to view the live dashboard!)*

---

## 💻 Local Evaluation (Run All At Once)

Copy and paste this single block into  terminal to download, setup, and train the model all at once:

*(🍎 Mac Users: Please run `brew install libomp` before executing, as XGBoost requires it).*

```bash
git clone [https://github.com/kushagrbhatia0315-dot/DEVJAMS-project.git](https://github.com/kushagrbhatia0315-dot/DEVJAMS-project.git)
cd DEVJAMS-project
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python setup_data.py
python main.py
```

**Launch the Dashboards:**
Once the model finishes training, open two separate terminal windows inside the project folder to start the servers:
*   **Terminal 1 (UI):** `streamlit run app.py` 
*   **Terminal 2 (API):** `uvicorn api:app --reload`

---

## 🧠 Methodology

The model merges two data sources — historical wildfire records and NOAA storm events — using spatial and temporal alignment (geo-bins and date windows). Weather conditions around each fire event (e.g., nearby storm activity, seasonality, day-of-week patterns) are engineered into features that feed an XGBoost classifier trained to predict the most likely **cause** of a fire (lightning, arson, campfire, equipment, etc.).

Sample weighting is used to address class imbalance, since some fire causes (e.g., arson) are far rarer in the historical record than others.

---

## 📊 Model Performance

Live model evaluation metrics, including real-time accuracy scores and an interactive confusion matrix, are generated dynamically and visualized directly within the Streamlit dashboard.

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
**Kushagr Bhatia** (26BIT0109)  
**Prince Choudhary** (26BIT0107)
**S. Meenatchi Sundaram** (26BCE0038)
**Saran Sathish Kumar** (26BIT0218)
