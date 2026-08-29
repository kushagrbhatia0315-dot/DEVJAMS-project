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

## 🖥️ Demo

*(Add a screenshot or GIF of the Streamlit dashboard / geographic map here)*

```
outputs/demo_screenshot.png
```

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

1. **Clone the repository**
   ```bash
   git clone https://github.com/kushagrbhatia0315-dot/DEVJAMS-project.git
   cd DEVJAMS-project
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add required data**
   - Place `FPA_FOD_20170508.sqlite` in the project root.
   - Place NOAA storm CSV files in `data/Storms 1996-2019/`.
   - *(Add a link or instructions here for where to download these datasets.)*

---

## 🚀 Usage

**1. Run the ML pipeline** (ingest → engineer → train → evaluate)
```bash
python main.py
```

**2. Launch the Streamlit app**
```bash
streamlit run app.py
```

**3. Start the FastAPI server**
```bash
uvicorn api:app --reload
```
API docs will be available at `http://localhost:8000/docs`.

---

## 🧠 Methodology

The model merges two data sources — historical wildfire records and NOAA storm events — using spatial and temporal alignment (geo-bins and date windows). Weather conditions around each fire event (e.g., nearby storm activity, seasonality, day-of-week patterns) are engineered into features that feed an XGBoost classifier trained to predict the most likely **cause** of a fire (lightning, arson, campfire, equipment, etc.).

Sample weighting is used to address class imbalance, since some fire causes (e.g., arson) are far rarer in the historical record than others.

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Accuracy | *TBD* |
| F1 (weighted) | *TBD* |
| Top predicted cause classes | *TBD* |

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
-Kushagr Bhatia   (26BIT0109)

-Prince Choudhary (26BIT0107)
-
-
