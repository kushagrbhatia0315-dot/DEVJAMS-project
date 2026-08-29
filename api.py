import io
import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from fastapi import FastAPI, Response, HTTPException
from sklearn.metrics import confusion_matrix
app = FastAPI(
    title="Disaster Prediction & Analytics API",
    description="Interactive UI to view model evaluations and disaster trend charts."
)
def get_plot_bytes() -> bytes:
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close() 
    buf.seek(0)
    return buf.getvalue()
@app.get("/graphs/disasters-by-year", response_class=Response)
def plot_disasters_by_year():
    if not os.path.exists("api_disaster_data.csv"):
        raise HTTPException(status_code=404, detail="Data file not found. Run main.py first.")
    df = pd.read_csv("api_disaster_data.csv")
    year_column = "Year" 
    target_column = "Target"
    counts = df.groupby([year_column, target_column]).size().reset_index(name="Incident_Count")
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(
        data=counts,
        x=year_column,
        y="Incident_Count",
        hue=target_column,
        palette="Set1"
    )
    plt.title("Annual Disaster Frequency", fontsize=14, weight="bold", pad=15)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Number of Recorded Incidents", fontsize=12)
    plt.legend(title="Disaster Category")
    return Response(content=get_plot_bytes(), media_type="image/png")
@app.get("/data/fall-2026-forecast")
def get_2026_forecast():
    """Returns the hypothetical predicted fire causes for Fall 2026."""
    if not os.path.exists("api_2026_forecast.csv"):
        raise HTTPException(status_code=404, detail="Forecast not found. Run main.py first.")    
    df = pd.read_csv("api_2026_forecast.csv")
    return {"scenario_data": df.to_dict(orient="records")}
@app.get("/graphs/confusion-matrix", response_class=Response)
def plot_confusion_matrix():
    if not os.path.exists("api_predictions.csv"):
        raise HTTPException(status_code=404, detail="Predictions file not found. Run main.py first.")
    df = pd.read_csv("api_predictions.csv")
    y_test = df['y_test']
    predictions = df['predictions']
    matrix = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Oranges", cbar=False)
    plt.title("Wildfire & Storm Prediction Matrix", fontsize=13, pad=15, weight="bold")
    plt.xlabel("Predicted Cause", fontsize=11)
    plt.ylabel("Actual Cause", fontsize=11)
    return Response(content=get_plot_bytes(), media_type="image/png")
@app.get("/")
def root():
    return {"message": "API is running. Visit /docs for the UI."}
