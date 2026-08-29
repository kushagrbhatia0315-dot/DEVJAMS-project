import streamlit as st
import pandas as pd
import subprocess
from src import config
from src.ingestor import extract_wildfires, extract_storms
from src.engineer import merge_and_engineer, prepare_for_ml
from src.modeler import train_and_predict, predict_fall_2026_scenario
from src.evaluator import evaluate_and_plot, plot_wildfires_vs_storms_per_year
st.set_page_config(page_title="Wildfire Predictor", layout="wide")
st.title("🔥 Wildfire & Storms AI Predictor")
st.write("This web application merges 24 years of US Storms data with 1.88 Million Wildfires to predict fire causes.")
if st.button("Run Machine Learning Pipeline"):
    with st.spinner("1. Extracting 24 years of Storm data & Wildfires..."):
        wf_df = extract_wildfires(config.WILDFIRE_DB, config.ROW_LIMIT, config.WF_FEATURES, config.WF_TARGET)
        storm_df = extract_storms(config.STORMS_DIR, config.STORM_FEATURES)
    with st.spinner("2. Merging Datasets & Engineering Math..."):
        merged_df = merge_and_engineer(wf_df, storm_df, config.WF_TARGET)
        st.subheader("📊 Live Data Table Preview")
        st.dataframe(merged_df.head(100)) 
    with st.spinner("3. Training Random Forest Algorithm..."):
        X_train, X_test, y_train, y_test = prepare_for_ml(
            merged_df, config.WF_TARGET, config.TEST_SIZE
        )
        predictions = train_and_predict(
            X_train, y_train, X_test, config.MODEL_SAVE_PATH
        )
    with st.spinner("4. Generating Web Results..."):
        fig_trend = plot_wildfires_vs_storms_per_year(wf_df, storm_df, config.BAR_GRAPH_SAVE_PATH)
        st.subheader("📈 Annual Disaster Trends")
        st.pyplot(fig_trend)
        acc, fig_matrix, matrix = evaluate_and_plot(y_test, predictions, config.PLOT_SAVE_PATH)
        st.success(f"✅ Pipeline Complete! Model Accuracy: {round(acc * 100, 2)}%")
        st.subheader("🎯 Prediction Confusion Matrix")
        st.pyplot(fig_matrix)
    with st.spinner("5. Generating Fall 2026 Forecast..."):
        st.subheader("🍂 Fall 2026 Predictive Scenarios (Sep - Nov)")
        st.write("Hypothetical scenario: 10-acre fires occurring with 5 monthly storms.") 
        forecast_df = predict_fall_2026_scenario(config.MODEL_SAVE_PATH, X_train.columns)
        st.dataframe(forecast_df, use_container_width=True)
else:
    pass
if __name__ == "__main__":
    if not st.runtime.exists():
        subprocess.run(["streamlit", "run", __file__, "--server.port", "8000"])
