import streamlit as st
import pandas as pd
import subprocess
from src import config
from src.ingestor import extract_wildfires, extract_storms
from src.engineer import merge_and_engineer, prepare_for_ml
from src.modeler import train_and_predict
from src.evaluator import evaluate_and_plot, plot_wildfires_vs_storms_per_year

# Set up the web page
st.set_page_config(page_title="Wildfire Predictor", layout="wide")

st.title("🔥 Wildfire & Storms AI Predictor")
st.write("This web application merges 24 years of US Storms data with 1.88 Million Wildfires to predict fire causes.")

# A single button to trigger the whole pipeline
if st.button("Run Machine Learning Pipeline"):
    
    with st.spinner("1. Extracting 24 years of Storm data & Wildfires..."):
        wf_df = extract_wildfires(config.WILDFIRE_DB, config.ROW_LIMIT, config.WF_FEATURES, config.WF_TARGET)
        storm_df = extract_storms(config.STORMS_DIR, config.STORM_FEATURES)
        
        # --- NEW: Generate and show the double bar graph on the app ---
        st.subheader("📈 Annual Occurrences: Wildfires vs. Storms")
        bar_fig = plot_wildfires_vs_storms_per_year(wf_df, storm_df, config.BAR_GRAPH_SAVE_PATH)
        st.pyplot(bar_fig)
    
    with st.spinner("2. Merging Datasets & Engineering Math..."):
        merged_df = merge_and_engineer(wf_df, storm_df, config.WF_TARGET)
        
        # Display the Data Table directly on the website
        st.subheader("📊 Live Data Table Preview")
        st.dataframe(merged_df.head(100)) 
        
    with st.spinner("3. Training Random Forest Algorithm..."):
        # Removed config.RANDOM_SEED from parameters
        X_train, X_test, y_train, y_test = prepare_for_ml(
            merged_df, config.WF_TARGET, config.TEST_SIZE
        )
        # Removed config.RANDOM_SEED from parameters
        predictions = train_and_predict(
            X_train, y_train, X_test, config.MODEL_SAVE_PATH
        )
        
    with st.spinner("4. Generating Web Results..."):
        acc, fig, matrix = evaluate_and_plot(y_test, predictions, config.PLOT_SAVE_PATH)
        
        # Display the Accuracy and the Graph on the website
        st.success(f"✅ Pipeline Complete! Model Accuracy: {round(acc * 100, 2)}%")
        
        st.subheader("🎯 Prediction Confusion Matrix")
        st.pyplot(fig)

# --- THE PLAY BUTTON FIX ---
# Fixed the missing double underscores for Python magic variables
if __name__ == "__main__":
    if not st.runtime.exists():
        subprocess.run(["streamlit", "run", __file__, "--server.port", "8000"])
