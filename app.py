import streamlit as st
import pandas as pd
from src import config
from src.ingestor import extract_wildfires,extract_storms
from src.engineer import merge_and_engineer,prepare_for_ml
from src.modeler import train_and_predict
from src.evaluator import evaluate_and_plot

st.set_page_config(page_title="Wildfire AI Predictor",layout="wide")
st.title("Wildfire & Strom AI Predictor")
st.write("This web application merges 24 years of US Storms data with 1.88 MIllion Wildfires to predict fire causes.")
if st.button("Run Machine Learning Pipeline"):
   with st.spinner("1. Extracting 24 years of Storm data & Wildfires..."):
        wf_df = extract_wildfires(config.WILDFIRE_DB, config.ROW_LIMIT, config.WF_FEATURES, config.WF_TARGET)
        storm_df = extract_storms(config.STORMS_DIR, config.STORM_FEATURES)
   with st.spinner("2. Merging Datasets & Engineering Math..."):
        merged_df = merge_and_engineer(wf_df, storm_df, config.WF_TARGET)
        st.subheader("Live Data Table Preview")
        st.dataframe(merged_df.head(100)) 
   with st.spinner("3. Training Random Forest Algorithm..."):
        X_train, X_test, y_train, y_test = prepare_for_ml(merged_df, config.WF_TARGET, config.TEST_SIZE, config.RANDOM_SEED)
        predictions = train_and_predict(X_train, y_train, X_test, config.RANDOM_SEED, config.MODEL_SAVE_PATH)   
   with st.spinner("4. Generating Web Results..."):
        acc, fig, matrix = evaluate_and_plot(y_test, predictions, config.PLOT_SAVE_PATH)
        st.success(f" Pipeline Complete! Model Accuracy: {round(acc * 100, 2)}%") 
        st.subheader("Prediction Confusion Matrix")
        st.pyplot(fig)
