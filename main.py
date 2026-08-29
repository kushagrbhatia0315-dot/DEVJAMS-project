import streamlit as st
import pandas as pd
import subprocess
from src import config
from src.ingestor import extract_wildfires, extract_storms
from src.engineer import merge_and_engineer, prepare_for_ml
from src.modeler import train_and_predict, predict_fall_2026_scenario
from src.evaluator import evaluate_and_plot, plot_wildfires_vs_storms_per_year, plot_us_disaster_map

st.set_page_config(page_title="Wildfire & Storm Predictor", layout="wide")

st.title("Wildfire & Storms AI Predictor")
st.write("This web application merges US Storms data with Wildfire records to map disasters and predict fire causes.")

if st.button("Run Machine Learning Pipeline"):
    with st.spinner("1. Extracting Storm data & Wildfires..."):
        st.session_state['wf_df'] = extract_wildfires(config.WILDFIRE_DB, config.ROW_LIMIT, config.WF_FEATURES, config.WF_TARGET)
        st.session_state['storm_df'] = extract_storms(config.STORMS_DIR, config.STORM_FEATURES)
    
    with st.spinner("2. Merging Datasets & Engineering Math..."):
        st.session_state['merged_df'] = merge_and_engineer(
            st.session_state['wf_df'], st.session_state['storm_df'], config.WF_TARGET
        )
        
    with st.spinner("3. Training Random Forest Algorithm..."):
        X_train, X_test, y_train, y_test = prepare_for_ml(
            st.session_state['merged_df'], config.WF_TARGET, config.TEST_SIZE
        )
        predictions = train_and_predict(X_train, y_train, X_test, config.MODEL_SAVE_PATH)
        st.session_state['X_train_cols'] = X_train.columns
        st.session_state['y_test'] = y_test
        st.session_state['predictions'] = predictions

    with st.spinner("4. Generating Web Results & Forecasts..."):
        acc, fig_matrix, matrix = evaluate_and_plot(y_test, predictions, config.PLOT_SAVE_PATH)
        st.session_state['acc'] = acc
        st.session_state['fig_matrix'] = fig_matrix
        st.session_state['fig_trend'] = plot_wildfires_vs_storms_per_year(
            st.session_state['wf_df'], st.session_state['storm_df'], config.BAR_GRAPH_SAVE_PATH
        )
        
        st.session_state['forecast_df'] = predict_fall_2026_scenario(
            config.MODEL_SAVE_PATH, X_train.columns, st.session_state['merged_df']
        )
if 'merged_df' in st.session_state:
    st.success(f"✅ Pipeline Active! Model Accuracy: {round(st.session_state['acc'] * 100, 2)}%")
    
    st.subheader("🗺️ Interactive US Disaster Map")
    st.write("Select a year to display Wildfires (Red) and Storms (White) geographically.")
    
    min_year = int(st.session_state['wf_df']['FIRE_YEAR'].min())
    max_year = int(st.session_state['wf_df']['FIRE_YEAR'].max())
    selected_year = st.slider("Select Mapping Year", min_value=min_year, max_value=max_year, value=max_year)
    
    map_fig = plot_us_disaster_map(st.session_state['wf_df'], st.session_state['storm_df'], selected_year)
    st.plotly_chart(map_fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Annual Disaster Trends")
        st.pyplot(st.session_state['fig_trend'])
    with col2:
        st.subheader("🎯 Prediction Confusion Matrix")
        st.pyplot(st.session_state['fig_matrix'])

    st.subheader("📊 Live Data Table Preview")
    st.dataframe(st.session_state['merged_df'].head(100), use_container_width=True)
    
    st.subheader(" Fall 2026 Predictive Scenarios (Sep - Nov)")
    st.write("Forecasted fire causes based on state-specific historical median fire sizes and storm counts.")
    st.dataframe(st.session_state['forecast_df'], use_container_width=True)
if __name__ == "__main__":
    if not st.runtime.exists():
        subprocess.run(["streamlit", "run", __file__, "--server.port", "8000"])
