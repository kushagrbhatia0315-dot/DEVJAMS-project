import streamlit as st
import os
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import subprocess
from src import config
from src.ingestor import extract_wildfires, extract_storms
from src.engineer import merge_and_engineer, prepare_for_ml
from src.modeler import train_and_predict, predict_fall_2026_scenario
from src.evaluator import evaluate_and_plot, plot_wildfires_vs_storms_per_year, plot_us_disaster_map, create_gauge

st.set_page_config(page_title="Wildfire & Storm Predictor", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    /* Catchy Bright Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%);
    }
    /* Convert text to dark colors for readability */
    html, body, p, h1, h2, h3, h4, h5, h6, span, div {
        color: #102A43 !important; 
    }
    /* Style Tabs to Pop */
    .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(255, 255, 255, 0.6);
        border-radius: 12px;
        padding: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 700 !important;
    }
    /* Make DataFrames bright */
    .stDataFrame {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        padding: 10px;
    }
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.7) !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_and_prep_data():
    wf_df = extract_wildfires(config.WILDFIRE_DB, config.ROW_LIMIT, config.WF_FEATURES, config.WF_TARGET)
    storm_df = extract_storms(config.STORMS_DIR, config.STORM_FEATURES)
    merged_df = merge_and_engineer(wf_df, storm_df, config.WF_TARGET)
    return wf_df, storm_df, merged_df

@st.cache_resource(show_spinner=False)
def run_model_pipeline(_merged_df):
    X_train, X_test, y_train, y_test = prepare_for_ml(_merged_df, config.WF_TARGET, config.TEST_SIZE)
    predictions = train_and_predict(X_train, y_train, X_test, config.MODEL_SAVE_PATH)
    acc, fig_matrix, matrix = evaluate_and_plot(y_test, predictions, config.PLOT_SAVE_PATH)
    forecast_df = predict_fall_2026_scenario(config.MODEL_SAVE_PATH, X_train.columns, _merged_df)
    return acc, fig_matrix, forecast_df

with st.spinner("Initializing Datasets & AI Model..."):
    wf_df, storm_df, merged_df = load_and_prep_data()
    acc, fig_matrix, forecast_df = run_model_pipeline(merged_df)

with st.sidebar:
    st.title("⚙️ Dashboard Controls")
    st.markdown("---")
    min_year = int(wf_df['FIRE_YEAR'].min())
    max_year = int(wf_df['FIRE_YEAR'].max())
    selected_year = st.slider("Select Timeline Year", min_value=min_year, max_value=max_year, value=max_year)
    
    st.markdown("---")
    st.write("🔄 **System Status:** Active & Cached")

st.title("🔥 US Disaster AI & Spatial Analytics")
st.write("Visualizing the relationship between extreme weather and historical wildfire patterns in a vibrant workspace.")

fires_this_year = len(wf_df[wf_df['FIRE_YEAR'] == selected_year])
storms_this_year = len(storm_df[storm_df['YEAR'] == selected_year])
max_fires = wf_df.groupby('FIRE_YEAR').size().max()
max_storms = storm_df.groupby('YEAR').size().max()

col1, col2, col3 = st.columns(3)
with col1:
    st.plotly_chart(create_gauge(fires_this_year, max_fires, "Total Wildfires", "#ff4b4b"), use_container_width=True)
with col2:
    st.plotly_chart(create_gauge(storms_this_year, max_storms, "Total Storms", "#17a3f4"), use_container_width=True)
with col3:
    st.plotly_chart(create_gauge(acc*100, 100, "AI Accuracy (%)", "#82ca9d"), use_container_width=True)

st.markdown("---")

tab1, tab2, tab3, tab4,tab5 = st.tabs([
    "🎯 AI Confusion Matrix", 
    "🗺️ USA Boundary Map", 
    "📈 Analytics & Analytics", 
    "🔮 Fall 2026 Forecast",
    "🎮 Live Predictor"
])

with tab1:
    st.markdown("### Model Performance Breakdown")
    st.write("This matrix evaluates how accurately the AI identifies specific wildfire causes based strictly on weather/storm patterns.")
    st.plotly_chart(fig_matrix, use_container_width=True)

with tab2:
    st.markdown(f"### Regional Spread with State Boundaries ({selected_year})")
    map_fig = plot_us_disaster_map(wf_df, storm_df, selected_year)
    st.plotly_chart(map_fig, use_container_width=True)

with tab3:
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.markdown("### Historical Event Frequency")
        trend_fig = plot_wildfires_vs_storms_per_year(wf_df, storm_df, config.BAR_GRAPH_SAVE_PATH)
        st.plotly_chart(trend_fig, use_container_width=True)
    with col_right:
        st.markdown("### Raw Merged Feature Set")
        st.dataframe(merged_df.head(500), use_container_width=True, height=400)

with tab4:
    st.markdown("### 🍂 Predictive AI Scenarios (Sep - Nov 2026)")
    st.write("Based on state-by-state median weather histories, here is what the Random Forest algorithm predicts the highest probability fire causes will be next Fall.")
    st.dataframe(forecast_df, use_container_width=True, hide_index=True)

with tab5:
    st.markdown("### 🎮 Interactive Wildfire Cause Predictor")
    st.write("Adjust the environmental factors below to see what the AI predicts as the most likely cause.")

    @st.cache_resource(show_spinner=False)
    def load_cached_model():
        if os.path.exists(config.MODEL_SAVE_PATH):
            return joblib.load(config.MODEL_SAVE_PATH)
        return None

    artifacts = load_cached_model()
    
    if artifacts:
        model = artifacts['model']
        le = artifacts['encoder']
     
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            available_states = sorted(merged_df['STATE'].dropna().unique())
            input_state = st.selectbox("State", available_states, index=available_states.index('CA') if 'CA' in available_states else 0)
        with col_b:
            input_month = st.selectbox("Month", list(range(1, 13)), index=7) # Default to August (8)
        with col_c:
            input_fire_size = st.number_input("Fire Size (Acres)", min_value=0.1, max_value=500000.0, value=100.0, step=50.0)
        with col_d:
            input_storms = st.number_input("Monthly Storms in State", min_value=0, max_value=500, value=15, step=1)
            
        if st.button("🚀 Run AI Prediction", use_container_width=True):
            state_data = merged_df[merged_df['STATE'] == input_state]
            med_lat = state_data['LAT_ROUNDED'].median() if not state_data.empty else 39.8
            med_lon = state_data['LON_ROUNDED'].median() if not state_data.empty else -98.6

            input_df = pd.DataFrame([{
                'MONTH': input_month,
                'MONTH_SIN': np.sin(2 * np.pi * input_month / 12.0),
                'MONTH_COS': np.cos(2 * np.pi * input_month / 12.0),
                'STATE': input_state,
                'LOG_FIRE_SIZE': np.log1p(input_fire_size), 
                'MONTHLY_STORMS': input_storms,
                'LAT_ROUNDED': med_lat,
                'LON_ROUNDED': med_lon
            }])
 
            X_input_raw = input_df.drop(columns=['MONTH'], errors='ignore')
            X_input = pd.get_dummies(input_df)
            
            expected_cols = model.feature_names_in_
            X_input = X_input.reindex(columns=expected_cols, fill_value=0)
     
            pred_encoded = model.predict(X_input)[0]
            predicted_cause = le.inverse_transform([pred_encoded])[0]
            probs = model.predict_proba(X_input)[0]
 
            prob_df = pd.DataFrame({
                'Cause': le.inverse_transform(range(len(probs))),
                'Probability (%)': probs * 100
            }).sort_values('Probability (%)', ascending=False).head(4)
 
            st.success(f"### 🎯 Top Predicted Cause: **{predicted_cause}**")
            
            fig_probs = px.bar(
                prob_df, 
                x='Probability (%)', 
                y='Cause', 
                orientation='h',
                color='Probability (%)',
                color_continuous_scale='Teal',
                title="AI Confidence Breakdown"
            )
            fig_probs.update_layout(
                yaxis={'categoryorder':'total ascending'}, 
                plot_bgcolor='rgba(255,255,255,0.5)', 
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#102A43')
            )
            st.plotly_chart(fig_probs, use_container_width=True)
            
    else:
        st.warning("⚠️ Model not found! Please run `main.py` first to generate `rf_model_v2.pkl`.")

if __name__ == "__main__":
    if not st.runtime.exists():
        subprocess.run(["streamlit", "run", __file__, "--server.port", "8501"])

