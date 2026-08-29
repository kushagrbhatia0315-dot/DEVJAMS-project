import os
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

def train_and_predict(X_train, y_train, X_test, save_path: str):
    """Trains the XGBoost algorithm and saves it and the encoder to disk."""
    print("--> [4/5] Training XGBoost with Engineered Data...")
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    model = XGBClassifier(
        n_estimators=30, 
        max_depth=8, 
        learning_rate=0.1, 
        n_jobs=-1, 
        random_state=42,
        eval_metric='mlogloss',
        verbosity=1
    )
    model.fit(X_train, y_train_encoded, sample_weight=sample_weights)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump({'model': model, 'encoder': le}, save_path)
    predictions_encoded = model.predict(X_test)
    predictions_strings = le.inverse_transform(predictions_encoded)
    
    return predictions_strings

def predict_fall_2026_scenario(save_path: str, training_columns: list, merged_df: pd.DataFrame):
    """Generates Fall 2026 forecasts using real historical state/month metrics."""
    print("--> Forecasting Fall 2026 Scenarios...")
    artifacts = joblib.load(save_path)
    model = artifacts['model']
    le = artifacts['encoder']
    states = ['CA', 'TX', 'FL', 'OR', 'WA', 'CO', 'AZ']
    months = [9, 10, 11]
    scenarios = []
    state_month_stats = merged_df.groupby(['STATE', 'MONTH']).agg({
        'LOG_FIRE_SIZE': 'median',
        'MONTHLY_STORMS': 'mean',
        'LAT_ROUNDED': 'median',
        'LON_ROUNDED': 'median'
    }).reset_index()
    
    overall_fire_size = merged_df['LOG_FIRE_SIZE'].median()
    overall_storms = merged_df['MONTHLY_STORMS'].mean()
    overall_lat = merged_df['LAT_ROUNDED'].median()
    overall_lon = merged_df['LON_ROUNDED'].median()
    for state in states:
        for month in months:
            match = state_month_stats[(state_month_stats['STATE'] == state) & (state_month_stats['MONTH'] == month)]
            if not match.empty:
                scenarios.append({
                    'MONTH': month,
                    'MONTH_SIN': np.sin(2 * np.pi * month / 12.0),
                    'MONTH_COS': np.cos(2 * np.pi * month / 12.0),
                    'STATE': state,
                    'LOG_FIRE_SIZE': match['LOG_FIRE_SIZE'].values[0],
                    'MONTHLY_STORMS': match['MONTHLY_STORMS'].values[0],
                    'LAT_ROUNDED': match['LAT_ROUNDED'].values[0],
                    'LON_ROUNDED': match['LON_ROUNDED'].values[0]
                })
            else:
                scenarios.append({
                    'MONTH': month,
                    'MONTH_SIN': np.sin(2 * np.pi * month / 12.0),
                    'MONTH_COS': np.cos(2 * np.pi * month / 12.0),
                    'STATE': state,
                    'LOG_FIRE_SIZE': overall_fire_size,
                    'MONTHLY_STORMS': overall_storms,
                    'LAT_ROUNDED': overall_lat,
                    'LON_ROUNDED': overall_lon
                })
                
    scenario_df = pd.DataFrame(scenarios)
    X_features = scenario_df.drop(columns=['LOG_FIRE_SIZE', 'MONTHLY_STORMS', 'LAT_ROUNDED', 'LON_ROUNDED', 'MONTH_SIN', 'MONTH_COS', 'MONTH'], errors='ignore')
    X_scenario = pd.get_dummies(X_features)
    X_scenario['LOG_FIRE_SIZE'] = scenario_df['LOG_FIRE_SIZE']
    X_scenario['MONTHLY_STORMS'] = scenario_df['MONTHLY_STORMS']
    X_scenario['LAT_ROUNDED'] = scenario_df['LAT_ROUNDED']
    X_scenario['LON_ROUNDED'] = scenario_df['LON_ROUNDED']
    X_scenario['MONTH_SIN'] = scenario_df['MONTH_SIN']
    X_scenario['MONTH_COS'] = scenario_df['MONTH_COS']
    X_scenario = X_scenario.reindex(columns=training_columns, fill_value=0)
    preds_encoded = model.predict(X_scenario)
    scenario_df['PREDICTED_CAUSE'] = le.inverse_transform(preds_encoded)
    
    return scenario_df
